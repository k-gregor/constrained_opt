import numpy as np
from enum import Enum
import pandas as pd

import pandas_helper as ph
from helper.chunk_filter import ChunkFilter
from compute_entropy import compute_entropy
from constants import months

to_t_dry_biomass_per_ha = 10 / 0.47  # convert kgC/m2 to t dry mass / ha @McGroddy2004


from pft_to_forest_type import *

class NoManagementFeasibleError(Exception):
    pass


SLOW_POOL_DECAY_RATE = 24 / 25

MIN_FPC_DEF_FOREST = 0.1


class OptimizationType(Enum):
    SINGLE_RCP = 1
    MINMAX = 2
    ZEROMAX = 3
    ZSCORE = 4


def normalize_for_uncertainty_scenarios(rcps, scores, feasible_managements=None, lower_is_better=False):
    # first, convert the scores map of the current variable (simulation --> [value_rcp1, value_rcp2, ...] )
    # to rcp --> [value_simulation1, value_simulation_2]
    # because we want to normalize the values inside an rcp. note that values across rcps should not be mixed, they are the different points in our uncertainty space.
    normalized_scores_per_rcp = {}
    for rcp in rcps:
        normalized_scores_per_rcp[rcp] = []

    for simulation in scores.keys():
        if simulation in feasible_managements:
            for idx, rcp in enumerate(rcps):
                normalized_scores_per_rcp[rcp].append(scores[simulation][idx])

    # normalize for each rcp separately because they are disjoint futures and should not be mixed together.
    for rcp in rcps:

        # Unlike python3, python2 will still do integer division, e.g. 1/2=0 so just to avoid anyone using this with python2 and getting strange results, cast one variable to float.
        max_score = float(np.max(normalized_scores_per_rcp[rcp]))
        min_score = np.min(normalized_scores_per_rcp[rcp])

        if lower_is_better:
            if min_score == max_score:
                normalized_scores_per_rcp[rcp] = np.array(normalized_scores_per_rcp[rcp]) * 0
            else:
                normalized_scores_per_rcp[rcp] = (max_score - np.array(normalized_scores_per_rcp[rcp])) / (max_score - min_score)
        else:
            if min_score == max_score:
                normalized_scores_per_rcp[rcp] = np.array(normalized_scores_per_rcp[rcp]) * 0
            else:
                normalized_scores_per_rcp[rcp] = (np.array(normalized_scores_per_rcp[rcp]) - min_score) / (max_score - min_score)


        for idx, simulation in enumerate(scores.keys()):
            if simulation not in feasible_managements:
                normalized_scores_per_rcp[rcp] = np.insert(normalized_scores_per_rcp[rcp], idx, 0.0)

    # at this point, we have normalizes_scores_per_rcp, so 1 array of length n_similations of normalized values per rcp.
    return normalized_scores_per_rcp


def get_mean_over_rcps(dataframe: pd.DataFrame):
    return dataframe.mean()


def get_1sem_worst_case(dataframe: pd.DataFrame):
    return dataframe.mean() - dataframe.sem()


def aggregation_mean_lambda(min_year, max_year):
    return lambda dataframe, field: aggregation_mean(dataframe, field, min_year, max_year)


def aggregation_mean(dataframe: pd.DataFrame, field, min_year, max_year):
    return dataframe[field].loc[(dataframe['Year'] >= min_year) & (dataframe['Year'] <= max_year)].mean()




class Gc:

    def __init__(self, lon, lat, simulation_names, boundary_simulations, used_variables, rcps):
        self.lon = lon
        self.lat = lat

        self.et = {}
        self.swp = {}
        self.harvest = {}
        self.hlp = {}
        self.albedo_jul = {}
        self.albedo_jan = {}
        self.csequestration = {}
        self.vegc = {}
        self.biodiversity_cwd = {}
        self.biodiversity_big_trees = {}
        self.biodiversity_size_diversity = {}
        self.surface_roughness = {}
        self.forest_fpc = {}
        self.mitigation = {}

        for idx, simulation in enumerate(simulation_names + boundary_simulations):
            self.et[simulation] = [] # array, one entry per rcp
            self.swp[simulation] = []
            self.harvest[simulation] = []
            self.hlp[simulation] = []
            self.csequestration[simulation] = []
            self.albedo_jul[simulation] = []
            self.albedo_jan[simulation] = []
            self.vegc[simulation] = []
            self.biodiversity_cwd[simulation] = []
            self.biodiversity_big_trees[simulation] = []
            self.biodiversity_size_diversity[simulation] = []
            self.surface_roughness[simulation] = []
            self.forest_fpc[simulation] = []
            self.mitigation[simulation] = []
            self.feasible_managements = []
            self.feasible_managements_ids = []
            self.infeasible_forest_types = set()
            self.infeasible_managements_ids = []
            self.no_management_feasible = False

            self.all_scores_dict_u = {}
            self.all_scores_dict_raw = {}
            self.add_constraint_base_values = None
            self.additional_constraints = None

            self.es_vals_u = np.zeros((len(used_variables) * len(rcps), len(simulation_names)))


def get_es_vals_new_all_gcs(rcps,
                            simulation_paths,
                            simulation_names,
                            lon_lats_of_interest,
                            used_variables,
                            future_year1,
                            future_year2,
                            optimizationType=OptimizationType.MINMAX,
                            boundary_simulations=[],
                            rounding=False,
                            variables_for_additional_constraints=[],
                            discounting=True, only_forest_swp=False):

    compute_biodiv = True in (ele.startswith('biodiv') for ele in used_variables)
    aggregation = aggregation_mean_lambda(future_year1, future_year2)

    gc_data = {}

    for man_idx, simulation in enumerate(simulation_names + boundary_simulations):
        for rcp in rcps:
            basepath = simulation_paths[rcp][simulation]

            print('Read data for', simulation, 'and', rcp)
            # need to read from 1990 to get present value too for feasibility checks
            fpc = get_fpc(basepath, 1990, future_year2, lon_lats_of_interest)
            # need to read in more as we want to get present day values as well.
            cpool = get_cpool(basepath, 1990, future_year2, lon_lats_of_interest)
            fluxes = get_fluxes_with_new_harvests(basepath, 1990, future_year2, lon_lats_of_interest)
            et = get_forest_et(basepath, future_year1, future_year2, lon_lats_of_interest)
            forest_swp = get_forest_swp(basepath, future_year1, future_year2, lon_lats_of_interest, only_forest=only_forest_swp)
            albedo = get_albedo(basepath, future_year1, future_year2, lon_lats_of_interest)
            stem_harvests_m3_per_ha = get_harvests_via_species_file(basepath, 1800, 2200, lons_lats_of_interest=lon_lats_of_interest, residuals=False)
            roughness = get_surface_roughness(basepath, future_year1, future_year2, lon_lats_of_interest)
            mitigationvals = get_new_total_mitigation(fluxes, cpool, discounting=(rcp if discounting else None))
            if compute_biodiv:
                biodiv_size = get_biodiversity_tree_sizes(basepath, future_year1, future_year2, lon_lats_of_interest)
                biodiv_cwd = get_biodiversity_cwd(basepath, future_year1, future_year2, lon_lats_of_interest)
            print('Done reading data for', simulation, 'and', rcp)

            # get present day values for additional constraints
            present_day_cpool = cpool.groupby(['Lon', 'Lat']).apply(lambda grp: grp[(grp['Year'] >= 1990) & (grp['Year'] <= 2010)]['Total'].mean())
            present_day_vegc = cpool.groupby(['Lon', 'Lat']).apply(lambda grp: grp[(grp['Year'] >= 1990) & (grp['Year'] <= 2010)]['VegC'].mean())
            present_day_hlp = fluxes.groupby(['Lon', 'Lat']).apply(lambda grp: grp[(grp['Year'] >= 1990) & (grp['Year'] <= 2010)]['slow_harv'].mean())
            present_day_harvest = stem_harvests_m3_per_ha.groupby(['Lon', 'Lat']).apply(lambda grp: grp[(grp['Year'] >= 1990) & (grp['Year'] <= 2010)]['total_harv_m3_wood_per_ha'].mean())
            present_day_forest_fpc= fpc.groupby(['Lon', 'Lat']).apply(lambda grp: grp[(grp['Year'] >= 1990) & (grp['Year'] <= 2010)]['forest_fpc'].mean())

            print('Aggregate for', simulation, 'and', rcp)
            forest_fpc = fpc.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'forest_fpc'))
            et = et.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'YearlyTotal'))
            forest_swp = forest_swp.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'min'))
            vegc = cpool.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'VegC'))
            csequestration = cpool.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'Total'))
            hlp = fluxes.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'slow_harv'))
            albedo_jul = albedo.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'albedo_jul'))
            albedo_jan = albedo.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'albedo_jan'))
            harvest = stem_harvests_m3_per_ha.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'total_harv_m3_wood_per_ha'))
            surface_roughness = roughness.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'z0'))
            mitigation = mitigationvals.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'total_mitigation'))
            if compute_biodiv:
                biodiversity_cwd = biodiv_cwd.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'ForestCWD'))
                biodiversity_big_trees = biodiv_size.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'thick_trees'))
                biodiversity_size_diversity = biodiv_size.groupby(['Lon', 'Lat']).apply(lambda grp: aggregation(grp, 'size_diversity'))
            print('Done with aggregation for', simulation, 'and', rcp)


            fpc_for_fpc_constraint = fpc.set_index(['Lon', 'Lat'])

            for gc in lon_lats_of_interest:

                if (gc[0], gc[1]) not in gc_data.keys():
                    gc_data[(gc[0], gc[1])] = Gc(gc[0], gc[1], simulation_names, boundary_simulations, used_variables, rcps)

                mygc = gc_data[(gc[0], gc[1])]

                mygc.forest_fpc[simulation].append(forest_fpc.loc[(gc[0], gc[1])])
                mygc.et[simulation].append(et.loc[(gc[0], gc[1])])
                mygc.swp[simulation].append(forest_swp.loc[(gc[0], gc[1])])
                mygc.csequestration[simulation].append(csequestration.loc[(gc[0], gc[1])])
                mygc.vegc[simulation].append(vegc.loc[(gc[0], gc[1])])
                mygc.hlp[simulation].append(hlp.loc[(gc[0], gc[1])])
                mygc.albedo_jul[simulation].append(albedo_jul.loc[(gc[0], gc[1])])
                mygc.albedo_jan[simulation].append(albedo_jan.loc[(gc[0], gc[1])])
                mygc.harvest[simulation].append(harvest.loc[(gc[0], gc[1])])
                mygc.surface_roughness[simulation].append(surface_roughness.loc[(gc[0], gc[1])])
                mygc.mitigation[simulation].append(mitigation.loc[(gc[0], gc[1])])

                if compute_biodiv:
                    mygc.biodiversity_cwd[simulation].append(biodiversity_cwd.loc[(gc[0], gc[1])])
                    mygc.biodiversity_big_trees[simulation].append(biodiversity_big_trees.loc[(gc[0], gc[1])])
                    mygc.biodiversity_size_diversity[simulation].append(biodiversity_size_diversity.loc[(gc[0], gc[1])])



                # Here we check if _converted_ stands yield an FPC of >=10%.
                # If not, we do not allow a conversion to this type of forest.
                if simulation.startswith('to'):
                    future_fpc = fpc_for_fpc_constraint.loc[(gc[0], gc[1])]
                    future_fpc = future_fpc[(future_fpc['Year'] >= future_year1) & (future_fpc['Year'] <= future_year2)].mean()

                    tree_type = simulation[2:].lower() if simulation != 'toCoppice' else 'bd'
                    if future_fpc[tree_type] < MIN_FPC_DEF_FOREST:
                        print(rcp, ': Converting to', simulation, 'not sensible, because converted stands will have less than 10% FPC. Removing this management option from the optimization.')
                        mygc.infeasible_forest_types.add(man_idx)

                mygc.add_constraint_base_values = dict(
                    csequestration=present_day_cpool.loc[(gc[0], gc[1])],
                    hlp=present_day_hlp.loc[(gc[0], gc[1])],
                    vegc=present_day_vegc.loc[(gc[0], gc[1])],
                    harvest=present_day_harvest.loc[(gc[0], gc[1])],
                    forest_fpc=present_day_forest_fpc.loc[(gc[0], gc[1])],
                )
    print('Data retrieval done')

    for gc in lon_lats_of_interest:
        mygc = gc_data[(gc[0], gc[1])]

        mygc.all_scores_dict_raw['et'] = mygc.et
        mygc.all_scores_dict_raw['harvest'] = mygc.harvest
        mygc.all_scores_dict_raw['hlp'] = mygc.hlp
        mygc.all_scores_dict_raw['csequestration'] = mygc.csequestration
        mygc.all_scores_dict_raw['vegc'] = mygc.vegc
        mygc.all_scores_dict_raw['albedo_jan'] = mygc.albedo_jan
        mygc.all_scores_dict_raw['albedo_jul'] = mygc.albedo_jul
        mygc.all_scores_dict_raw['biodiversity_cwd'] = mygc.biodiversity_cwd
        mygc.all_scores_dict_raw['biodiversity_big_trees'] = mygc.biodiversity_big_trees
        mygc.all_scores_dict_raw['biodiversity_size_diversity'] = mygc.biodiversity_size_diversity
        mygc.all_scores_dict_raw['surface_roughness'] = mygc.surface_roughness
        mygc.all_scores_dict_raw['swp'] = mygc.swp
        mygc.all_scores_dict_raw['forest_fpc'] = mygc.forest_fpc
        mygc.all_scores_dict_raw['mitigation'] = mygc.mitigation

        for sim_idx, sim in enumerate(simulation_names):
            is_feasible = sim_idx not in mygc.infeasible_forest_types
            for idxr, rcp2 in enumerate(rcps):
                if mygc.all_scores_dict_raw['forest_fpc'][sim][idxr] < MIN_FPC_DEF_FOREST:
                    print(sim + ' infeasible for ' + rcp2 + ' since fpc=' + "{:.3f}".format(mygc.all_scores_dict_raw['forest_fpc'][sim][idxr]) + ' compared to ' + "{:.3f}".format(MIN_FPC_DEF_FOREST))
                    mygc.infeasible_forest_types.add(sim_idx)
                    is_feasible = False
            if is_feasible:
                mygc.feasible_managements.append(sim)
                mygc.feasible_managements_ids.append(sim_idx)
            else:
                mygc.infeasible_managements_ids.append(sim_idx)

        print('infeasible managements', mygc.infeasible_forest_types)
        print('infeasible managements ids', mygc.infeasible_managements_ids)
        print('feasible managements', mygc.feasible_managements)
        print('feasible managements ids', mygc.feasible_managements_ids)

        if len(mygc.feasible_managements) == 0:
            mygc.no_management_feasible = True
            continue

        for variable in used_variables + [x for x in variables_for_additional_constraints if x not in used_variables]:
            lower_is_better = True if variable in ['water_avail'] else False
            if variable == 'biodiversity_combined':
                mygc.biodiversity_cwd_norm = normalize_for_uncertainty_scenarios(rcps, mygc.all_scores_dict_raw['biodiversity_cwd'], feasible_managements=mygc.feasible_managements, lower_is_better=lower_is_better)
                mygc.biodiversity_big_trees_norm = normalize_for_uncertainty_scenarios(rcps, mygc.all_scores_dict_raw['biodiversity_big_trees'], feasible_managements=mygc.feasible_managements, lower_is_better=lower_is_better)
                mygc.biodiversity_size_diversity_norm = normalize_for_uncertainty_scenarios(rcps, mygc.all_scores_dict_raw['biodiversity_size_diversity'], feasible_managements=mygc.feasible_managements, lower_is_better=lower_is_better)

                mygc.biodiversity_combined_norm = {}
                for rcp in rcps:
                    mean_bio_vals = np.mean(pd.DataFrame([mygc.biodiversity_big_trees_norm,mygc.biodiversity_size_diversity_norm,mygc.biodiversity_cwd_norm])[rcp].values)
                    max_mean_val = np.max(mean_bio_vals)
                    if max_mean_val > 0:
                        mygc.biodiversity_combined_norm[rcp] = mean_bio_vals / max_mean_val # to make sure that biodiversity indicator is also in [0, 1]
                    else:
                        print('biodiversity is always 0 for', (gc[0], gc[1]))
                        mygc.biodiversity_combined_norm[rcp] = mean_bio_vals

                mygc.all_scores_dict_u[variable] = mygc.biodiversity_combined_norm
            else:
                mygc.all_scores_dict_u[variable] = normalize_for_uncertainty_scenarios(rcps, mygc.all_scores_dict_raw[variable], feasible_managements=mygc.feasible_managements, lower_is_better=lower_is_better)

        print('additional_constraints', variables_for_additional_constraints, 'rcps', len(rcps))

        mygc.additional_constraints = dict(
            lhs=np.zeros((len(variables_for_additional_constraints) * len(rcps), len(simulation_names))),
            rhs=np.zeros(0)
        )
        for es in variables_for_additional_constraints:
            # wo do not need to normalize the additional constraints.
            # they will look like: 12.4w_base + 13.5w_toNe + 18.2w_toBd >= 15.2
            # 15.2 is the value of 2010 for example, and this enforces (in this example) that we have to take some toBd, otherwise we will never reach this value.
            # it does not go into the maximization objective function, hence it needs not be normalized
            add_const = np.tile(mygc.add_constraint_base_values[es], len(rcps))
            mygc.additional_constraints['rhs'] = np.concatenate((mygc.additional_constraints['rhs'], add_const))

        es_val_idx = 0
        idx2 = 0

        for es in used_variables:
            vall_u = mygc.all_scores_dict_u[es]  # vall is a dict of rcps now

            for rcp3 in rcps:
                mygc.es_vals_u[es_val_idx, :] = vall_u[rcp3][:len(simulation_names)]
                es_val_idx += 1

            idx2 += 1

        idx_add = 0
        for es in variables_for_additional_constraints:
            for idx3, rcp4 in enumerate(rcps):
                raw_vals_of_rcp = []
                for idx, simulation in enumerate(simulation_names):
                    raw_vals_of_rcp.append(mygc.all_scores_dict_raw[es][simulation][idx3])

                mygc.additional_constraints['lhs'][idx_add, :] = raw_vals_of_rcp
                idx_add += 1

        delete_rows = []
        for idxc, constraint in enumerate(mygc.additional_constraints['lhs']):
            if sum(constraint) == 0:
                delete_rows.append(idxc)
        if delete_rows:
            print('warning, additional contraint empty, deleting it.')
            mygc.additional_constraints['lhs'] = np.delete(mygc.additional_constraints['lhs'], delete_rows, axis=0)
            mygc.additional_constraints['rhs'] = np.delete(mygc.additional_constraints['rhs'], delete_rows)

    if optimizationType == OptimizationType.MINMAX:
        return gc_data
    else:
        raise ValueError('Specified an optimization method that was not implemented')


def get_fpc(basepath, min_year, max_year, lons_lats_of_interest, only_managed=True, fpc_type='crownarea'):
    # need to take only managed FPC here since only this tells us whether the management leads to sustained forest at the end of the century
    fpc_forest_real = (get_real_fpc_forest(basepath, years_of_interest=[min_year, max_year], lons_lats_of_interest=lons_lats_of_interest, only_managed=only_managed, fpc_type=fpc_type))
    fpc_forest_real = normalize_fpc(fpc_forest_real)
    fpc_forest_real['forest_fpc'] = fpc_forest_real['deciduous'] + fpc_forest_real['evergreen']
    return fpc_forest_real.reset_index()


def get_real_fpc_forest(basepath, lons_lats_of_interest=None, years_of_interest=None, only_managed=False, fpc_type='fpc'):
    """
    FPC is tricky: LPJ puts out the FPC of all PFTs but only related to the are where they are 'allowed' to be active.
    Since management changes this 'allowed' area, we need to also output this area and we call it 'active_fraction_forest'
    So here we scale the fpc_forest output file such that we really have the FPCs related to the whole forest area.
    """

    if fpc_type not in ['lai', 'crownarea', 'fpc']:
        raise ValueError('Incorrect FPC type specified')

    chunk_filter = ChunkFilter(years_of_interest=years_of_interest, lons_lats_of_interest=lons_lats_of_interest)

    if only_managed:
        filename_active_fraction = 'active_fraction_forest.out'
        filename_fpc = fpc_type + '_forest.out'
    else:
        filename_active_fraction = 'active_fraction.out'
        filename_fpc = fpc_type + '.out'

    iter_csv = pd.read_csv(basepath + filename_active_fraction, delim_whitespace=True, iterator=True)
    active_fraction_forest = pd.concat([chunk_filter.filter_chunk(chunk) for chunk in iter_csv])
    iter_csv = pd.read_csv(basepath + filename_fpc, delim_whitespace=True, iterator=True)
    fpc_forest2 = pd.concat([chunk_filter.filter_chunk(chunk) for chunk in iter_csv])

    active_fraction_forest['Total'] = 0.0  # a0dd dummy dimension to make multiplication work
    active_fraction_forest = active_fraction_forest.set_index(['Lon', 'Lat', 'Year'])
    active_fraction_forest = active_fraction_forest.div(active_fraction_forest.C3_gr, axis=0)  # put active fraction in relation to total grass active fraction (=total forest area)
    fpc_forest2 = fpc_forest2.set_index(['Lon', 'Lat', 'Year'])

    fpc_forest2 = fpc_forest2.loc[~fpc_forest2.index.duplicated(keep='first')]
    active_fraction_forest = active_fraction_forest.loc[~active_fraction_forest.index.duplicated(keep='first')]

    if fpc_type == 'lai':
        fpc_forest2 = 1-np.exp(-0.5*fpc_forest2)

    mul = fpc_forest2.mul(active_fraction_forest)
    if not only_managed:
        mul.drop(['Barren_sum', 'Forest_sum', 'Natural_sum'], inplace=True, axis=1)
    return mul


def compute_mitigation(cflux, discounting_factors, base_year = 2010):

    cflux = cflux.set_index('Year')

    #note: cflux['slow_h'] is accounted for in C stocks!!

    #Harvest = H_R_atm + H_S_atm
    #Lu_ch = LU_R_atm + Lu_S_atm
    # harvest residues and parts of stem harvests are used as fuelwood
    harvested_fuel_wood = cflux['H_R_atm'] + cflux['LU_R_atm'] + (cflux['H_S_atm'] + cflux['LU_S_atm']) * 0.305
    harvested_short_pool = (cflux['H_S_atm'] + cflux['LU_S_atm']) * (1-0.305)
    harvests_for_medium_and_slow_pool = (cflux['H_slow'] + cflux['Slow_LU'])

    # the Knauf value does not contain end of life burning of material in the substitution factor!
    # 1.5: wood usage in Germany Knauf2015, cf. factors on average 2.1 for Sathre2010
    # now also account for landfilling, 1.1 from Sathre2010 for landfilled wood products, 23% of waste is landfilled in Europe
    material_substitution_factor = 0.77 * 1.5 + 0.23 * 1.1
    cflux['material_substitution'] = (harvests_for_medium_and_slow_pool) * material_substitution_factor

    # now, all material usage has been accounted for including end-of-life, except the 77% of products that are not landfilled, for those we assume energy recovery
    decayed_medium_and_long_pool = cflux['Slow_h']
    energy_substitution_factor = 0.67  # Knauf2015
    cflux['fuel_substitution']  = (harvested_fuel_wood + harvested_short_pool * 0.77 + decayed_medium_and_long_pool * 0.77) * energy_substitution_factor

    if discounting_factors is not None:
        cflux['fuel_substitution'] *= discounting_factors['factor']
        cflux['material_substitution'] *= discounting_factors['factor']

    cflux['acc_fuel_substitution'] = cflux['fuel_substitution'].cumsum()
    cflux['fuel_mitigation'] = cflux['acc_fuel_substitution'] - cflux.loc[base_year, 'acc_fuel_substitution']

    cflux['acc_material_substitution'] = cflux['material_substitution'].cumsum()
    cflux['material_mitigation'] = cflux['acc_material_substitution'] - cflux.loc[base_year, 'acc_material_substitution']

    cflux['cstorage_mitigation'] = cflux['c_storage'] - cflux.loc[base_year, 'c_storage']

    cflux['total_mitigation'] = cflux['cstorage_mitigation'] + cflux['material_mitigation'] + cflux['fuel_mitigation']

    # for some reason Lon and Lat are in the index _and_ in the columns after this, so we delete the columns here...
    del cflux['Lon']
    del cflux['Lat']

    return cflux


def get_new_total_mitigation(cflux, cpool, discounting=None, base_year = 2010):
    cflux = cflux.set_index(['Lon', 'Lat', 'Year'])
    cpool = cpool.set_index(['Lon', 'Lat', 'Year'])

    discounting_factors = None
    if discounting:
        discounting_factors = get_discounting_factors_rcp(discounting)

    cflux['c_storage'] = cpool['Total']
    return cflux.reset_index().groupby(['Lon', 'Lat'], as_index=True).apply(lambda group: compute_mitigation(group, discounting_factors, base_year)).reset_index()


def get_discounting_factors_rcp(rcp):
    co2_emissions = pd.read_csv('total_co2_emissions_oecd_rcp_db-1.csv').drop(columns=['Region', 'Variable', 'Unit']).rename(columns={'Scenario':'Year'}).set_index('Year').transpose()
    co2_emissions.index = co2_emissions.index.astype(int)
    for year in range(2000, 2201):
        if year not in co2_emissions.index:
            co2_emissions.loc[year] = np.nan
    co2_emissions = co2_emissions.sort_index()
    co2_emissions = co2_emissions.interpolate()
    discount_rates = co2_emissions / co2_emissions.loc[2010, 'rcp26']
    discount_rates['rcp85'].loc[2010:2200] = 1
    discount_rates['rcp26'][discount_rates['rcp26'] < 0] = 0
    discount_rates_df = pd.DataFrame(discount_rates[rcp]).rename(columns={rcp: 'factor'})
    discount_rates_df.index.name = 'Year'
    return discount_rates_df


def get_fluxes_with_new_harvests(basepath, min_year, max_year, lons_lats_of_interest=None):
    cflux = ph.read_for_years(basepath + 'cflux.out', min_year, max_year, lons_lats_of_interest)
    cflux['slow_harv'] = cflux[['H_slow', 'Slow_LU']].sum(axis=1)
    # Harvest (total emissions to atmosphere from harvests) plus H_slow (Harvests that went into one of the longer lived pools) is total harvested wood
    cflux['total_harv'] = cflux[['Harvest', 'H_slow', 'Slow_LU', 'LU_S_atm', 'LU_R_atm']].sum(axis=1)

    cflux['total_harv_t'] = cflux['total_harv'] * to_t_dry_biomass_per_ha
    cflux['slow_harv_t'] = cflux['slow_harv'] * to_t_dry_biomass_per_ha

    return cflux


def get_harvests_via_species_file(basepath, min_year, max_year, lons_lats_of_interest=None, residuals=False, slow_only=False):

    # no need to multiply with active fraction! It is already contained in the output!
    harvest_file_type = 'slow' if slow_only else 'stem'
    harvests = ph.read_for_years(basepath + 'cmass_harv_per_species_' + harvest_file_type + '.out', min_year, max_year, lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
    luc = ph.read_for_years(basepath + 'cmass_luc_per_species_' + harvest_file_type + '.out', min_year, max_year, lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
    harvests += luc

    if residuals:
        assert slow_only is False
        harvests_residuals = ph.read_for_years(basepath + 'cmass_harv_per_species_residue_to_atm.out', min_year, max_year, lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
        luc_residuals = ph.read_for_years(basepath + 'cmass_luc_per_species_residue_to_atm.out', min_year, max_year, lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
        harvests += harvests_residuals
        harvests += luc_residuals

    # wooddens -- mostly from Savill 2019 -- is measured in kg/m3 at 15% moisture content
    # 15% is considered "air dry"
    # lpj output is kgC
    # the 0.47 carbon content refers to biomass
    # I am not calculating out the 15% since the values here are for dry mass and 15% is considered air dry
    wooddens_t_per_m3 = pd.read_csv('wood_density.csv')
    wooddens2 = wooddens_t_per_m3.transpose()
    wooddens2.columns = wooddens2.iloc[0]
    wooddens2 = wooddens2.drop(wooddens2.index[0])
    wooddens3 = wooddens2.loc['Value']
    wooddens3['BES'] = 1
    wooddens3['C3_gr'] = 1

    woodens_kg_per_m3 = wooddens3 * 1000.0

    m3_per_kg_wood = 1 / woodens_kg_per_m3
    m3_per_kg_wood['BES'] = 0
    m3_per_kg_wood['MRS'] = 0
    m3_per_kg_wood['C3_gr'] = 0

    harvests_in_wood_kg = harvests / 0.47

    harvests_in_wood_kg = harvests_in_wood_kg.mul(m3_per_kg_wood)

    kg_sum = harvests_in_wood_kg.sum(axis=1)
    harvests_in_wood_kg['total_harv_m3_wood_per_m2'] = kg_sum
    harvests_in_wood_kg['total_harv_m3_wood_per_ha'] = kg_sum * 10000
    harvests_in_wood_kg['total_harv_kg_C_per_m2'] = harvests.sum(axis=1)

    return harvests_in_wood_kg.reset_index()


def get_cpool(basepath, year1, year2, lons_lats_of_interest):
    cpool = ph.read_for_years(basepath + 'cpool.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest)
    return cpool


def get_albedo(basepath, year1, year2, lons_lats_of_interest):
    fpc_forest_real = sr.get_real_fpc_forest(basepath, years_of_interest=[year1, year2], lons_lats_of_interest=lons_lats_of_interest, fpc_type='lai', only_managed=False)

    snow = ph.read_for_years(basepath + 'mysnow.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
    snow = snow.loc[~snow.index.duplicated(keep='first')]

    fpc_with_albedo = get_forest_area_albedo(fpc_forest_real, snow)
    fpc_with_albedo = fpc_with_albedo.reset_index()
    return fpc_with_albedo


def get_forest_et(basepath, year1, year2, lons_lats_of_interest, aet_only=False):

    aet = ph.read_for_years(basepath + 'maet_forest.out', year1, year2, lons_lats_of_interest=lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
    aet['YearlyTotal'] = aet.loc[:, months].sum(axis=1)

    if not aet_only:
        soil_et = ph.read_for_years(basepath + 'mevap_forest.out', year1, year2, lons_lats_of_interest=lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
        soil_et['YearlyTotal'] = soil_et.loc[:, months].sum(axis=1)

        interception = ph.read_for_years(basepath + 'mintercep_forest.out', year1, year2, lons_lats_of_interest=lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])
        interception['YearlyTotal'] = interception.loc[:, months].sum(axis=1)

        aet['YearlyTotal'] += soil_et['YearlyTotal'] + interception['YearlyTotal']

    return aet.reset_index()


def get_forest_swp(basepath, year1, year2, lons_lats_of_interest, only_forest=True):
    swp_upper = ph.read_for_years(basepath + 'mpsi_s_upper_forest.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest)
    swp_lower = ph.read_for_years(basepath + 'mpsi_s_lower_forest.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest)
    if not only_forest:
        swp_upper = ph.read_for_years(basepath + 'mpsi_s_upper.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest)
        swp_lower = ph.read_for_years(basepath + 'mpsi_s_lower.out', year1=year1, year2=year2, lons_lats_of_interest=lons_lats_of_interest)

    # upper layer is 50cm, lower layer is 100cm deep
    swp_upper.iloc[:, 3:15] = (0.5 * swp_upper.iloc[:, 3:15] + swp_lower.iloc[:, 3:15]) / 1.5

    swp_upper['min'] = swp_upper.loc[:, months].min(axis=1)
    swp_upper['mean'] = swp_upper.loc[:, months].mean(axis=1)
    return swp_upper


def get_biodiversity_tree_sizes(basepath, year1, year2, lons_lats_of_interest):
    diams = ph.read_for_years(basepath + 'diamstruct_forest.out', year1, year2, lons_lats_of_interest).set_index(['Lon', 'Lat', 'Year'])

    tmp = compute_entropy(diams)  # store in temporary so that it does not affect thick_trees
    # index 0: 1-10, index 5: 51-60
    diams['thick_trees'] = diams.iloc[:, 5:].sum(axis=1)
    diams['size_diversity'] = tmp
    return diams.reset_index()


def get_biodiversity_cwd(basepath, year1, year2, lons_lats_of_interest):
    return ph.read_for_years(basepath + 'diversity.out', year1, year2, lons_lats_of_interest)


def get_surface_roughness(basepath, year1, year2, lons_lats_of_interest):
    roughness = ph.read_for_years(basepath + 'canopy_height.out', year1, year2, lons_lats_of_interest)
    roughness['z0'] = 0.5 * roughness['z0'] + 0.5 * roughness['z0_Win']
    return roughness




def normalize_fpc(fpc_forest):
    # we have the mean fractional plant cover per PFT
    fpc_forest['ne'] = fpc_forest.loc[:, ne_pfts].sum(axis=1)
    fpc_forest['bd'] = fpc_forest.loc[:, bd_pfts].sum(axis=1)
    fpc_forest['be'] = fpc_forest.loc[:, be_pfts].sum(axis=1)
    fpc_forest['nd'] = fpc_forest.loc[:, nd_pfts].sum(axis=1)

    fpc_forest['evergreen'] = fpc_forest['ne'] + fpc_forest['be']
    fpc_forest['deciduous'] = fpc_forest['nd'] + fpc_forest['bd']
    # In Boisier2013, shrubs are considered grass
    fpc_forest['grass'] = fpc_forest.loc[:, shrub_pfts].sum(axis=1) + fpc_forest.loc[:, grass_pfts].sum(axis=1)
    fpc_forest['bare'] = 0
    fpc_forest['total_fpc_trees'] = fpc_forest['evergreen'] + fpc_forest['deciduous']
    fpc_forest['total_fpc'] = fpc_forest['evergreen'] + fpc_forest['deciduous'] + fpc_forest['grass']
    fpc_forest['bare'] = 1 - fpc_forest['total_fpc']
    # normalize when total fpc is larger than 1
    fpc_forest.loc[fpc_forest['total_fpc'] > 1, ['evergreen', 'deciduous', 'grass']] = fpc_forest.loc[:, ['evergreen', 'deciduous', 'grass']].div(fpc_forest['total_fpc'], axis=0)
    fpc_forest.loc[fpc_forest['total_fpc'] > 1, ['ne', 'bd', 'be', 'nd', 'grass']] = fpc_forest.loc[:, ['ne', 'bd', 'be', 'nd', 'grass']].div(fpc_forest['total_fpc'], axis=0)
    fpc_forest.loc[fpc_forest['total_fpc'] >= 1, ['total_fpc']] = 1
    fpc_forest.loc[fpc_forest['total_fpc'] >= 1, ['bare']] = 0

    fpc_forest['tree_fpc'] = fpc_forest['evergreen'] + fpc_forest['deciduous']

    return fpc_forest


def get_forest_area_albedo(fpc_forest, snow):
    """
    fpc_forest has to be adapted for land cover fraction beforehand!
    I.e., the fpc.out from LPJ has the FPC according to the "active area" of that FPC. We need to take that into account.
    But this is done before it is passed in here.
    """

    # summer, winter winter+snow, from Boisier
    # MODIS seasonal mean shortwave broadband (0.3–5 µm) bihemispherical reflectance (white-sky albedo)
    # They argue: white-sky albedo iss a good approx for the daily mean surface albedo.
    albedos = dict(
        crops=(0.178, 0.141, 0.546),
        grass=(0.176, 0.161, 0.568),
        evergreen=(0.104, 0.094, 0.205),
        deciduous=(0.153, 0.117, 0.244),
        bare=(0.246, 0.205, 0.535),  # first value: from Boisier, which is average over Northern Hemisphere extratropics so should be ok.
        unknown=0.15
    )




    pd.testing.assert_index_equal(fpc_forest.index, snow.index)

    fpc_forest = normalize_fpc(fpc_forest)

    # note: if there is snow in summer, we take the winter+snow albedo values.
    fpc_forest['albedo_jul'] = fpc_forest['evergreen'] * (snow['Jul'] * albedos['evergreen'][2] + (1 - snow['Jul']) * albedos['evergreen'][0]) \
                               + fpc_forest['deciduous'] * (snow['Jul'] * albedos['deciduous'][2] + (1 - snow['Jul']) * albedos['deciduous'][0]) \
                               + fpc_forest['grass'] * (snow['Jul'] * albedos['grass'][2] + (1 - snow['Jul']) * albedos['grass'][0]) \
                               + fpc_forest['bare'] * (snow['Jul'] * albedos['bare'][2] + (1 - snow['Jul']) * albedos['bare'][0])

    fpc_forest['albedo_jan'] = fpc_forest['evergreen'] * (snow['Jan'] * albedos['evergreen'][2] + (1 - snow['Jan']) * albedos['evergreen'][1]) \
                               + fpc_forest['deciduous'] * (snow['Jan'] * albedos['deciduous'][2] + (1 - snow['Jan']) * albedos['deciduous'][1]) \
                               + fpc_forest['grass'] * (snow['Jan'] * albedos['grass'][2] + (1 - snow['Jan']) * albedos['grass'][1]) \
                               + fpc_forest['bare'] * (snow['Jan'] * albedos['bare'][2] + (1 - snow['Jan']) * albedos['bare'][1])
    return fpc_forest
