import exceptions
import optimization as opt
import optimization_europe_aggregation
import optimization_preparation as oprep
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import OptimizeResult
from datetime import datetime
import pandas as pd

luyssaert_variables = ['base_broadleaved_frac', 'base_coniferous_frac', 'total_unmanaged', 'total_broadleaved', 'total_coniferous', 'total_grass', 'total_coppice', 'total_high', 'forest_frac']


FACTOR_TO_HAVE_LOW_VALUES = 100000000  # necessary, because otherwise the numbers get too high and there will be numerical issues in the optimization.





def optimize_multiple_gridcells(rcps, simulations_paths, used_simulations, es, constraints, lambda_opt, weights, gcs, min_year, max_year, file_basepath=None, plot=False, gc_location=None, discounting=True, plot_for_paper_name=None, minmax=True, gc_lambda=False, use_dataset=None, store=False, only_forest_swp=False, unmanaged_constraint=0.0, unmanaged_constraint_every_cell=0.0, harvest_constraint_m3=0.0, hlp_constraint_m3=0.0, cpool_constraint=0.0):

    gcs = get_optimization_inputs_for_all_cells_new(constraints, discounting, es, gcs, max_year, min_year, rcps, simulations_paths, used_simulations, only_forest_swp=only_forest_swp, use_dataset=use_dataset, store=store)

    assert gcs.set_index(['Lon', 'Lat']).index.is_unique, "Index not unique after computing the optimization inputs"

    print('Combining all inputs into one LP')

    gcs, total_boundss, total_lhs_eqs, total_lhs_ineqs, total_objs, total_rhs_eqs, total_rhs_ineqs = combine_all_inputs_to_one_optimization(gcs, lambda_opt, minmax, rcps, used_simulations, weights, unmanaged_constraint=unmanaged_constraint, unmanaged_constraint_every_cell=unmanaged_constraint_every_cell, harvest_constraint_m3=harvest_constraint_m3, hlp_constraint_m3=hlp_constraint_m3, cpool_constraint=cpool_constraint)

    assert gcs.set_index(['Lon', 'Lat']).index.is_unique, "Index not unique after combining inputs for optimization"

    print('size of LP, eqs:', total_lhs_eqs.shape, ', ineqs:', total_lhs_ineqs.shape)

    print('Starting optimization...')
    opt_result = linprog(c=total_objs, A_ub=total_lhs_ineqs, b_ub=total_rhs_ineqs, A_eq=total_lhs_eqs, b_eq=total_rhs_eqs, bounds=total_boundss, method="highs-ipm", options={"maxiter": 100000, 'disp': False})

    print('Optimization is done.')
    outputs_all_gcs = []

    # put into output format per GC

    assert opt_result.success , "optimization was not successful!" + str(opt_result)

    start_of_gc_in_result_vector = 1 if minmax else 0 # because of the additional z variable for the overall minmax optimization

    print('Aggregating results')

    for i, row in gcs.iterrows():

        feas = row.loc['feasible_managements_list']

        # the dummy variable is included here!
        # TODO do a bunch of tests and assertions!
        gc_result = opt_result.x[start_of_gc_in_result_vector:(start_of_gc_in_result_vector + len(used_simulations) + 1)]
        portfolio_fractions = gc_result[1:]

        assert np.sum(portfolio_fractions) > 0.99 and np.sum(portfolio_fractions) < 1.01, "the portfolios do not sum up to 1, but to " + str(np.sum(portfolio_fractions))

        start_of_gc_in_result_vector += (len(used_simulations)+1)

        row.loc[used_simulations] = portfolio_fractions

        # row.loc['feasible'] = all_rcp_solution_new.success

        # if infeasible_managements:
        #     row.loc['all_managements_possible'] = False

        for idx, management in enumerate(feas):
            row.loc['feasible_' + management] = True

        # except oplot.NoManagementFeasibleError:
        # row.loc['feasible'] = False
        # row.loc['no_management_feasible'] = True
        # print('No management is feasible for this gridcell')

        try:
            luyssaert_vals = optimization_europe_aggregation.luyssaert_values(used_simulations, portfolio_fractions, (row['Lon'], row['Lat']))

            for luyssaert_variable in luyssaert_variables:
                row.loc[luyssaert_variable] = luyssaert_vals[luyssaert_variable]
        except exceptions.NoManagedForestError:
            row.loc['has_forest'] = False
            print('This gridcell does not have any managed forest.')

            # we can get values like -1E-16 sometimes in the optimization.optimize
        for man in used_simulations:
            if row.loc[man] < 0:
                row.loc[man] = 0

        output = dict(
            row=row,
            es=es,
            feasible_managements=feas,
            all_managements=used_simulations,
            es_vals_all_rcps_new=row['es_vals_all_rcps_new'],
            all_rcp_solution_new=OptimizeResult(x=gc_result, success=1),
            scores=row['all_scores_all_rcps_new'],
            rcps=rcps,
            all_scores_raw=row['all_scores_raw']
        )

        outputs_all_gcs.append(output)

    return outputs_all_gcs


def combine_all_inputs_to_one_optimization(gcs, lambda_opt, minmax, rcps, used_simulations, weights, unmanaged_constraint=0.0, unmanaged_constraint_every_cell=0.0, harvest_constraint_m3=0.0, hlp_constraint_m3=0.0, cpool_constraint=0.0):
    objs = []
    lhs_ineqs = []
    rhs_ineqs = []
    lhs_eqs = []
    rhs_eqs = []
    boundss = []

    dropped_rows = []

    for i, row in gcs.iterrows():  # get a copy of each row

        if gcs.loc[i, 'no_management_feasible']:
            print('SKIP GC!' + str(row['Lat']) + ', ' + str(row['Lon']))
            dropped_rows.append(i)
            continue

        obj, lhs_ineq, rhs_ineq, lhs_eq, rhs_eq, bounds = opt.get_optimization_inputs_for_gridcell_general_min_max_distance(
            gcs.loc[i, 'es_vals_all_rcps_new'], rcps,
            lambda_opt=lambda_opt,
            es_weights=weights,
            additional_constraints=gcs.at[i, 'additional_constraints'],
            infeasible_management_idxs=[])

        objs.append(obj)
        lhs_ineqs.append(lhs_ineq)
        rhs_ineqs.append(rhs_ineq)
        lhs_eqs.append(lhs_eq)
        rhs_eqs.append(rhs_eq)
        boundss.append(bounds)
    gcs = gcs[~gcs['no_management_feasible']]
    if minmax:
        total_boundss, total_lhs_eqs, total_lhs_ineqs, total_objs, total_rhs_eqs, total_rhs_ineqs = merge_inputs_for_optimization_minmax2(
            boundss, gcs, lhs_eqs, lhs_ineqs, objs, rhs_eqs, rhs_ineqs, used_simulations, lambda_opt=lambda_opt, unmanaged_constraint=unmanaged_constraint, unmanaged_constraint_every_cell=unmanaged_constraint_every_cell, harvest_constraint_m3=harvest_constraint_m3, hlp_constraint_m3=hlp_constraint_m3, cpool_constraint=cpool_constraint)
    else:
        total_boundss, total_lhs_eqs, total_lhs_ineqs, total_objs, total_rhs_eqs, total_rhs_ineqs = merge_inputs_for_optimization_sum(
            boundss, gcs, lhs_eqs, lhs_ineqs, objs, rhs_eqs, rhs_ineqs, used_simulations, lambda_opt=lambda_opt, unmanaged_constraint=unmanaged_constraint, unmanaged_constraint_every_cell=unmanaged_constraint_every_cell, harvest_constraint_m3=harvest_constraint_m3, hlp_constraint_m3=hlp_constraint_m3)
    return gcs, total_boundss, total_lhs_eqs, total_lhs_ineqs, total_objs, total_rhs_eqs, total_rhs_ineqs


def get_optimization_inputs_for_all_cells_new(constraints, discounting, es, gcs, max_year, min_year, rcps, simulations_paths, used_simulations, use_dataset=None, store=False, only_forest_swp=False):

    lonslats = gcs.reset_index()[['Lon', 'Lat']].drop_duplicates().apply(tuple, axis=1)

    if not use_dataset:

        gcs['feasible_managements_list'] = None
        gcs['feasible_managements_list'] = gcs['feasible_managements_list'].astype(object)
        gcs['es_vals_all_rcps_new'] = None
        gcs['es_vals_all_rcps_new'] = gcs['es_vals_all_rcps_new'].astype(object)
        gcs['all_scores_all_rcps_new'] = None
        gcs['all_scores_all_rcps_new'] = gcs['all_scores_all_rcps_new'].astype(object)
        gcs['all_scores_raw'] = None
        gcs['all_scores_raw'] = gcs['all_scores_raw'].astype(object)
        gcs['additional_constraints'] = None
        gcs['additional_constraints'] = gcs['additional_constraints'].astype(object)
        gcs['add_constraint_base_values'] = None
        gcs['add_constraint_base_values'] = gcs['add_constraint_base_values'].astype(object)
        gcs['no_management_feasible'] = False

        gc_data = oprep.get_es_vals_new_all_gcs(
            rcps,
            simulations_paths,
            future_year1=min_year,
            future_year2=max_year,
            simulation_names=used_simulations,
            optimizationType=oprep.OptimizationType.MINMAX,
            used_variables=es,
            boundary_simulations=[], lon_lats_of_interest=lonslats,
            variables_for_additional_constraints=constraints,
            discounting=discounting,
            only_forest_swp=only_forest_swp)

        gcs = gcs.set_index(['Lon', 'Lat'])

        for gc, mygc in gc_data.items():
            gcs.at[(gc[0], gc[1]), 'feasible_managements_list'] = mygc.feasible_managements
            gcs.at[(gc[0], gc[1]), 'es_vals_all_rcps_new'] = mygc.es_vals_u
            gcs.at[(gc[0], gc[1]), 'all_scores_all_rcps_new'] = mygc.all_scores_dict_u
            gcs.at[(gc[0], gc[1]), 'all_scores_raw'] = mygc.all_scores_dict_raw
            gcs.at[(gc[0], gc[1]), 'additional_constraints'] = mygc.additional_constraints
            gcs.at[(gc[0], gc[1]), 'add_constraint_base_values'] = mygc.add_constraint_base_values
            gcs.at[(gc[0], gc[1]), 'no_management_feasible'] = mygc.no_management_feasible

        gcs = gcs.reset_index()

        if store:
            gcs.to_pickle('data_for_optimization_after_revision_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.dat')

        return gcs
    else:
        return pd.read_pickle(use_dataset).set_index(['Lon', 'Lat']).loc[lonslats].reset_index()


def merge_inputs_for_optimization_sum(boundss, gcs, lhs_eqs, lhs_ineqs, objs, rhs_eqs, rhs_ineqs, used_simulations, lambda_opt, unmanaged_constraint=0.0, unmanaged_constraint_every_cell=0.0, harvest_constraint_m3=0.0, hlp_constraint_m3=0.0):
    # objective simply get concatenated to get sum of two grid cells
    total_objs = np.concatenate(objs)
    total_rhs_ineqs = np.concatenate(rhs_ineqs)
    total_rhs_eqs = np.concatenate(rhs_eqs)
    total_boundss = np.concatenate(boundss)
    # lhs need to be concatenated
    # lhs ineqs need to be concatenated like this:
    # LHS1 0 0 0
    # 0 0 0 LHS2
    nrow, ncol = lhs_ineqs[0].shape  # 28 (esis*rcps) x 7 (6 managements plus 1 for the z-value (proxy to transform maximin problem to LP))

    check_lambda_opt_in_objectives(lambda_opt, ncol, total_objs)

    total_lhs_ineqs = np.zeros((nrow * len(gcs), ncol * len(gcs)))
    for idx, lhs in enumerate(lhs_ineqs):
        total_lhs_ineqs[(idx * nrow):((idx + 1) * nrow), (idx * ncol):((idx + 1) * ncol)] = lhs
    # lhs eqs need to be concatenated like this:
    # LHS1 0
    # 0 LHS2
    total_lhs_eqs = np.zeros((len(gcs), ncol * len(gcs)))
    for idx, lhs in enumerate(lhs_eqs):
        total_lhs_eqs[idx, (idx * ncol):((idx + 1) * ncol)] = np.squeeze(lhs)

    #TODO extract these constraint additions into own functions.
    if unmanaged_constraint > 0.0:
        # add up the omegas for unmanaged, weighted by the forest areas of the grid cells
        total_forest_area = 0.0
        vec = []
        for i, row in gcs.iterrows():
            forest_area = row['forest_frac']*row['area']
            total_forest_area += forest_area
            vec = np.concatenate((vec, [0, 0, 0, 0, 0, 0, -forest_area]), axis=0)
        total_lhs_ineqs = np.concatenate((total_lhs_ineqs, [vec]), axis=0)
        total_rhs_ineqs = np.concatenate([total_rhs_ineqs, [-total_forest_area * unmanaged_constraint]])

    if unmanaged_constraint_every_cell > 0.0:
        unmanaged_i=0
        for row in gcs.iterrows():
            vec = np.zeros(ncol * len(gcs))
            vec[unmanaged_i*ncol + len(used_simulations)] = -1
            total_lhs_ineqs = np.concatenate((total_lhs_ineqs, [vec]))
            total_rhs_ineqs = np.concatenate((total_rhs_ineqs, [-unmanaged_constraint_every_cell]))
            unmanaged_i+=1


    if harvest_constraint_m3 > 0.0:
        harvest_vec = []
        for i, row in gcs.iterrows():
            # forest_area_m2 = row['forest_frac']*row['area']
            harvests_per_scenario = row['all_scores_raw']['harvest']  # this is total_harv_m3_wood_per_ha
            row_harvest_vec = [0] # for gc z
            for sim in used_simulations:
                # for all 4 RCPS
                min_harvest_of_scenario = np.min(harvests_per_scenario[sim]) # m3 wood / ha
                row_harvest_vec.append(- min_harvest_of_scenario * (row['area']/(10000*FACTOR_TO_HAVE_LOW_VALUES))) # m3

            harvest_vec = np.concatenate((harvest_vec, row_harvest_vec), axis=0)

        total_lhs_ineqs = np.concatenate((total_lhs_ineqs, [harvest_vec]), axis=0)
        total_rhs_ineqs = np.concatenate([total_rhs_ineqs, [-harvest_constraint_m3]])


    if hlp_constraint_m3 > 0.0:
        hlp_vec = []
        for i, row in gcs.iterrows():
            # forest_area_m2 = row['forest_frac']*row['area']
            hlps_per_scenario = row['all_scores_raw']['hlp']  # this is total_harv_m3_wood_per_ha
            row_hlp_vec = [0] # for gc z
            for sim in used_simulations:
                # for all 4 RCPS
                min_hlp_of_scenario = np.min(hlps_per_scenario[sim]) # m3 wood / ha
                row_hlp_vec.append(- min_hlp_of_scenario * (row['area']/(FACTOR_TO_HAVE_LOW_VALUES))) # m3

            hlp_vec = np.concatenate((hlp_vec, row_hlp_vec), axis=0)

        total_lhs_ineqs = np.concatenate((total_lhs_ineqs, [hlp_vec]), axis=0)
        total_rhs_ineqs = np.concatenate([total_rhs_ineqs, [-hlp_constraint_m3]])


    # example unmananaged >= 50% and unmanaged GC2 >= 0.5
    # total_lhs_ineqs = np.concatenate((total_lhs_ineqs, [[0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1]]), axis=0)
    # total_rhs_ineqs = np.concatenate([total_rhs_ineqs, [-1, -0.5]])

    return total_boundss, total_lhs_eqs, total_lhs_ineqs, total_objs, total_rhs_eqs, total_rhs_ineqs


# obj = [1, 0, 0, 0, 0, 0, 0]
# lhs_ineq = [[-1, 1, 0, 0, 0, 0, 0], # z1 <= zgc
#             [-1, 0, 0, 0, 1, 0, 0], # z2 <= zgc because zgc is the max, is that right?
#             [0, -1, -0.5, -0.9, 0, 0, 0], # gc1
#             [0, -1, -0.8, -0.6, 0, 0, 0],
#             [0, 0, 0, 0, -1, -0.7, -0.9], # gc2
#             [0, 0, 0, 0, -1, -0.8, -0.6],
#             ]
# rhs_ineq = [0, 0, 0, 0, 0, 0]
# lhs_eq = [[0, 0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1]]
# rhs_eq = [1, 1]
# NOTE ABOUT lambda_gc
# if we only optimize the worst case over the grid cells, we don't get a good solution.
# Example: Let's say we have two grid cells and each grid cell has two possible portfolios (of course this is a super simplification)
# ESI performances for GC1: Portfolio 1: 0.4, Portfolio 2: 0.3
# ESI performances for GC2: Portfolio 1: 0.4, Portfolio 2: 0.8
# If we maximize the worst case over the grid cells, the optimization will most likely select Portfolio 1 both for GC1 and GC2.
# The worst case of the two is thus 0.4
# But if we chose GC1-PF1 and GC2-PF2, the worst case would still be 0.4, but the solution would be much better objectively (but not to the optimizer as it only cares for the worst case)
# This is why to the (max min performance) we can add a tiny bit of average performance for each GC.
# That way, the results differ, e.g.: 0.99 * 0.4 + 0.005 * 0.4 + 0.005 * 0.4 = 0.4
# But: 0.99 * 0.4 + 0.005 * 0.4 + 0.005 * 0.8 = 0.402.
# So the optimizer will choose the solution GC1-PF1, GC2-PF2.
# The factors are ever so tiny, they don't affect the selection of the worst case, but will make sure if there are multiple solutions for a GC, the best one is chosen.
# In out optimization, we add the sum of performances to the objective as well (which is 0.8*min +0.2*mean), but scale it with 0.001.
def merge_inputs_for_optimization_minmax2(boundss, gcs, lhs_eqs, lhs_ineqs, objs, rhs_eqs, rhs_ineqs, used_simulations, lambda_opt=0.2, unmanaged_constraint=0.0, unmanaged_constraint_every_cell=0.0, harvest_constraint_m3=0.0, hlp_constraint_m3=0.0, cpool_constraint=0.0):
    nrow, ncol = lhs_ineqs[0].shape  # 28 (esis*rcps) x 7 (6 managements plus 1 proxy variable to convert maximin grid cell problem to LP)
    total_objs = np.concatenate(objs)

    LAMBDA_GC = 0.001

    check_lambda_opt_in_objectives(lambda_opt, ncol, total_objs)

    n_variables = len(total_objs)

    total_objs_new = np.zeros(n_variables+1)
    total_objs_new[0] = 1 # add entry for the _one_ additional proxy variable to make the maximin problem over all grid cells into a LP
    total_objs_new[1:] = total_objs * LAMBDA_GC # the objs already contain the formulas for the performances (0.8wc + 0.2mean), so we can simply add them to the objective
    total_objs = total_objs_new

    total_rhs_ineqs = np.concatenate(rhs_ineqs)
    total_rhs_eqs = np.concatenate(rhs_eqs)
    total_boundss = np.concatenate(boundss)
    # lhs need to be concatenated
    # lhs ineqs need to be concatenated like this:
    # -1 1 0 0 0 for the additional z variable (linearization of max min perf --> perf1 >= z, perf2 >= z, ...)
    # -1 0 1 0 0
    # 0 LHS1 0 0
    # 0 0 0 LHS2
    # so two extra rows (one per gc) and 1 extra col
    total_lhs_ineqs = np.zeros(((nrow+1) * len(gcs), ncol * len(gcs) + 1))

    # just like for the sum
    total_lhs_ineqs_subset = np.zeros((nrow * len(gcs), ncol * len(gcs)))
    for idx, lhs in enumerate(lhs_ineqs):
        total_lhs_ineqs_subset[(idx * nrow):((idx + 1) * nrow), (idx * ncol):((idx + 1) * ncol)] = lhs


    curr_row = 0
    for curr_row in range(len(gcs)):
        new_row = np.zeros(total_lhs_ineqs.shape[1])
        new_row[0] = -1  # overall z
        new_row[(1 + curr_row*ncol):(1 + curr_row*ncol + ncol)] = objs[curr_row] # objs already contains the formula for the performance of grid cells (0.8min + 0.2mean)

        total_lhs_ineqs[curr_row] = new_row

    total_lhs_ineqs[(curr_row+1):, 1:] = total_lhs_ineqs_subset
    total_rhs_ineqs = np.concatenate((np.zeros(len(gcs)), total_rhs_ineqs))


    # lhs eqs need to be concatenated like this:
    # 0 LHS1 0
    # 0 0 LHS2
    total_lhs_eqs = np.zeros((len(gcs), ncol * len(gcs) + 1))
    for idx, lhs in enumerate(lhs_eqs):
        total_lhs_eqs[idx, (idx * ncol + 1):((idx + 1) * ncol + 1)] = np.squeeze(lhs)

    if unmanaged_constraint > 0.0:
        # add up the omegas for unmanaged
        total_forest_area = 0.0
        vec = [0] # for the global z
        for i, row in gcs.iterrows():
            forest_area = row['forest_frac']*row['area']
            total_forest_area += forest_area
            vec = np.concatenate((vec, [0, 0, 0, 0, 0, 0, -forest_area]), axis=0)
        total_lhs_ineqs = np.concatenate((total_lhs_ineqs, [vec]), axis=0)
        total_rhs_ineqs = np.concatenate([total_rhs_ineqs, [-total_forest_area * unmanaged_constraint]])

    if harvest_constraint_m3 > 0.0:
        harvest_vec = [0] # for the global z
        for i, row in gcs.iterrows():
            harvests_per_scenario = row['all_scores_raw']['harvest']  # this is total_harv_m3_wood_per_ha
            row_harvest_vec = [0] # for gc z
            for sim in used_simulations:
                # for all 4 RCPS
                min_harvest_of_scenario = np.min(harvests_per_scenario[sim])  # m3 wood / ha
                row_harvest_vec.append(- min_harvest_of_scenario * (row['area']/(10000*FACTOR_TO_HAVE_LOW_VALUES))) # m3

            harvest_vec = np.concatenate((harvest_vec, row_harvest_vec), axis=0)

        total_lhs_ineqs = np.concatenate((total_lhs_ineqs, [harvest_vec]), axis=0)
        total_rhs_ineqs = np.concatenate([total_rhs_ineqs, [-harvest_constraint_m3]])

    if hlp_constraint_m3 > 0.0:
        hlp_vec = [0]
        for i, row in gcs.iterrows():
            hlp_per_scenario = row['all_scores_raw']['hlp']  # this is total_harv_m3_wood_per_ha
            row_hlp_vec = [0] # for gc z
            for sim in used_simulations:
                # for all 4 RCPS
                min_hlp_of_scenario = np.min(hlp_per_scenario[sim]) # m3 wood / ha
                row_hlp_vec.append(- min_hlp_of_scenario * (row['area']/FACTOR_TO_HAVE_LOW_VALUES)) # m3

            hlp_vec = np.concatenate((hlp_vec, row_hlp_vec), axis=0)

        total_lhs_ineqs = np.concatenate((total_lhs_ineqs, [hlp_vec]), axis=0)
        total_rhs_ineqs = np.concatenate([total_rhs_ineqs, [-hlp_constraint_m3]])

    if unmanaged_constraint_every_cell > 0.0:
        unmanaged_i=0
        for row in gcs.iterrows():
            vec = np.zeros(ncol * len(gcs) + 1) #
            vec[unmanaged_i*ncol + len(used_simulations) + 1] = -1
            total_lhs_ineqs = np.concatenate((total_lhs_ineqs, [vec]))
            total_rhs_ineqs = np.concatenate((total_rhs_ineqs, [-unmanaged_constraint_every_cell]))
            unmanaged_i+=1

    total_boundss = np.insert(total_boundss, 0, (0, float("inf")), axis=0)

    return total_boundss, total_lhs_eqs, total_lhs_ineqs, total_objs, total_rhs_eqs, total_rhs_ineqs


def check_lambda_opt_in_objectives(lambda_opt, ncol, total_objs):
    # check: objective of each cell should be 0.8 * min_omega (es) + 0.2 * (omega_1 * mean_es(man1) + ... + omega_6 * mean_es(man6))
    for i in range(len(total_objs)):
        if i % ncol == 0:
            assert (total_objs[i] == (
                        1 - lambda_opt)), "aggregation of objectives not correct for minmax optimization. Expected {:.2f} but was {:.6f}".format(
                1 - lambda_opt, total_objs[i])


