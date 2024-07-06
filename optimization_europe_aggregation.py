import pandas as pd

import analyze_lpj_output as analysis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import pandas_helper
from constants import *
from constants import forest_fraction_file
from exceptions import NoManagedForestError
from optimization_preparation import get_forest_swp, get_cpool, \
    get_biodiversity_tree_sizes, get_biodiversity_cwd, get_surface_roughness, get_fluxes_with_new_harvests, get_new_total_mitigation, \
    get_forest_et, get_harvests_via_species_file, get_albedo
from pandas_helper import *

from variable_units import *

accuracy_for_table = dict(
    cpool=lambda x: "{:.1f}".format(x),
    mitigation=lambda x: "{:.1f}".format(x),
    slow_pool=lambda x: "{:.2f}".format(x),
    harvest=lambda x: "{:.2f}".format(x),
    hlp=lambda x: "{:.2f}".format(x),
    surface_roughness=lambda x: "{:.2f}".format(x),
    et=lambda x: "{:.0f}".format(x),
    swp=lambda x: "{:.2f}".format(x),
    biodiversity_cwd=lambda x: "{:.2f}".format(x),
    biodiversity_big_trees=lambda x: "{:.0f}".format(x),
    biodiversity_size_diversity=lambda x: "{:.2f}".format(x),
    biodiversity_combined=lambda x: "{:.2f}".format(x),
    albedo_jul=lambda x: "{:.2f}".format(x),
    albedo_jan=lambda x: "{:.2f}".format(x),
)
accuracy_for_table_std = dict(
    cpool=lambda x: "{:.1f}".format(x),
    mitigation=lambda x: "{:.1f}".format(x),
    slow_pool=lambda x: "{:.2f}".format(x),
    harvest=lambda x: "{:.2f}".format(x),
    hlp=lambda x: "{:.2f}".format(x),
    surface_roughness=lambda x: "{:.2f}".format(x),
    et=lambda x: "{:.0f}".format(x),
    swp=lambda x: "{:.2f}".format(x),
    biodiversity_cwd=lambda x: "{:.1f}".format(x),
    biodiversity_big_trees=lambda x: "{:.0f}".format(x),
    biodiversity_size_diversity=lambda x: "{:.0f}".format(x),
    biodiversity_combined=lambda x: "{:.2f}".format(x),
    albedo_jul=lambda x: "{:.2f}".format(x),
    albedo_jan=lambda x: "{:.2f}".format(x),
)

# 2010
iter_csv = pd.read_csv('/home/konni/Documents/phd/data/simulation_inputs/other_inputs/global_forest_ST_1871-2011_CORRECTED_1y_mask_gridlist_europe_forexclim.txt',
                       delim_whitespace=True, iterator=True)
stand_type_fractions = pd.concat([chunk[(chunk['year'] == 2010)] for chunk in iter_csv]).set_index(['Lon', 'Lat'])
stand_type_fractions = stand_type_fractions.loc[~stand_type_fractions.index.duplicated(keep='first')]

iter_forest_csv = pd.read_csv(
    '/home/konni/Documents/phd/data/simulation_inputs/other_inputs/global_nat_for_LC_1871-2011_CORRECTED_1y_mask_gridlist_europe_CORRECT.txt_capitalized',
    delim_whitespace=True, iterator=True)
forest_type_fractions = pd.concat([chunk for chunk in iter_forest_csv]).set_index(['Lon', 'Lat', 'Year'])
forest_type_fractions = forest_type_fractions.loc[~forest_type_fractions.index.duplicated(keep='first')]


def plot_es_performance(simulation_files, different_portfolios, used_simulations, es_of_interest=es_variable_names.keys(), lons_lats_of_interest=None, add_title=''):
    table_mean = pd.DataFrame(index=used_simulations + ['present'] + list(different_portfolios.keys()))
    table_rcp26_full = pd.DataFrame(index=used_simulations + ['present'] + list(different_portfolios.keys()))
    table_rcp45_full = pd.DataFrame(index=used_simulations + ['present'] + list(different_portfolios.keys()))
    table_rcp60_full = pd.DataFrame(index=used_simulations + ['present'] + list(different_portfolios.keys()))
    table_rcp85_full = pd.DataFrame(index=used_simulations + ['present'] + list(different_portfolios.keys()))
    table_mean_tex = pd.DataFrame(index=used_simulations + ['present'] + list(different_portfolios.keys()))
    table_rcp26_tex = pd.DataFrame(index=used_simulations + ['present'] + list(different_portfolios.keys()))
    table_rcp45_tex = pd.DataFrame(index=used_simulations + ['present'] + list(different_portfolios.keys()))
    table_rcp60_tex = pd.DataFrame(index=used_simulations + ['present'] + list(different_portfolios.keys()))
    table_rcp85_tex = pd.DataFrame(index=used_simulations + ['present'] + list(different_portfolios.keys()))

    n_porfolios = len(different_portfolios)
    portfolio_colors = [np.repeat((idx / (n_porfolios + 1)), 3) for idx, val in enumerate(different_portfolios.keys())]
    colorz = [color_discrete_map[x] for x in used_simulations] + portfolio_colors + [color_discrete_map['present']]

    for es_name, es_variable_name in es_variable_names.items():

        if es_of_interest and es_name not in es_of_interest:
            continue

        fig, axs = plt.subplots(1, 6, figsize=(30, 4))

        max_val = 0
        min_val = 0

        table_for_es = []

        for idx, rcp in enumerate(['rcp26', 'rcp45', 'rcp60', 'rcp85']):

            vals = {}

            val_per_gc = None
            if es_name != 'biodiversity_combined':
                val_per_gc = get_simulations_and_optimal_value_2100(simulation_files, rcp, used_simulations, es_name, es_variable_name, different_portfolios, lons_lats_of_interest=lons_lats_of_interest)
            else:
                bio_cwd_per_gc = get_combined_biodiv_indicator(different_portfolios, lons_lats_of_interest, rcp, simulation_files, used_simulations)
                val_per_gc = bio_cwd_per_gc

            valtmp = val_per_gc.copy()  # explicitly copying here makes sure that the drop of NAs below does not mess with our original data!
            for simulation in used_simulations:
                if simulation == 'toBe':
                    val_per_gc = val_per_gc.dropna(subset=['toBe'], inplace=False)
                # values for cpools and harvests are per total modeled area, for the others, we need to take the forest fraction into account.
                # e.g. two equally sized GCs, one has 10% forest cover, the other 90%.
                # then if ET is 0mm in the first but 100mm in the second, the total should not be 50mm, but 90mm, because 90% of the forest area is the one in GC2
                # for ET, surface roughness, SWP, and biodiversity, we're only considering changes of these values in the forests, so when aggregating, we need to
                # use only the forest fraction for those variables.
                if es_name not in ['cpool', 'mitigation', 'harvest', 'hlp']:
                    vals[simulation] = analysis.compute_avg_val_over_forested_gridcell_areas(val_per_gc, simulation) if not val_per_gc.empty else np.nan
                else:
                    vals[simulation] = analysis.compute_avg_val_over_gridcell_areas(val_per_gc, simulation) if not val_per_gc.empty else np.nan
                val_per_gc = valtmp.copy()

            for portfolio_name, portfolio in different_portfolios.items():
                if es_name not in ['cpool', 'mitigation', 'harvest', 'hlp']:
                    vals[portfolio_name] = analysis.compute_avg_val_over_forested_gridcell_areas(val_per_gc, portfolio_name)
                else:
                    vals[portfolio_name] = analysis.compute_avg_val_over_gridcell_areas(val_per_gc, portfolio_name)

            # there are slight differences in present values since we take 2000-2009 average as present value, but the rcp scenarios start already in 2006.
            # these differences are only notable for et
            # we take the four years from rcp60 since it is in the very middle of the rcps.
            if es_name != 'biodiversity_combined':
                vals['present'] = get_present_val(es_name, lons_lats_of_interest, 'rcp60', simulation_files, val_per_gc)
            else:
                vals['present'] = 0.0

            df = pd.DataFrame(vals.items(), columns=['simulation', 'value'])
            df['value'].plot(ax=axs[idx], kind='bar', color=colorz, alpha=0.8, width=0.8)

            max_for_rcp = max(vals.values())
            min_for_rcp = min(vals.values())
            if max_for_rcp > max_val:
                max_val = max_for_rcp
            if min_for_rcp < min_val:
                min_val = min_for_rcp

            table_for_es.append(pd.DataFrame(vals, index=[es_name]))

            axs[idx].set_title(rcp)
            axs[idx].spines['right'].set_visible(False)
            if es_name == 'swp':
                axs[idx].spines['bottom'].set_visible(False)
                axs[idx].tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off

            else:
                axs[idx].spines['top'].set_visible(False)
                axs[idx].tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)  # labels along the bottom edge are off

        for idx, rcp in enumerate(['rcp26', 'rcp45', 'rcp60', 'rcp85']):
            axs[idx].set_ylim([min_val, max_val])

        axs[-1].set_axis_off()
        axs[0].set_ylabel(variable_units_unconverted[es_name])

        patches = [Patch(color=color_discrete_map[sim], label=sim) for sim in used_simulations] + [Patch(color=portfolio_colors[idx], label=portfolio_name) for idx, portfolio_name
                                                                                                   in enumerate(different_portfolios.keys())] + [
                      Patch(color=color_discrete_map['present'], label='present')]
        fig.legend(labels=used_simulations + list(different_portfolios.keys()) + ['present'], handles=patches, loc='right')

        title = 'Absolute values Europe ' + es_name + ' ' + add_title
        fig.suptitle(title)

        df_concat = pd.concat(table_for_es)

        table_mean[es_name] = df_concat.mean()
        std = df_concat.std()

        table_rcp26 = table_for_es[0]
        table_rcp45 = table_for_es[1]
        table_rcp60 = table_for_es[2]
        table_rcp85 = table_for_es[3]

        table_rcp26_full[es_name] = pd.DataFrame(table_rcp26).loc[es_name, :]
        table_rcp45_full[es_name] = pd.DataFrame(table_rcp45).loc[es_name, :]
        table_rcp60_full[es_name] = pd.DataFrame(table_rcp60).loc[es_name, :]
        table_rcp85_full[es_name] = pd.DataFrame(table_rcp85).loc[es_name, :]

        table_rcp26_tex[es_name] = pd.DataFrame(table_rcp26).loc[es_name, :].apply(accuracy_for_table[es_name])
        table_rcp45_tex[es_name] = pd.DataFrame(table_rcp45).loc[es_name, :].apply(accuracy_for_table[es_name])
        table_rcp60_tex[es_name] = pd.DataFrame(table_rcp60).loc[es_name, :].apply(accuracy_for_table[es_name])
        table_rcp85_tex[es_name] = pd.DataFrame(table_rcp85).loc[es_name, :].apply(accuracy_for_table[es_name])

        table_mean_tex[es_name] = table_mean[es_name].apply(accuracy_for_table[es_name])
        table_mean_tex.loc[~table_mean_tex.index.isin(['present']), es_name] += "\\tiny{($\pm$" + std.loc[~std.index.isin(['present'])].apply(accuracy_for_table_std[es_name]) + ')}'

        if es_name == 'biodiversity_combined':
            # present day value makes no sense for biodiv, it is normalized by RCPs.
            table_rcp26_tex[es_name]['present'] = '--$^{**}$'
            table_rcp45_tex[es_name]['present'] = '--$^{**}$'
            table_rcp60_tex[es_name]['present'] = '--$^{**}$'
            table_rcp85_tex[es_name]['present'] = '--$^{**}$'
            table_mean_tex[es_name]['present'] = '--$^{**}$'

        bp = axs[4].boxplot(df_concat.values, patch_artist=True)
        for idx, box in enumerate(bp['boxes']):
            box.set(facecolor=colorz[idx])

        plt.show()

    table_mean_tex = pd.concat([pd.DataFrame({k: v for k, v in variable_units_unconverted.items() if k in es_of_interest}, index=['unit']), table_mean_tex])
    table_rcp26_tex = pd.concat([pd.DataFrame({k: v for k, v in variable_units_unconverted.items() if k in es_of_interest}, index=['unit']), table_rcp26_tex])
    table_rcp45_tex = pd.concat([pd.DataFrame({k: v for k, v in variable_units_unconverted.items() if k in es_of_interest}, index=['unit']), table_rcp45_tex])
    table_rcp60_tex = pd.concat([pd.DataFrame({k: v for k, v in variable_units_unconverted.items() if k in es_of_interest}, index=['unit']), table_rcp60_tex])
    table_rcp85_tex = pd.concat([pd.DataFrame({k: v for k, v in variable_units_unconverted.items() if k in es_of_interest}, index=['unit']), table_rcp85_tex])

    table_mean_tex = table_mean_tex.rename(table_names, axis=1)
    table_rcp26_tex = table_rcp26_tex.rename(table_names, axis=1)
    table_rcp45_tex = table_rcp45_tex.rename(table_names, axis=1)
    table_rcp60_tex = table_rcp60_tex.rename(table_names, axis=1)
    table_rcp85_tex = table_rcp85_tex.rename(table_names, axis=1)

    table_mean_tex = table_mean_tex.rename(lambda x: x.replace('_', ' '), axis=1)
    table_rcp26_tex = table_rcp26_tex.rename(lambda x: x.replace('_', ' '), axis=1)
    table_rcp45_tex = table_rcp45_tex.rename(lambda x: x.replace('_', ' '), axis=1)
    table_rcp60_tex = table_rcp60_tex.rename(lambda x: x.replace('_', ' '), axis=1)
    table_rcp85_tex = table_rcp85_tex.rename(lambda x: x.replace('_', ' '), axis=1)

    return table_mean_tex, table_rcp26_tex, table_rcp45_tex, table_rcp60_tex, table_rcp85_tex, table_rcp26_full, table_rcp45_full, table_rcp60_full, table_rcp85_full


def get_combined_biodiv_indicator(different_portfolios, lons_lats_of_interest, rcp, simulation_files, used_simulations, min_year=2100, max_year=2130):
    bio_cwd_per_gc = get_simulations_and_optimal_value_2100(simulation_files, rcp, used_simulations, 'biodiversity_cwd',
                                                            'ForestCWD', different_portfolios,
                                                            lons_lats_of_interest=lons_lats_of_interest, min_year=min_year, max_year=max_year)
    bio_large_tree_per_gc = get_simulations_and_optimal_value_2100(simulation_files, rcp, used_simulations,
                                                                   'biodiversity_big_trees', 'thick_trees',
                                                                   different_portfolios,
                                                                   lons_lats_of_interest=lons_lats_of_interest, min_year=min_year, max_year=max_year)
    bio_size_diversity_per_gc = get_simulations_and_optimal_value_2100(simulation_files, rcp, used_simulations,
                                                                       'biodiversity_size_diversity', 'size_diversity',
                                                                       different_portfolios,
                                                                       lons_lats_of_interest=lons_lats_of_interest, min_year=min_year, max_year=max_year)
    max_cwd = bio_cwd_per_gc[used_simulations].max(axis=1, numeric_only=True)
    max_large_trees = bio_large_tree_per_gc[used_simulations].max(axis=1, numeric_only=True)
    max_size_diversity = bio_size_diversity_per_gc[used_simulations].max(axis=1, numeric_only=True)
    min_cwd = bio_cwd_per_gc[used_simulations].min(axis=1, numeric_only=True)
    min_large_trees = bio_large_tree_per_gc[used_simulations].min(axis=1, numeric_only=True)
    min_size_diversity = bio_size_diversity_per_gc[used_simulations].min(axis=1, numeric_only=True)
    bio_cwd_per_gc[used_simulations] = (bio_cwd_per_gc[used_simulations].subtract(min_cwd, axis=0)).div(
        max_cwd.subtract(min_cwd, axis=0), axis=0)
    bio_large_tree_per_gc[used_simulations] = (
        bio_large_tree_per_gc[used_simulations].subtract(min_large_trees, axis=0)).div(
        max_large_trees.subtract(min_large_trees, axis=0), axis=0)
    bio_size_diversity_per_gc[used_simulations] = (
        bio_size_diversity_per_gc[used_simulations].subtract(min_size_diversity, axis=0)).div(
        max_size_diversity.subtract(min_size_diversity, axis=0), axis=0)
    # there are nans when there was a division by 0 above, in which case biodiv is the same value for all management options.
    # we can just set it to 0, see corresponding test.
    bio_cwd_per_gc[used_simulations] = bio_cwd_per_gc[used_simulations].fillna(0)
    bio_large_tree_per_gc[used_simulations] = bio_large_tree_per_gc[used_simulations].fillna(0)
    bio_size_diversity_per_gc[used_simulations] = bio_size_diversity_per_gc[used_simulations].fillna(0)

    bio_cwd_per_gc[used_simulations] = (
        bio_cwd_per_gc[used_simulations].add(bio_large_tree_per_gc[used_simulations], axis=0).add(
            bio_size_diversity_per_gc[used_simulations], axis=0)).div(3)
    # at this point we have also in the portfolio columns the values, we can take those values and normalize them
    # they are already weighted raw numbers
    for portfolio_name, portfolio_values in different_portfolios.items():
        bio_cwd_per_gc[portfolio_name] = (bio_cwd_per_gc[portfolio_name].subtract(min_cwd, axis=0)).div(
            max_cwd.subtract(min_cwd, axis=0), axis=0)
        bio_large_tree_per_gc[portfolio_name] = (
            bio_large_tree_per_gc[portfolio_name].subtract(min_large_trees, axis=0)).div(
            max_large_trees.subtract(min_large_trees, axis=0), axis=0)
        bio_size_diversity_per_gc[portfolio_name] = (
            bio_size_diversity_per_gc[portfolio_name].subtract(min_size_diversity, axis=0)).div(
            max_size_diversity.subtract(min_size_diversity, axis=0), axis=0)

        bio_cwd_per_gc[portfolio_name].replace([np.inf, -np.inf], np.nan, inplace=True)
        bio_large_tree_per_gc[portfolio_name].replace([np.inf, -np.inf], np.nan, inplace=True)
        bio_size_diversity_per_gc[portfolio_name].replace([np.inf, -np.inf], np.nan, inplace=True)
        bio_cwd_per_gc[portfolio_name] = bio_cwd_per_gc[portfolio_name].fillna(0)
        bio_large_tree_per_gc[portfolio_name] = bio_large_tree_per_gc[portfolio_name].fillna(0)
        bio_size_diversity_per_gc[portfolio_name] = bio_size_diversity_per_gc[portfolio_name].fillna(0)

        bio_cwd_per_gc[portfolio_name] = (
            bio_cwd_per_gc[portfolio_name].add(bio_large_tree_per_gc[portfolio_name], axis=0).add(
                bio_size_diversity_per_gc[portfolio_name], axis=0)).div(3)
    return bio_cwd_per_gc


def get_simulations_and_optimal_value_2100(simulations, rcp, used_simulations, es_name, variable_name, different_portfolios, lons_lats_of_interest, min_year=2100, max_year=2130):
    basepath = simulations[rcp]['base']
    es_value_per_gc = get_es_value(basepath, es_name, lons_lats_of_interest=lons_lats_of_interest, rcp=rcp, min_year=min_year, max_year=max_year)
    for portfolio_name in different_portfolios.keys():
        es_value_per_gc[portfolio_name] = 0.0
    for sim_name in used_simulations:
        es_value_per_gc[sim_name] = 0.0

    some_portfolio_result = list(different_portfolios.values())[0]

    for simulation in used_simulations:
        basepath = simulations[rcp][simulation]

        # The future values for the managements are computed for all the given grid cells, including some cells where we would deem this management infeasible.
        # Those are then of course not used in the optimization however. But here we just look at how the ESI would perform if we applied this management everywhere, even though
        # it might result in no forest cover by the end of the century for some grid cells. The managements will work in most grid cells anyway so this should not bias too much.
        simulation_es_values = get_es_value(basepath, es_name, lons_lats_of_interest=lons_lats_of_interest, rcp=rcp, min_year=2100, max_year=2130)
        # The number of grid cells where toBe is feasible is very low, so aggregating over all grid cells makes no sense here, so in this particular case we exclude them.
        if simulation == 'toBe':
            lons_lats_of_interest_be_feasible = some_portfolio_result[some_portfolio_result['feasible_toBe']].index
            simulation_es_values = get_es_value(basepath, es_name, lons_lats_of_interest=lons_lats_of_interest_be_feasible, rcp=rcp, min_year=2100, max_year=2130)

        es_value_per_gc[simulation] = simulation_es_values[variable_name]

        for portfolio_name, portfolio_values in different_portfolios.items():

            if lons_lats_of_interest is not None:
                portfolio_values = portfolio_values.reindex(lons_lats_of_interest)

            assert len(portfolio_values[~portfolio_values['has_forest']]) == 0, "grid cells without managed forest need to have been removed beforehand"

            es_val_weighted_by_portfolio_fraction = simulation_es_values[variable_name] * portfolio_values[simulation]
            es_val_weighted_by_portfolio_fraction.loc[es_val_weighted_by_portfolio_fraction.isna()] = 0
            es_value_per_gc[portfolio_name] += es_val_weighted_by_portfolio_fraction

            # technically overrides but does not matter, this is of course independent of the portfolio
            es_value_per_gc['forest_frac'] = portfolio_values['forest_frac']

    return es_value_per_gc


def get_es_value(basepath, es_name, min_year=2100, max_year=2130, lons_lats_of_interest=None, rcp=None):
    data = None
    if es_name == 'cpool':
        data = get_cpool(basepath, min_year, max_year, lons_lats_of_interest)
    elif es_name == 'hlp':
        data = get_harvests_via_species_file(basepath, min_year, max_year, lons_lats_of_interest, slow_only=True)
    elif es_name == 'surface_roughness':
        data = get_surface_roughness(basepath, min_year, max_year, lons_lats_of_interest)
    elif es_name == 'harvest':
        data = get_harvests_via_species_file(basepath, min_year, max_year, lons_lats_of_interest)
    elif es_name == 'mitigation':
        # since, when aggregating, we normally only want to show the results where discounting was on in the optimization
        # for the case of aggregating data of portfolios that were created without discounting, this needs to be manually adapted.
        assert rcp is not None, 'If you really want to aggregate the values without substitution discounting, adapt the code. This is to prevent wrong combination of portfolio creation and aggregation'
        data = get_new_total_mitigation(get_fluxes_with_new_harvests(basepath, 2010, max_year, lons_lats_of_interest), get_cpool(basepath, 2010, max_year, lons_lats_of_interest), discounting=rcp)
    elif es_name == 'swp':
        data = get_forest_swp(basepath, min_year, max_year, lons_lats_of_interest, only_forest=False)
    elif es_name == 'et':
        data = get_forest_et(basepath, min_year, max_year, lons_lats_of_interest)
    elif es_name == 'biodiversity_cwd':
        data = get_biodiversity_cwd(basepath, min_year, max_year, lons_lats_of_interest)
    elif es_name == 'biodiversity_size_diversity' or es_name == 'biodiversity_big_trees':
        data = get_biodiversity_tree_sizes(basepath, min_year, max_year, lons_lats_of_interest)
    elif es_name == 'albedo_jul' or es_name == 'albedo_jan':
        data = get_albedo(basepath, min_year, max_year, lons_lats_of_interest)

    data = data.groupby(['Lon', 'Lat']).mean()
    data['Year'] = min_year

    return data.loc[~data.index.duplicated(keep='first')]


def get_present_val(es_name, lons_lats_of_interest, rcp, simulation_files, val_per_gc, min_year=2000, max_year=2010):
    present_val_per_gc = get_es_value(simulation_files[rcp]['base'], es_name, lons_lats_of_interest=lons_lats_of_interest, rcp=rcp, min_year=min_year, max_year=max_year)

    assert present_val_per_gc.index.equals(val_per_gc.index), 'indices do not match!'

    present_val_per_gc['forest_frac'] = val_per_gc['forest_frac']

    if es_name not in ['cpool', 'mitigation', 'harvest', 'hlp']:
        return analysis.compute_avg_val_over_forested_gridcell_areas(present_val_per_gc, es_variable_names[es_name])
    else:
        return analysis.compute_avg_val_over_gridcell_areas(present_val_per_gc, es_variable_names[es_name])


def aggregate_to_europe(portfolioss, used_simulations, title='Europe'):

    fig, axs = plt.subplots(1, 4, figsize=(10, 4))

    aggregate_to_europe_given_fig(fig, axs, portfolioss, used_simulations, title)

    plt.show()


def aggregate_to_europe_given_fig(axs, portfolioss, used_simulations, subaxes_titles=True, compute_areas=False, translations=None):
    if 'has_forest' in portfolioss.columns:
        portfolioss = portfolioss[portfolioss['has_forest']]

    if compute_areas:
        europe_wide_share = analysis.compute_avg_val_over_forested_gridcell_areas(portfolioss,
                                                                     used_simulations + ['total_coniferous', 'total_broadleaved', 'total_high', 'base_broadleaved_frac',
                                                                                         'base_coniferous_frac', 'feasible'])
    else:
        europe_wide_share = analysis.compute_avg_val_over_forested_gridcell_areas_when_area_already_in_df(portfolioss,
                                                                                                          used_simulations + ['total_coniferous', 'total_broadleaved', 'total_high', 'base_broadleaved_frac',
                                                                                                                              'base_coniferous_frac', 'feasible'])

    wedgeprops = dict(width=0.7)
    textprops = {'fontsize': 10}
    radius = 0.95
    textdist = 1.15

    europe_wide_share['infeasible'] = 1 - europe_wide_share['feasible']
    axs[0].pie([europe_wide_share[key] for key in used_simulations], startangle=90, radius=radius, pctdistance=textdist, autopct='%.0f%%', textprops=textprops,
               wedgeprops=wedgeprops, colors=[color_discrete_map[key] for key in used_simulations])
    patches = axs[1].pie(europe_wide_share[['base_coniferous_frac', 'base_broadleaved_frac', ]], startangle=90, radius=radius, autopct='%.0f%%', wedgeprops=wedgeprops, textprops=textprops, pctdistance=textdist,
                         colors=['green', 'lightgreen'])[0]

    patches[1].set_hatch('///')
    patches2 = axs[2].pie(europe_wide_share[['total_coniferous', 'total_broadleaved']], startangle=90, radius=radius, autopct='%.0f%%', wedgeprops=wedgeprops, textprops=textprops, pctdistance=textdist,
                          colors=['green', 'lightgreen'])[0]
    patches2[1].set_hatch('///')

    title_fontsize = 10
    if subaxes_titles:
        axs[0].set_title('Optimal Portfolio', fontsize=title_fontsize)
        axs[1].set_title('Present Day Share NL/BL', fontsize=title_fontsize)
        axs[2].set_title('Future Share NL/BL', fontsize=title_fontsize)


    return europe_wide_share


def aggregate_to_europe_given_fig_bar_multiple(axs, pffss, used_simulations, subaxes_titles=True, ticks=False, perc_fontsize=10):

    bernd = None

    for name, portfolioss in pffss.items():

        if 'has_forest' in portfolioss.columns:
            portfolioss = portfolioss[portfolioss['has_forest']]

        europe_wide_share = analysis.compute_avg_val_over_forested_gridcell_areas_when_area_already_in_df(portfolioss, used_simulations + ['total_coniferous', 'total_broadleaved', 'total_high', 'base_broadleaved_frac',
                                                                                                                              'base_coniferous_frac', 'feasible'])
        europe_wide_share['infeasible'] = 1 - europe_wide_share['feasible']

        if bernd is None:
            bernd = europe_wide_share.to_frame(name=name)
            bernd['present'] = 0.0
            bernd.loc['total_coniferous', 'present'] = bernd.loc['base_coniferous_frac', name]
            bernd.loc['total_broadleaved', 'present'] = bernd.loc['base_broadleaved_frac', name]
        else:
            bernd = bernd.join(europe_wide_share.to_frame(name=name))

    bernd.transpose()[used_simulations].loc[list(pffss.keys())].plot.bar(ax=axs[0], stacked=True, color=[color_discrete_map[key] for key in used_simulations], legend=False, width=0.9, align='center')

    bl_nl_share_graphs = ['present'] + list(pffss.keys())
    bar_plot3 = bernd.transpose()[['total_coniferous', 'total_broadleaved']].loc[bl_nl_share_graphs].plot.bar(ax=axs[1], stacked=True, color=['green', 'lightgreen'], legend=False, width=0.9)

    for idx, val in enumerate(bl_nl_share_graphs):
        bar_plot3.containers[1][idx].set_hatch('///')

    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=-30, ha='left')
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=-30, ha='left')

    for i, rect in enumerate(axs[1].patches):
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()

        label_x = x + width / 2
        label_y = y + height / 2

        colorr = 'white' if i <= len(pffss.items()) else 'black'

        if height > 0.001:
            axs[1].text(label_x, label_y, f'{100*height:.0f}%', ha='center', va='center', fontsize=perc_fontsize, color=colorr)

    n_pffs = len(pffss)
    for i, rect in enumerate(axs[0].patches):
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()

        label_x = x + width / 2
        label_y = y + height / 2

        # managements 1, 2, 5 need to be white (base, unmanaged, toNe)
        white_idxs = np.concatenate([np.arange(n_pffs*i, n_pffs*(i+1)) for i in [1, 2, 5]])
        colorr = 'white' if i in white_idxs else 'black'

        if height >= 0.09:
            axs[0].text(label_x, label_y, f'{100*height:.0f}%', ha='center', va='center', fontsize=perc_fontsize, color=colorr)




    title_fontsize = 10
    if subaxes_titles:
        axs[0].set_title('Optimized Portfolio', fontsize=title_fontsize)
        axs[1].set_title('Share needle-/broad-leaved', fontsize=title_fontsize)

    axs[0].get_yaxis().set_ticks([])
    axs[1].get_yaxis().set_ticks([])

    if not ticks:
        axs[0].get_xaxis().set_ticks([])
        axs[1].get_xaxis().set_ticks([])

    for axx in axs:
        axx.spines['top'].set_visible(False)
        axx.spines['right'].set_visible(False)
        axx.spines['bottom'].set_visible(False)
        axx.spines['left'].set_visible(False)

    return europe_wide_share







def aggregate_to_europe_given_fig_bar(axs, portfolioss, portfolioss2, used_simulations, subaxes_titles=True, ticks=False, perc_fontsize=10):
    if 'has_forest' in portfolioss.columns:
        portfolioss = portfolioss[portfolioss['has_forest']]

    europe_wide_share = analysis.compute_avg_val_over_forested_gridcell_areas_when_area_already_in_df(portfolioss,
                                                                                                      used_simulations + ['total_coniferous', 'total_broadleaved', 'total_high', 'base_broadleaved_frac',
                                                                                                                          'base_coniferous_frac', 'feasible'])
    europe_wide_share2 = analysis.compute_avg_val_over_forested_gridcell_areas_when_area_already_in_df(portfolioss2,
                                                                                                      used_simulations + ['total_coniferous', 'total_broadleaved', 'total_high', 'base_broadleaved_frac',
                                                                                                                          'base_coniferous_frac', 'feasible'])

    europe_wide_share['infeasible'] = 1 - europe_wide_share['feasible']

    bernd = europe_wide_share.to_frame(name='default').join(europe_wide_share2.to_frame(name='harv-constraint'))
    bernd['present'] = 0.0
    bernd.loc['total_coniferous', 'present'] = bernd.loc['base_coniferous_frac', 'default']
    bernd.loc['total_broadleaved', 'present'] = bernd.loc['base_broadleaved_frac', 'default']

    bernd.transpose()[used_simulations].loc[['default', 'harv-constraint']].plot.bar(ax=axs[0], stacked=True, color=[color_discrete_map[key] for key in used_simulations], legend=False, width=0.9, align='center')


    bar_plot3 = bernd.transpose()[['total_coniferous', 'total_broadleaved']].loc[['present', 'default', 'harv-constraint']].plot.bar(ax=axs[1], stacked=True, color=['green', 'lightgreen'], legend=False, width=0.9)
    bar_plot3.containers[1][0].set_hatch('///')
    bar_plot3.containers[1][1].set_hatch('///')
    bar_plot3.containers[1][2].set_hatch('///')

    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=-30, ha='left')
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=-30, ha='left')

    for i, rect in enumerate(axs[1].patches):
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()

        label_x = x + width / 2
        label_y = y + height / 2

        colorr = 'white' if i <= 2 else 'black'

        if height > 0.001:
            axs[1].text(label_x, label_y, f'{100*height:.0f}%', ha='center', va='center', fontsize=perc_fontsize, color=colorr)

    for i, rect in enumerate(axs[0].patches):
        height = rect.get_height()
        width = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()

        label_x = x + width / 2
        label_y = y + height / 2

        colorr = 'white' if i in [2, 3, 4, 5, 10, 11] else 'black'

        if height >= 0.09:
            axs[0].text(label_x, label_y, f'{100*height:.0f}%', ha='center', va='center', fontsize=perc_fontsize, color=colorr)




    title_fontsize = 10
    if subaxes_titles:
        axs[0].set_title('Optimized Portfolio', fontsize=title_fontsize)
        axs[1].set_title('Share NL/BL', fontsize=title_fontsize)

    axs[0].get_yaxis().set_ticks([])
    axs[1].get_yaxis().set_ticks([])

    if not ticks:
        axs[0].get_xaxis().set_ticks([])
        axs[1].get_xaxis().set_ticks([])

    for axx in axs:
        axx.spines['top'].set_visible(False)
        axx.spines['right'].set_visible(False)
        axx.spines['bottom'].set_visible(False)
        axx.spines['left'].set_visible(False)

    return europe_wide_share



# When the optimization says we should manage BD/NE in proportion 40/60, this does not mean that by the end of the period we're looking at the conversion to this fraction is completed.
# It could be that only 90% of forests have been converted by that time.
# This method accounts for this.
def account_for_conversion_not_happened_until_end_of_period(portfolioss):
    converted = pandas_helper.read_for_years(
        '/home/konni/Documents/konni/projekte/phd/runs/cluster_runs/european_climate_regions_full_spinup_que_ile_rob_tcmax/base_rcp45/converted_fraction.out',
        2130, 2130).set_index(['Lon', 'Lat'])
    assert converted.index.is_unique, "Converted Index is not unique! Maybe grid cells are duplicated?"
    portfolioss['total_broadleaved'] = portfolioss['total_broadleaved'] * converted['Fraction'] + portfolioss[
        'base_broadleaved_frac'] * (1 - converted['Fraction'])
    portfolioss['total_coniferous'] = portfolioss['total_coniferous'] * converted['Fraction'] + portfolioss[
        'base_coniferous_frac'] * (1 - converted['Fraction'])

    return portfolioss


def luyssaert_values(simulation_list, portfolio_fractions, gridcell):
    forest_fractions = pd.read_csv(forest_fraction_file, delim_whitespace=True)

    forest_fractions = forest_fractions.query('Lon==' + str(gridcell[0]) + ' and Lat==' + str(gridcell[1])).set_index(['year'])

    forest_fractions['Total'] = forest_fractions['ForestBD'] + forest_fractions['ForestBE'] + forest_fractions['ForestND'] + forest_fractions['ForestNE']

    total_forest_fraction = forest_fractions['Total'].values[-1]

    if total_forest_fraction == 0:
        raise NoManagedForestError('Gridcell without managed forest! ' + str(gridcell[0]) + ', ' + str(gridcell[1]))

    base_broadleaved_frac = (forest_fractions.iloc[-1, :]['ForestBD'] + forest_fractions.iloc[-1, :]['ForestBE']) / total_forest_fraction
    base_coniferous_frac = (forest_fractions.iloc[-1, :]['ForestND'] + forest_fractions.iloc[-1, :]['ForestNE']) / total_forest_fraction

    total_broadleaved = 0.0
    total_coniferous = 0.0
    total_grass = 0.0
    total_high = 0.0
    total_coppice = 0.0
    total_unmanaged = 0.0

    for idx, simulation in enumerate(simulation_list):
        if simulation == 'toBd' or simulation == 'toBe':
            total_broadleaved += portfolio_fractions[idx]
            total_high += portfolio_fractions[idx]
        elif simulation == 'toCoppice':
            total_broadleaved += portfolio_fractions[idx]
            total_coppice += portfolio_fractions[idx]
        elif simulation == 'toNe':
            total_coniferous += portfolio_fractions[idx]
            total_high += portfolio_fractions[idx]
        elif simulation == 'base':
            total_broadleaved += portfolio_fractions[idx] * base_broadleaved_frac
            total_coniferous += portfolio_fractions[idx] * base_coniferous_frac
            total_high += portfolio_fractions[idx]
        elif simulation == 'baseRefrain' or simulation == 'unmanaged':
            total_broadleaved += portfolio_fractions[idx] * base_broadleaved_frac
            total_coniferous += portfolio_fractions[idx] * base_coniferous_frac
            total_unmanaged += portfolio_fractions[idx]
        elif simulation == 'toGrass':
            total_grass += portfolio_fractions[idx]
        else:
            raise ValueError('A simulation was used that was not accounted for in the species distribution computation. It was: ' + str(simulation))

    return dict(base_broadleaved_frac=base_broadleaved_frac, base_coniferous_frac=base_coniferous_frac, total_unmanaged=total_unmanaged, total_broadleaved=total_broadleaved,
                total_coniferous=total_coniferous, total_grass=total_grass, total_high=total_high, total_coppice=total_coppice, forest_frac=total_forest_fraction)
