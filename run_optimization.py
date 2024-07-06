import optimize_gridcell as og
import numpy as np
import sys
from datetime import datetime
import pandas as pd
import optimization_preparation as oprep
import pandas_helper as ph
import analyze_lpj_output as alo
import pickle

used_simulations = ['base', 'toBd', 'toBe', 'toCoppice', 'toNe', 'unmanaged']
boundary_simulations = []
rcpppp = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
n_rcp = len(rcpppp)


varrrs =['harvest', 'hlp', 'mitigation', 'et', 'surface_roughness', 'swp']
biodiv_vars = ['biodiversity_combined']
varrrs += biodiv_vars
weights = np.ones(len(varrrs))

sims = ['base', 'toBd', 'toBe', 'toCoppice', 'toNe', 'unmanaged']
constraints = []





def get_all_feasible_cells_from_paper(gridlist, simulations):

    gcs = pd.read_csv(gridlist, delim_whitespace=True, comment='#', header=None, names=['Lon', 'Lat'])
    gcs['gc_region']=None

    # skip cells where not much conversion is happening --> does not make sense to create a portfolio
    harv2 = oprep.get_fluxes_with_new_harvests(simulations['rcp45']['base'], 1990, 2010, gcs[['Lon', 'Lat']].apply(tuple, axis=1))
    harv2 = harv2.groupby(['Lon', 'Lat'], as_index=True).mean()
    toskip = harv2[harv2['total_harv'] < 0.0001].index
    conversion = ph.read_for_years(simulations['rcp45']['base'] + 'converted_fraction.out', 2130, 2130).set_index(['Lon', 'Lat'])
    skip_due_to_low_conversion = conversion.loc[conversion['Fraction'] < 0.25].index

    gcs = gcs.set_index(['Lon', 'Lat'])
    gcs = gcs.loc[set(gcs.index) - set(toskip) - set(skip_due_to_low_conversion)]


    luyssaert_variables = ['base_broadleaved_frac', 'base_coniferous_frac', 'total_unmanaged', 'total_broadleaved', 'total_coniferous', 'total_grass', 'total_coppice', 'total_high']
    for sim in sims + luyssaert_variables:
        gcs[sim]=0

    gcs['feasible'] = True

    # these GCs are not caught by the above check for some reason, exclude manually...
    infeasible_cells = [(27.75, 45.75), (29.75, 45.75), (8.25, 59.25), (8.25, 59.75)]
    gcs = gcs.loc[[idxval for idxval in gcs.index if idxval not in infeasible_cells]]

    return gcs.reset_index()




def add_area_and_forest_frac2(gcs, basepath):
    forest_fracs = ph.read_for_years(basepath + 'active_fraction_forest.out', 2010, 2010,
                                     lons_lats_of_interest=gcs[['Lon', 'Lat']].apply(tuple, axis=1)).set_index(
        ['Lon', 'Lat'])
    gcs = gcs.set_index(['Lon', 'Lat'])
    gcs['forest_frac'] = forest_fracs['C3_gr']
    gcs = gcs.reset_index()
    gcs['area'] = gcs['Lat'].apply(
        lambda x: alo.get_area_for_lon_and_lat_with_formulas_specific_gc_size(x, lat_frac=0.5, lon_frac=0.5))
    return gcs


def optimize_multiple_gridcells(rcps, simulations_paths, used_simulations, es, constraints, lambda_opt, weights, gcs, min_year, max_year, file_basepath=None, plot=False, gc_location=None, discounting=True, plot_for_paper_name=None, minmax=True, gc_lambda=False, use_dataset=None, store=False, only_forest_swp=False, unmanaged_constraint=0.0, unmanaged_constraint_every_cell=0.0, harvest_constraint_m3=0.0, hlp_constraint_m3=0.0, cpool_constraint=0.0):
    return og.optimize_multiple_gridcells(rcps, simulations_paths, used_simulations, es, constraints, lambda_opt, weights, gcs, min_year, max_year, file_basepath=None, plot=False, gc_location=None, discounting=True, plot_for_paper_name=None, minmax=minmax, gc_lambda=gc_lambda, use_dataset=use_dataset, store=store, only_forest_swp=False, unmanaged_constraint=0.0, unmanaged_constraint_every_cell=0.0, harvest_constraint_m3=0.0, hlp_constraint_m3=0.0, cpool_constraint=0.0)


def run_optimization(run_dir, gridlist, minmax=False, dataset_prepared_for_optimization=None, unmanaged_constraint=0.0, unmanaged_constraint_every_cell=0.0, harvest_constraint_m3=0.0, hlp_constraint_m3=0.0, cpool_constraint=0.0):

    rcp26_simulations = dict(
        base=run_dir + 'base_rcp26/',
        toBd=run_dir + 'tobd_rcp26/',
        toBe=run_dir + 'tobe_rcp26/',
        toCoppice=run_dir + 'toCoppice_rcp26/',
        toNe=run_dir + 'tone_rcp26/',
        unmanaged=run_dir + 'unmanaged_rcp26/',
    )

    rcp45_simulations = dict(
        base=run_dir + 'base_rcp45/',
        toBd=run_dir + 'tobd_rcp45/',
        toBe=run_dir + 'tobe_rcp45/',
        toCoppice=run_dir + 'toCoppice_rcp45/',
        toNe=run_dir + 'tone_rcp45/',
        unmanaged=run_dir + 'unmanaged_rcp45/',
    )

    rcp60_simulations = dict(
        base=run_dir + 'base_rcp60/',
        toBd=run_dir + 'tobd_rcp60/',
        toBe=run_dir + 'tobe_rcp60/',
        toCoppice=run_dir + 'toCoppice_rcp60/',
        toNe=run_dir + 'tone_rcp60/',
        unmanaged=run_dir + 'unmanaged_rcp60/',
    )

    rcp85_simulations = dict(
        base=run_dir + 'base_rcp85/',
        toBd=run_dir + 'tobd_rcp85/',
        toBe=run_dir + 'tobe_rcp85/',
        toCoppice=run_dir + 'toCoppice_rcp85/',
        toNe=run_dir + 'tone_rcp85/',
        unmanaged=run_dir + 'unmanaged_rcp85/',
    )


    simulations = {'rcp26': rcp26_simulations, 'rcp45': rcp45_simulations, 'rcp60': rcp60_simulations,
                   'rcp85': rcp85_simulations}


    gcs = add_area_and_forest_frac2(get_all_feasible_cells_from_paper(gridlist, simulations), basepath=simulations['rcp45']['base'])

    print('### Starting,', datetime.now())
    res = og.optimize_multiple_gridcells(rcpppp, simulations, sims, varrrs, constraints, 0.2, weights, gcs, 2100, 2130, file_basepath=None, plot=False, gc_location=None, discounting=True, plot_for_paper_name=None, minmax=minmax, gc_lambda=True, use_dataset=dataset_prepared_for_optimization, store=False, only_forest_swp=True, unmanaged_constraint=unmanaged_constraint, unmanaged_constraint_every_cell=unmanaged_constraint_every_cell, harvest_constraint_m3=harvest_constraint_m3, hlp_constraint_m3=hlp_constraint_m3, cpool_constraint=cpool_constraint)
    print('### Done.', datetime.now())

    with open('optimized_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.dat', 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return res


if __name__ == '__main__':
    args = sys.argv[1:]

    run_optimization(args[0], args[1])