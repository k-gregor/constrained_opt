# run_dir = '/home/konni/Documents/phd/runs/cluster_runs/dist_no_fire_in_managed/'
# run_dir = '/home/konni/Documents/phd/runs/cluster_runs/increase_dist/'
run_dir = '/home/konni/Documents/phd/runs/cluster_runs/increase_dist_natural/'

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

used_simulations = ['toBd', 'toBe', 'toCoppice', 'toNe', 'base', 'unmanaged']
boundary_simulations = []
rcpppp = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
n_rcp = len(rcpppp)
