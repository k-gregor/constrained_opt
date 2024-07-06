import mpl_toolkits.basemap # Needs matplotlib 3.2.2 or lower!
import matplotlib.pyplot as plt
from pandas_helper import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection
import numpy as np
import optimization_europe_aggregation as og

from pft_to_forest_type import *

import importlib
import constants

importlib.reload(constants)


[minLon, minLat, maxLon, maxLat] = [-15, 35, 32.1, 75]  # Europe
# [minLon, minLat, maxLon, maxLat] = [-15, 20, 45, 75]  # Europe for tmerc projection





def get_map(ax1):
    m = mpl_toolkits.basemap.Basemap(projection='cyl', resolution=None,
                                     llcrnrlat=minLat, llcrnrlon=minLon, urcrnrlat=maxLat, urcrnrlon=maxLon,
                                     lat_0=(maxLat + minLat) / 2, lon_0=(minLon + maxLon) / 2, ax=ax1)
    m.shadedrelief()
    return m


def plot_on_map(x, y, z, ax1, cmap='jet', norm=None, label=None, s=10, vmin=None, vmax=None, relief=True, countrylines=True, projection='cyl', min_lon=minLon, min_lat=minLat, max_lon=maxLon, max_lat=maxLat):
    m = mpl_toolkits.basemap.Basemap(projection=projection, resolution='l',
                                     llcrnrlat=min_lat, llcrnrlon=min_lon, urcrnrlat=max_lat, urcrnrlon=max_lon,
                                     lat_0=(maxLat + minLat) / 2, lon_0=(minLon + maxLon) / 2, ax=ax1)
    if relief:
        m.shadedrelief(alpha=0.5)
    sc = m.scatter(x, y, c=z, cmap=cmap, alpha=1, latlon=True, s=s, norm=norm, label=label, vmin=vmin, vmax=vmax, marker='s')
    if countrylines:
        m.drawcountries()
    m.drawcoastlines(zorder=1)
    return sc

def plot_forest_types_for_one_file(simulation, filepath, year_of_interest, ax):
    iter_csv = pd.read_csv(filepath, iterator=True, chunksize=1000, delim_whitespace=True)
    output = pd.concat([chunk[chunk['Year'] == year_of_interest] for chunk in iter_csv])

    output['ForestNE'] = output.loc[:,ne_pfts].sum(axis=1)
    output['ForestND'] = output.loc[:,nd_pfts].sum(axis=1)
    output['ForestBE'] = output.loc[:,be_pfts].sum(axis=1)
    output['ForestBD'] = output.loc[:,bd_pfts].sum(axis=1)
    output['Shrub'] = output.loc[:,shrub_pfts].sum(axis=1)

    output['MostCommonType'] = output.loc[:,['ForestNE', 'ForestND', 'ForestBE', 'ForestBD', 'Shrub']].idxmax(axis=1)

    m = get_map(ax)

    colors = dict(ForestNE = 'r', ForestND = 'g', ForestBE = 'b', ForestBD = 'k', Shrub = 'y')

    for forest_type, dff in output.groupby("MostCommonType"):
        sc = m.scatter(dff['Lon'].values, dff['Lat'].values, color=colors[forest_type], alpha=1, latlon=True, s=10, label=forest_type)
    ax.set_title('Main forest types ' + simulation + ' ' + str(year_of_interest))
    ax.legend(loc=2)


def plot_forest_types(files, year_of_interest):
    fig, ax = plt.subplots(1, 5, figsize = (30, 15))
    idx = 0
    for simulation, filepath in files.items():
        plot_forest_types_for_one_file(simulation, filepath, year_of_interest, ax[idx])
        idx += 1





def plot_species_distribution(filepath, lambda_opt=0.0, additional_info=''):
    portfolios = pd.read_csv(filepath)

    fig, ax = plt.subplots(figsize=(15, 15))
    fig.tight_layout()
    plt.title(r'Present and Future Species Share, $\lambda=' + str(lambda_opt) + r'$ ' + additional_info, fontsize=20)

    bm = mpl_toolkits.basemap.Basemap(llcrnrlat=minLat,
                                      llcrnrlon=minLon,
                                      urcrnrlat=maxLat,
                                      urcrnrlon=maxLon,
                                      ax=ax,
                                      resolution='l', projection='cyl', lon_0=(maxLat + minLat) / 2, lat_0=(minLon + maxLon) / 2)

    bm.fillcontinents(color='lightyellow', zorder=0)
    bm.drawcoastlines(color='gray', linewidth=0.3, zorder=2)
    bm.drawcountries(linewidth=0.5)

    for i, row in portfolios.iterrows():
        x1, y1 = bm(row['Lon'], row['Lat'])  # get data coordinates for plotting

        ax_h = inset_axes(ax, width=0.5,
                          height=0.5,
                          loc=10,
                          bbox_to_anchor=(x1, y1),
                          bbox_transform=ax.transData,
                          borderpad=0,
                          axes_kwargs={'alpha': 0.25, 'visible': True})

        size = 0.5

        ax_h.pie(row[['total_broadleaved', 'total_coniferous']], radius=1, startangle=90, colors=['lightgreen', 'green'],
                 wedgeprops=dict(width=size, edgecolor='w'))
        ax_h.pie(row[['base_broadleaved_frac', 'base_coniferous_frac']], radius=1 - size, startangle=90, colors=['lightgreen', 'green'],
                 wedgeprops=dict(width=size, edgecolor='w'))

        ax_h.set_facecolor('b')
        ax_h.axis('off')

    patch0 = Patch(color='lightgreen', label='Broadleaved')
    patch1 = Patch(color='green', label='Coniferous')
    ax.legend(handles=[patch0, patch1], loc=2)

    plt.show()


def plot_ipcc_map(figsize=(15, 15), color=False):
    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout()
    bm = mpl_toolkits.basemap.Basemap(llcrnrlat=minLat,
                                      llcrnrlon=minLon,
                                      urcrnrlat=maxLat,
                                      urcrnrlon=maxLon,
                                      ax=ax,
                                      resolution='l', projection='cyl', lon_0=(maxLat + minLat) / 2, lat_0=(minLon + maxLon) / 2)
    bm.drawcoastlines(color='gray', linewidth=0.3, zorder=2)
    bm.drawcountries(linewidth=0.5)
    bm.readshapefile("/home/konni/Documents/phd/data/climate_zones/climate_europe/EnSv8/enz_v8_mapped", 'climate_zones', drawbounds=False)
    cmap = plt.cm.RdYlBu_r
    norm = plt.Normalize(1, 15)

    zone_map_to_ipcc_region = dict(
        ATC='Atlantic',
        ATN='Atlantic',
        LUS='Atlantic',
        ALN='Alpine',
        ALS='Alpine',
        PAN='Continental',
        CON='Continental',
        BOR='Northern',
        NEM='Northern',
        MDS='Southern',
        MDM='Southern',
        MDN='Southern',
        ANA='Southern',
    )

    ipcc_region_to_color = dict(
            Atlantic=('grey', 1),
            AtlanticN=('dimgrey', 1),
            Alpine=('black', 3),
            Continental=('whitesmoke', 1),
            Northern=('darkgray', 2),
            Southern=('lightgrey', 1),
    )

    if color:
        ipcc_region_to_color = dict(
            Atlantic=('darkred', 1),
            AtlanticN=('pink', 1),
            Alpine=('red', 3),
            Continental=('lightblue', 1),
            Northern=('orange', 2),
            Southern=('gray', 1),
        )

    patch_collections = dict(
        Atlantic=[],
        AtlanticN=[],
        Continental=[],
        Northern=[],
        Southern=[],
        Alpine=[],
    )

    ipcc_style = True
    alpha = 0.8

    for info, shape in zip(bm.climate_zones_info, bm.climate_zones):
        if ipcc_style:
            color = ipcc_region_to_color[zone_map_to_ipcc_region[info['EnZ_name']]]
            patch_collections[zone_map_to_ipcc_region[info['EnZ_name']]].append(Polygon(np.array(shape), True))
        else:
            color = cmap(norm(info['EnZ']))
            patch = Polygon(np.array(shape), True, facecolor=color, alpha=alpha)
            ax.add_patch(patch)

    if ipcc_style:
        for region, color in ipcc_region_to_color.items():
            ax.add_collection(PatchCollection(patch_collections[region], facecolor=color[0], zorder=color[1], alpha=alpha))

    return fig, ax, bm, patch_collections


def plot_portfolios_on_map(portfolios, lambda_opt, additional_info='', print_cell_info=False, plot_luyssaert=False, save_for_paper_path=None):
    fig, ax, bm, patch_collections = plot_ipcc_map()

    if plot_luyssaert:
        plt.title(r'Optimal Portfolios (Luyssaert plot), $\lambda=' + str(lambda_opt) + r'$ ' + additional_info, fontsize=20)
    else:
        plt.title(r'Optimal Portfolios, $\lambda=' + str(lambda_opt) + r'$ ' + additional_info, fontsize=20)

    used_simulations = list(portfolios.columns[(portfolios.columns.get_loc('gc_region')+1):portfolios.columns.get_loc('base_broadleaved_frac')])

    portfolios['region'] = 'na'

    no_region_counter = 0

    for i, row in portfolios.iterrows():
        if 'has_forest' in portfolios.columns and row['has_forest'] is False:
            continue

        found_region = False
        for region_name in patch_collections.keys():
            for patch in patch_collections[region_name]:
                if patch.contains_point((row['Lon'], row['Lat'])):
                    found_region = True
                    portfolios.at[i, 'region'] = region_name
                elif patch.contains_point((row['Lon']-0.25, row['Lat']-0.25)) or patch.contains_point((row['Lon']-0.25, row['Lat']+0.25)) or patch.contains_point((row['Lon']+0.25, row['Lat']+0.25)) or patch.contains_point((row['Lon']+0.25, row['Lat']-0.25)):
                    found_region = True
                    portfolios.at[i, 'region'] = region_name
                elif patch.contains_point((row['Lon'], row['Lat']-0.25)) or patch.contains_point((row['Lon'], row['Lat']+0.25)) or patch.contains_point((row['Lon']+0.25, row['Lat'])) or patch.contains_point((row['Lon']+0.25, row['Lat'])):
                    found_region = True
                    portfolios.at[i, 'region'] = region_name

        if not found_region:
            if row['Lon'] >= 31.75:
                # our cells expand to east more than the climate region map, but they're all continental cells, these are all within Ukraine.
                portfolios.at[i, 'region'] = 'Continental'
            elif row['Lon'] <= 10:
                # 4 cells in France and NW Germany
                portfolios.at[i, 'region'] = 'Atlantic'
            elif row['Lat'] == 52.25 and (row['Lon'] == 23.75 or row['Lon'] == 24.75):
                portfolios.at[i, 'region'] = 'Continental'
            elif row['Lat'] == 57.25 and (row['Lon'] == 23.75 or row['Lon'] == 24.75):
                portfolios.at[i, 'region'] = 'Northern'
            else:
                print("did not find a region :(", row['Lon'], row['Lat'])
                no_region_counter += 1

        plot_pie_cell(ax, bm, plot_luyssaert, print_cell_info, row, used_simulations)

    if plot_luyssaert:
        legend_patches = [Patch(color='red', label='Unmanaged'), Patch(color='orange', label='Coppice'), Patch(color='blue', label='High-stand')]
    else:
        legend_patches = [Patch(color=constants.color_discrete_map[strategy], label=strategy) for strategy in used_simulations]
    ax.legend(handles=legend_patches, loc=2)

    print('did not find a region for', no_region_counter, 'cells')

    plt.show()

    if save_for_paper_path:
        fig.savefig(save_for_paper_path, dpi=fig.dpi, bbox_inches='tight')


    return portfolios


def plot_pie_cell(ax, bm, plot_luyssaert, print_cell_info, row, used_simulations):
    alpha = 1
    x1, y1 = bm(row['Lon'], row['Lat'])  # get data coordinates for plotting
    ax_h = inset_axes(ax, width=0.5,
                      height=0.5,
                      loc=10,
                      bbox_to_anchor=(x1, y1),
                      bbox_transform=ax.transData,
                      borderpad=0,
                      axes_kwargs={'alpha': alpha, 'visible': True})
    if not row['feasible']:
        ax_h.pie([1.0], startangle=90, radius=1, wedgeprops=dict(width=0.7, edgecolor='w', alpha=alpha),
                 colors=[constants.color_discrete_map['infeasible']])
    else:
        if plot_luyssaert:
            ax_h.pie(row[['total_unmanaged', 'total_coppice', 'total_high']], startangle=90, radius=1,
                     wedgeprops=dict(width=0.7, edgecolor='w', alpha=alpha),
                     colors=['red', 'orange', 'blue'])
        else:
            ax_h.pie(row[used_simulations], startangle=90, radius=1,
                     wedgeprops=dict(width=0.7, edgecolor='w', alpha=alpha),
                     colors=[constants.color_discrete_map[key] for key in used_simulations])
    if print_cell_info:
        ax_h.set_title(str(row['Lon']) + ' ' + str(row['Lat']), fontsize=7)


def plot_europe_with_pies(pff, fontsize=10, save_for_paper_path=None):
    fig, map_ax, bm, ignore2 = plot_ipcc_map(figsize=(12, 12))

    plt.rcParams['hatch.linewidth'] = 2
    plt.rcParams['hatch.color'] = 'limegreen'

    # different order is no problem here, only has influence on the order they are shown in the pie graphs
    used_simulations = ['toCoppice', 'base', 'unmanaged', 'toBe', 'toBd', 'toNe']

    positions = dict(
        Atlantic=(.05, .45),
        Continental=(.5, .35),
        Alpine=(.55, .72),
        Southern=(.25, .12),
        Northern=(.6, .52),
    )
    shares = {}

    letters = dict(Atlantic='c', Continental='e', Alpine='b', Southern='f', Northern='d')

    for region_name in ['Atlantic', 'Continental', 'Alpine', 'Southern', 'Northern']:

        n_cells_in_region = len(pff[pff['region'] == region_name])

        if n_cells_in_region == 0:
            continue

        width = 0.11
        inlay_fraction = 0.92
        aa = plt.axes([positions[region_name][0], positions[region_name][1], 3.15*width, width], facecolor='w', alpha=0.3)

        ax = plt.axes([positions[region_name][0] + 0.005, positions[region_name][1], inlay_fraction * width, inlay_fraction * width])
        ax2 = plt.axes([positions[region_name][0]+width + 0.01, positions[region_name][1], inlay_fraction * width, inlay_fraction * width])
        ax3 = plt.axes([positions[region_name][0]+2*width + 0.015, positions[region_name][1], inlay_fraction * width, inlay_fraction * width])
        axs2 = [ax, ax2, ax3]
        region_title = letters[region_name] + ') ' + region_name + ', n=' + str(n_cells_in_region)
        aa.set_title(region_title, fontsize=16, fontweight='bold')
        aa.axes.xaxis.set_visible(False)
        aa.axes.yaxis.set_visible(False)
        aa.axes.set_alpha(0.4)
        aa.set_alpha(0.4)

        shares[region_name] = og.aggregate_to_europe_given_fig(axs2, pff[pff['region'] == region_name], used_simulations, subaxes_titles=False)

    width = 0.15
    most_left = 0.06
    most_top = 0.7
    aa = plt.axes([most_left, most_top, 3*width, width + 0.03], facecolor='w', alpha=0.4)
    ax = plt.axes([most_left, most_top, width, width])
    ax2 = plt.axes([most_left+width, most_top, width, width])
    ax3 = plt.axes([most_left+2*width, most_top, width, width])
    axs2 = [ax, ax2, ax3]
    aa.axes.xaxis.set_visible(False)
    aa.axes.yaxis.set_visible(False)
    aa.set_title('a) Europe Total, n=' + str(len(pff)), fontsize=16, fontweight='bold')
    shares['Europe'] = og.aggregate_to_europe_given_fig(axs2, pff, used_simulations, subaxes_titles=True)

    legend_patches = [Patch(color=constants.color_discrete_map[strategy], label=strategy) for strategy in used_simulations]
    leg1 = map_ax.legend(handles=legend_patches, loc='lower right', title='Portfolios', bbox_to_anchor=(1.0, 0.1))


    legend_patches = [Patch(color='green', label='Total Share Needleleaved'), Patch(facecolor='lightgreen', label='Total Share Broadleaved (BE & BD)', hatch='///')]
    map_ax.legend(handles=legend_patches, loc='lower right', title='Species Shares', title_fontsize='large',)

    map_ax.add_artist(leg1)

    plt.show()

    if save_for_paper_path:
        fig.savefig(save_for_paper_path, dpi=fig.dpi, bbox_inches='tight')

    return pd.DataFrame(shares)


def plot_europe_with_pies_german(pff, fontsize=10, save_for_paper_path=None):
    fig, map_ax, bm, ignore2 = plot_ipcc_map(figsize=(15, 15))

    plt.rcParams['hatch.linewidth'] = 2
    plt.rcParams['hatch.color'] = 'limegreen'

    # different order is no problem here, only has influence on the order they are shown in the pie graphs
    used_simulations = ['toCoppice', 'base', 'unmanaged', 'toBe', 'toBd', 'toNe']
    used_simulations_german = ['Niederwald', 'BAU', 'Unbewirtschaftet', 'Laubwald (immergrün)', 'Laubwald (sommergrün)', 'Nadelwald']
    regions = ['Atlantic', 'Continental', 'Alpine', 'Southern', 'Northern']
    regions_german = ['Atlantisch', 'Kontinental', 'Alpin', 'Südeuropa', 'Nordeuropa']

    translations = {
    }

    for idx, sim in enumerate(used_simulations):
        translations[sim] = used_simulations_german[idx]
    for idx, sim in enumerate(regions):
        translations[sim] = regions_german[idx]

    positions = dict(
        Atlantic=(.05, .45),
        Continental=(.5, .35),
        Alpine=(.55, .72),
        Southern=(.25, .12),
        Northern=(.6, .52),
    )
    shares = {}

    letters = dict(Atlantic='c', Continental='e', Alpine='b', Southern='f', Northern='d')

    for region_name in regions:

        n_cells_in_region = len(pff[pff['region'] == region_name])

        if n_cells_in_region == 0:
            continue

        width = 0.11
        inlay_fraction = 0.92
        aa = plt.axes([positions[region_name][0], positions[region_name][1], 3.15*width, width], facecolor='w', alpha=0.3)

        ax = plt.axes([positions[region_name][0] + 0.005, positions[region_name][1], inlay_fraction * width, inlay_fraction * width])
        ax2 = plt.axes([positions[region_name][0]+width + 0.01, positions[region_name][1], inlay_fraction * width, inlay_fraction * width])
        ax3 = plt.axes([positions[region_name][0]+2*width + 0.015, positions[region_name][1], inlay_fraction * width, inlay_fraction * width])
        axs2 = [ax, ax2, ax3]
        region_title = letters[region_name] + ') ' + translations[region_name] + ', n=' + str(n_cells_in_region)
        aa.set_title(region_title, fontsize=16, fontweight='bold')
        aa.axes.xaxis.set_visible(False)
        aa.axes.yaxis.set_visible(False)
        aa.axes.set_alpha(0.4)
        aa.set_alpha(0.4)

        shares[region_name] = og.aggregate_to_europe_given_fig(axs2, pff[pff['region'] == region_name], used_simulations, subaxes_titles=False, compute_areas=True)

    width = 0.15
    most_left = 0.06
    most_top = 0.7
    aa = plt.axes([most_left, most_top, 3*width, width + 0.03], facecolor='w', alpha=0.4)
    ax = plt.axes([most_left, most_top, width, width])
    ax2 = plt.axes([most_left+width, most_top, width, width])
    ax3 = plt.axes([most_left+2*width, most_top, width, width])
    axs2 = [ax, ax2, ax3]
    aa.axes.xaxis.set_visible(False)
    aa.axes.yaxis.set_visible(False)
    aa.set_title('a) Europa, n=' + str(len(pff)), fontsize=16, fontweight='bold')
    shares['Europe'] = og.aggregate_to_europe_given_fig(axs2, pff, used_simulations, subaxes_titles=True, compute_areas=True)

    legend_patches = [Patch(color=constants.color_discrete_map[strategy], label=translations[strategy]) for strategy in used_simulations]
    leg1 = map_ax.legend(handles=legend_patches, loc='lower right', title='Portfolios', bbox_to_anchor=(1.0, 0.1))


    legend_patches = [Patch(color='green', label='Gesamtanteil Nadelwald'), Patch(facecolor='lightgreen', label='Gesamtanteil Laubwald', hatch='///')]
    map_ax.legend(handles=legend_patches, loc='lower right', title='Waldanteile', title_fontsize='large',)

    map_ax.add_artist(leg1)

    plt.show()

    if save_for_paper_path:
        fig.savefig(save_for_paper_path, dpi=4*fig.dpi, bbox_inches='tight')

    return pd.DataFrame(shares)



def plot_europe_with_bars_and_compare(pffs, fontsize=10, save_for_paper_path=None):
    fig, map_ax, bm, ignore2 = plot_ipcc_map(figsize=(12, 12))

    plt.rcParams['hatch.linewidth'] = 2
    plt.rcParams['hatch.color'] = 'limegreen'

    # different order is no problem here, only has influence on the order they are shown in the pie graphs
    used_simulations = ['toCoppice', 'base', 'unmanaged', 'toBe', 'toBd', 'toNe']

    positions = dict(
        Atlantic=(.05, .45),
        Continental=(.5, .35),
        Alpine=(.55, .72),
        Southern=(.35, .12),
        Northern=(.6, .52),
    )
    shares = {}

    letters = dict(Atlantic='c', Continental='e', Alpine='b', Southern='f', Northern='d')

    for region_name in ['Atlantic', 'Continental', 'Alpine', 'Southern', 'Northern']:
        one_pff = next(iter(pffs.values()))
        n_cells_in_region = len(one_pff[one_pff['region'] == region_name])

        if n_cells_in_region == 0:
            continue

        width = 0.135
        inlay_fraction = 0.92
        container = plt.axes([positions[region_name][0], positions[region_name][1], 2.15*width, width], facecolor=None, alpha=0.3)

        container.spines['top'].set_visible(False)
        container.spines['right'].set_visible(False)
        container.spines['bottom'].set_visible(True)
        container.spines['left'].set_visible(False)

        container.patch.set_alpha(0.0)

        ax = plt.axes([positions[region_name][0] + 0.005, positions[region_name][1], inlay_fraction * width, inlay_fraction * width], facecolor='w', alpha=0.3)
        ax2 = plt.axes([positions[region_name][0]+width + 0.01, positions[region_name][1], inlay_fraction * width, inlay_fraction * width], facecolor=None)


        ax.patch.set_alpha(0.0)
        ax2.patch.set_alpha(0.0)

        axs2 = [ax, ax2]
        region_title = letters[region_name] + ') ' + region_name + ', n=' + str(n_cells_in_region)
        container.set_title(region_title, fontsize=16, fontweight='bold')
        container.axes.xaxis.set_visible(False)
        container.axes.yaxis.set_visible(False)
        container.axes.set_alpha(0.4)
        container.set_alpha(0.4)

        shares[region_name] = og.aggregate_to_europe_given_fig_bar_multiple(axs2, {key: val[val['region']==region_name] for key, val in pffs.items()}, used_simulations, subaxes_titles=False, perc_fontsize=8)

    width = 0.17
    most_left = 0.05
    most_top = 0.68
    container = plt.axes([most_left, most_top, 2.1*width, width + 0.03], facecolor=None, alpha=0.4)

    container.spines['top'].set_visible(False)
    container.spines['right'].set_visible(False)
    container.spines['bottom'].set_visible(True)
    container.spines['left'].set_visible(False)

    ax = plt.axes([most_left+0.01, most_top, width, width])
    ax2 = plt.axes([most_left+width, most_top, width, width])
    axs2 = [ax, ax2]
    container.axes.xaxis.set_visible(False)
    container.axes.yaxis.set_visible(False)
    container.set_title('a) Europe Total, n=' + str(len(next(iter(pffs.values())))), fontsize=16, fontweight='bold')
    shares['Europe'] = og.aggregate_to_europe_given_fig_bar_multiple(axs2, pffs, used_simulations, subaxes_titles=True, ticks=True, perc_fontsize=11)

    legend_patches = [Patch(color=constants.color_discrete_map[strategy], label=strategy) for strategy in used_simulations]
    leg1 = map_ax.legend(handles=legend_patches, loc='lower right', title='Portfolios', bbox_to_anchor=(1.0, 0.1))


    legend_patches = [Patch(color='green', label='Total Share Needleleaved'), Patch(facecolor='lightgreen', label='Total Share Broadleaved (BE & BD)', hatch='///')]
    map_ax.legend(handles=legend_patches, loc='lower right', title='Species Shares', title_fontsize='large',)

    map_ax.add_artist(leg1)

    plt.show()

    if save_for_paper_path:
        fig.savefig(save_for_paper_path, dpi=fig.dpi, bbox_inches='tight')

    return pd.DataFrame(shares)



