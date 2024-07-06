import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sin, cos, pi, radians, sqrt


[minLon, minLat, maxLon, maxLat] = [-15, 30, 45, 75]  # Europe

MAX_YEAR = 2130

lon_lat_to_area = {}


# calculation from https://en.wikipedia.org/wiki/Longitude#Length_of_a_degree_of_longitude
def compute_length_of_longitude(degree_lat):
    a = 6378137.0
    b = 6356752.3142
    e = sqrt((a**2 - b**2)/a**2)
    return (pi * a * cos(radians(degree_lat))) / (180 * sqrt(1 - e ** 2 * (sin(radians(degree_lat)) ** 2)))


# calculation from https://en.wikipedia.org/wiki/Longitude#Length_of_a_degree_of_longitude
def compute_length_of_latitude(degree_lat):
    l_lat = np.abs(111132.954 - 559.822 * cos(2 * radians(degree_lat)) + 1.175 * cos(4 * radians(degree_lat)))
    return l_lat

# trapezoid shape
# lat_frac: length of one gridcell (latitude-wise), e.g. 0.5 or in LfU: 0.0590848336423
def get_area_for_lon_and_lat_with_formulas_specific_gc_size(lat, lat_frac = 0.5, lon_frac = 0.5):
    if lat not in lon_lat_to_area:
        lat_length = compute_length_of_latitude(lat) * lat_frac
        lon_top_length = compute_length_of_longitude(lat + lat_frac / 2)
        lon_bottom_length = compute_length_of_longitude(lat - lat_frac / 2)
        small_length = min(lon_top_length, lon_bottom_length) * lon_frac
        long_length = max(lon_top_length, lon_bottom_length) * lon_frac
        area = lat_length * small_length + lat_length * (long_length - small_length) / 2 #last 2 is for the triangle area
        lon_lat_to_area[lat] = area

    return lon_lat_to_area[lat]


def compute_total_c_pool_in_giga_tons_new(lat_lon_total_values_per_m2, field_of_interest):
    assert lat_lon_total_values_per_m2.index.is_unique, "Index is not unique! Maybe grid cells are duplicated?"

    if 'Year' in lat_lon_total_values_per_m2.columns:
        assert is_unique(lat_lon_total_values_per_m2['Year']), "Values of multiple years in aggregation!"

    assert 'Lon' in lat_lon_total_values_per_m2.index.names and 'Lat' in lat_lon_total_values_per_m2.index.names

    lat_lon_total_values_per_m2['area'] = lat_lon_total_values_per_m2.index.map(lambda x: get_area_for_lon_and_lat_with_formulas_specific_gc_size(lat=x[1]))
    computed = lat_lon_total_values_per_m2.apply(lambda x: x['area'] * x[field_of_interest], axis=1).sum()

    kg_to_Gt = 1000 ** 4
    return np.sum(computed) / kg_to_Gt


def compute_total_value(lat_lon_total_values_per_m2, field_of_interest, lon_frac=0.5, lat_frac=0.5):
    assert lat_lon_total_values_per_m2.index.is_unique, "Index is not unique! Maybe grid cells are duplicated?"

    if 'Year' in lat_lon_total_values_per_m2.columns:
        assert is_unique(lat_lon_total_values_per_m2['Year']), "Values of multiple years in aggregation!"

    assert 'Lon' in lat_lon_total_values_per_m2.index.names and 'Lat' in lat_lon_total_values_per_m2.index.names

    lat_lon_total_values_per_m2['area'] = lat_lon_total_values_per_m2.index.map(lambda x: get_area_for_lon_and_lat_with_formulas_specific_gc_size(lat=x[1], lat_frac=lat_frac, lon_frac=lon_frac))
    computed = lat_lon_total_values_per_m2.apply(lambda x: x['area'] * x[field_of_interest], axis=1).sum()
    return computed


def compute_total_gridcell_value_multiple_years(lat_lon_total_values_per_m2, field_of_interest, lon_frac=0.5, lat_frac=0.5):
    assert lat_lon_total_values_per_m2.index.is_unique, "Index is not unique! Maybe grid cells are duplicated?"

    # assert 'Lon' in lat_lon_total_values_per_m2.index.names and 'Lat' in lat_lon_total_values_per_m2.index.names and 'Year' in lat_lon_total_values_per_m2.index.names
    assert lat_lon_total_values_per_m2.index.names == ['Lon', 'Lat', 'Year']

    lat_lon_total_values_per_m2['area'] = lat_lon_total_values_per_m2.index.map(lambda x: get_area_for_lon_and_lat_with_formulas_specific_gc_size(lat=x[1], lat_frac=lat_frac, lon_frac=lon_frac))
    computed = lat_lon_total_values_per_m2.apply(lambda x: x['area'] * x[field_of_interest], axis=1)
    return computed


def is_unique(s):
    a = s.to_numpy()  # s.values (pandas<0.24)
    return (a[0] == a).all()


def compute_avg_val_over_gridcell_areas(lat_lon_total_values_per_m2, fields_of_interest, formula=True, skip_index_name_check=False, lat_frac = 0.5, lon_frac = 0.5):
    assert lat_lon_total_values_per_m2.index.is_unique, "Index is not unique! Maybe grid cells are duplicated?"

    if 'Year' in lat_lon_total_values_per_m2.columns:
        assert is_unique(lat_lon_total_values_per_m2['Year']), "Values of multiple years in aggregation!"

    if not skip_index_name_check:  # external files might have names like lon, lat instead of Lon, Lat
        assert lat_lon_total_values_per_m2.index.names == ['Lon', 'Lat'], "Index does not contain Lon and Lat or contains more than that."

    lat_lon_total_values_per_m2.loc[:, 'area'] = lat_lon_total_values_per_m2.index.map(lambda x: get_area_for_lon_and_lat_with_formulas_specific_gc_size(lat=x[1], lat_frac=lat_frac, lon_frac=lon_frac))
    weighted = lat_lon_total_values_per_m2.apply(lambda x: x['area'] * x[fields_of_interest], axis=1).sum()
    total_area = lat_lon_total_values_per_m2.apply(lambda x: x['area'], axis=1).sum()
    return weighted/total_area


def compute_avg_val_over_forested_gridcell_areas(lat_lon_total_values_per_m2, fields_of_interest, formula=True, skip_index_name_check=False):
    assert lat_lon_total_values_per_m2.index.is_unique, "Index is not unique! Maybe grid cells are duplicated?"

    if 'Year' in lat_lon_total_values_per_m2.columns:
        assert is_unique(lat_lon_total_values_per_m2['Year']), "Values of multiple years in aggregation!"

    if not skip_index_name_check:  # external files might have names like lon, lat instead of Lon, Lat
        assert lat_lon_total_values_per_m2.index.names == ['Lon', 'Lat']

    lat_lon_total_values_per_m2_copy = lat_lon_total_values_per_m2.copy()

    lat_lon_total_values_per_m2_copy.loc[:, 'area'] = lat_lon_total_values_per_m2_copy.index.map(lambda x: get_area_for_lon_and_lat_with_formulas_specific_gc_size(lat=x[1], lat_frac=0.5, lon_frac=0.5))

    # lat_lon_total_values_per_m2['forest_area'] = lat_lon_total_values_per_m2.loc[:, 'area'] * lat_lon_total_values_per_m2.loc[:, 'forest_frac']

    return compute_avg_val_over_forested_gridcell_areas_when_area_already_in_df(lat_lon_total_values_per_m2_copy, fields_of_interest)


def compute_avg_val_over_forested_gridcell_areas_when_area_already_in_df(lat_lon_total_values_per_m2_copy, fields_of_interest):
    weighted = lat_lon_total_values_per_m2_copy.apply(lambda x: x['area'] * x['forest_frac'] * x[fields_of_interest], axis=1).sum()
    total_area = lat_lon_total_values_per_m2_copy.apply(lambda x: x['area'] * x['forest_frac'], axis=1).sum()
    return weighted / total_area


def get_time_series_of_total_cpool(output, column_of_interest):
    return output.groupby('Year').apply(lambda x: compute_total_c_pool_in_giga_tons_new(x, column_of_interest))

threshold = 0
forbidden_names = ['Lon', 'Lat', 'Year', 'Forest_sum', 'Natural_sum', 'Barren_sum', 'Total']
def get_n_pfts(row):
    sum_pfts = 0
    for label, value in row.items():
        if label not in forbidden_names and value > threshold:
            sum_pfts += 1
    return sum_pfts


def get_mean_pfts_for_group(group):
    return group.apply(lambda row: get_n_pfts(row), axis=1).mean()


def plot_average_number_of_pfts(files, ax, start_year, end_year):
    for simulation, filepath in files.items():
        iter_csv = pd.read_csv(filepath, iterator=True, chunksize=1000, delim_whitespace=True)
        output = pd.concat([chunk[(chunk['Year'] >= start_year) & (chunk['Year'] <= end_year)] for chunk in iter_csv])

        mean_pfts = output.groupby('Year').apply(lambda group: get_mean_pfts_for_group(group))
        mean_pfts.plot(label="Simulation " + simulation, ax=ax, legend=True)

    ax.set_title('Average Number of PFTs per Gridcell (PFTs with density above threshold ' + str(threshold) + ' ind/m2')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of PFTs')
    ax.grid(b=True, which='major', color='#CCCCCC', linestyle='-')
    
    
def plot_c_pools(cpool_files, lon_of_interest, lat_of_interest, max_year=MAX_YEAR):

    fig1, ax1 = plt.subplots(1, 5, figsize=(25, 5))
    idx = 0
    for simulation, filepath in cpool_files.items():
        iter_csv = pd.read_csv(filepath, iterator=True, chunksize=1000, delim_whitespace=True, usecols=lambda x: x in ['Lon', 'Lat', 'Total', 'VegC', 'HarvSlowC', 'Year'])
        output = pd.concat([chunk[(chunk['Year'] >= 2000) & (chunk['Year'] <= max_year) & (chunk['Lat'] == lat_of_interest) & (chunk['Lon'] == lon_of_interest)] for chunk in iter_csv])
        for cpool in ['VegC', 'HarvSlowC', 'Total']:
            output.plot('Year', cpool, ax=ax1[idx], label = cpool, sharex=ax1[0], ylim=[0, 12])
        ax1[idx].set_title(simulation)
        idx+=1
    plt.ylim([0, 12])
    plt.show()
    
        
def plot_harvest_statistics(harv_files, lon_of_interest, lat_of_interest):
    fig1, ax1 = plt.subplots(1, 5, figsize=(25, 5))
    idx = 0
    for simulation, filepath in harv_files.items():
        iter_csv = pd.read_csv(filepath, iterator=True, chunksize=1000, delim_whitespace=True)
        output = pd.concat([chunk[(chunk['Year'] >= 2000) & (chunk['Year'] <= MAX_YEAR)
                                  & (chunk['Lat'] == lat_of_interest) & (chunk['Lon'] == lon_of_interest)
                            ] for chunk in iter_csv])
        for harv_type in ['Natural', 'Barren', 'ForestNE', 'ForestND', 'ForestBE', 'ForestBD']:
            output.plot('Year', harv_type, ax=ax1[idx], label = harv_type, ylim=[0, 0.5])
        ax1[idx].set_title(simulation)
        idx+=1


# Average over a 10-year period to average out seasonality
def get_monthly_data_accumulated_to_year_averaged_over_10_years(filepath, year_one):
    return get_monthly_data_accumulated_to_year_averaged_over_n_years(filepath, year_one, 10)


# Average over a 10-year period to average out seasonality
def get_monthly_data_accumulated_to_year_averaged_over_n_years(filepath, year_one, n):
    iter_csv = pd.read_csv(filepath, iterator=True, chunksize=1000, delim_whitespace=True)
    pet_early = pd.concat([chunk[(chunk['Year'] >= year_one) & (chunk['Year'] < year_one + n)] for chunk in iter_csv])
    pet_early['YearlyTotal'] = pet_early.iloc[:, 3:15].sum(axis=1)
    accumulated = pet_early.groupby(['Lat', 'Lon'], as_index=False).mean()
    return accumulated


def plot_c_pools_over_time(files):
    columns_of_interest = ['Total', 'VegC', 'SoilC', 'LitterC', 'HarvSlowC']
    plot_c_pools_over_time2(files, columns_of_interest, 1995, 2200)


def plot_c_pools_over_time2(files, columns_of_interest, first_year, last_year):
    fig1, ax1 = plt.subplots(1, 5, figsize=(25, 5))
    for simulation, filepath in files.items():
        iter_csv = pd.read_csv(filepath, iterator=True, chunksize=1000, usecols=lambda x: x in columns_of_interest + ['Lat', 'Lon', 'Year'], delim_whitespace=True)
        output = pd.concat([chunk[(chunk['Year'] >= first_year) & (chunk['Year'] <= last_year)] for chunk in iter_csv])

        for idx, column in enumerate(columns_of_interest):
            get_time_series_of_total_cpool(output, column).plot(y=column, ax=ax1[idx], legend=True, label=simulation)
            ax1[idx].set_ylabel('Carbon GtC')
            ax1[idx].set_title(column)
    plt.show()


def scale_fpc(row, snow_cover=None):
    if row['total_non_bare'] > 1 - row['bare']:
        row['evergreen'] *= (1-row['bare']) / row['total_non_bare']
        row['deciduous'] *= (1-row['bare']) / row['total_non_bare']
        row['grass'] *= (1-row['bare']) / row['total_non_bare']
    if row['total_non_bare'] < 1-row['bare']:
        row['unknown'] = 1-row['bare']-row['total_non_bare']

    sum_of_fpc = row['evergreen'] + row['deciduous'] + row['grass'] + row['bare'] + row['unknown']
    assert np.abs(sum_of_fpc - 1.0) < 0.0001, 'Sum of FPC values should be close to 1 but was ' + str(sum_of_fpc)

    if snow_cover is not None:
        row['snowcover'] = snow_cover[row['Lon'], row['Lat'], row['Year']]['Snowcover']
    else:
        row['snowcover'] = 0.0

    return row

