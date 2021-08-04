import os
import requests
import numpy as np
import pandas as pd
import netCDF4 as nc
import psycopg2 as psql
import configparser

# Needs to be set/updated manually before it can be used. It is not required for the public released version
INPUT_DATA_DIR = 'C:/dengueML/data/'

def hello_world():
    print('Hello, World, tanvir!')

def get_dbserver_creds():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return [config['psql']['host'], config['psql']['database'], config['psql']['user'], config['psql']['password']]

def get_brazil_location_data():
    # Loads location related metadata from file
    filePath = INPUT_DATA_DIR + 'brazil/br-city-codes.csv'
    loc_data = pd.read_csv(filePath)
    loc_data = loc_data.drop(columns=['wdId', 'lexLabel', 'creation', 'extinction', 'postalCode_ranges', 'ddd','abbrev3', 'notes'])
    loc_data.columns = ['name', 'state', 'code']
    return loc_data

def read_from_db(sqlQuery):
    # Performs a postgresql query and returns the results (as tuples)
    records = []
    try:
        creds = get_dbserver_creds()
        con = psql.connect(host=creds[0], database=creds[1], user=creds[2], password=creds[3])
        cur = con.cursor()
        cur.execute(sqlQuery)
        records = cur.fetchall()

    except (Exception, psql.DatabaseError) as error:
        print(error)

    finally:
        if con is not None:
            con.close()

    return records

def get_regional_locations(region_id, loc_data):
    # Returns a list of location_id's for which data are available in a particular region
    records = read_from_db('SELECT DISTINCT location_id FROM dengue_time_series;')
    locs_with_data = []
    for a in np.asarray(records):
        locs_with_data.append(int(a[0]))

    locs_in_region = loc_data[loc_data['state'] == region_id]['code'].values
    locs_in_region_with_data = [v for v in locs_with_data if v in locs_in_region]

    return locs_in_region_with_data

def get_locations_in_ecoregion(eco_name):
    # get a list of all locations inside the ecoregion
    query = 'SELECT loc.location_id, loc.population, ST_Y(loc.location::geometry) AS lat, ST_X(loc.location::geometry) AS lon '
    query = query + 'FROM local_population AS loc JOIN tnc_terr_ecoregions AS regions ON ST_Contains(ST_SetSRID(regions.geom,4326), loc.location::geometry) '
    query = query + f'WHERE regions.eco_name = \'{eco_name}\'';

    records = read_from_db(query)
    locs_in_region = []
    for a in np.asarray(records):
        locs_in_region.append(int(a[0]))

    # filter out based on locations where data are available
    records = read_from_db('SELECT DISTINCT location_id FROM dengue_time_series;')
    locs_with_data = []
    for a in np.asarray(records):
        locs_with_data.append(int(a[0]))

    locs_in_region_with_data = [v for v in locs_with_data if v in locs_in_region]
    return locs_in_region_with_data

def get_data_for_locs(loc_list, dropna = True):
    # Returns a dataset (incidence data) in a pandas dataframe with location ids as column names
    # for a given list of locations
    combined_df = pd.DataFrame({'date': []})
    for loc_id in loc_list:
        records = read_from_db(f'SELECT obs_date, data_val FROM dengue_time_series WHERE location_id=\'{loc_id}\' ORDER BY obs_date')
        loc_df = pd.DataFrame(records, columns=['date',str(loc_id)])
        loc_df[str(loc_id)] = pd.to_numeric(loc_df[str(loc_id)])
        combined_df = pd.merge(combined_df, loc_df, how='outer', on='date')
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    if dropna:
        combined_df = combined_df.dropna()
    return combined_df

def get_loc_coord(loc_id):
#     br_location_coords = pd.read_csv(os.getcwd()+'/data/brazil/brazil_muni_coord.csv')
    br_location_coords = pd.read_csv(INPUT_DATA_DIR+'brazil/brazil_muni_coord.csv')
    df_search = br_location_coords[br_location_coords['CD_GEOCODM'] == int(loc_id)]
    if df_search.shape[0] > 0:
        lat = df_search.iloc[0]['LAT']
        lon = df_search.iloc[0]['LONG']
    else:
        lat = np.nan
        lon = np.nan
    return lat, lon

def get_vect_sim_data(dataUrl, lat, lon, fromDate, toDate):
    x = requests.get(dataUrl, params = {"lat": lat, "lng": lon, "fromDate": fromDate, "toDate": toDate})
    vect_sim_df = pd.DataFrame.from_records(x.json())
    vect_sim_df['sim_date'] = pd.to_datetime(vect_sim_df['sim_date']).dt.date
    vect_sim_df['pop'] = pd.to_numeric(vect_sim_df['pop'])
    vect_sim_df.rename({'sim_date': 'date', 'pop':'vect_pop'}, axis='columns', inplace=True)
    vect_sim_df['date'] = pd.to_datetime(vect_sim_df['date'])
    return vect_sim_df

# modify it to get weather observation data
def get_station_data(lat, lon, search_range, urlStation, urlObs, fromDate, toDate, tol):
    timeDiff = toDate - fromDate
    expectedDays = timeDiff.days

    # Get list of stations from server
    x = requests.get(urlStation, params = {"lat": lat, "lon":lon, "range":search_range})
    station_df = pd.DataFrame.from_records(x.json())
    station_list = station_df.sort_values('dist').reset_index(drop=True)

    # start with the nearest station and check for data in the given time range.
    dataFound = False
    i = 0

    while dataFound == False:
        station_id = station_list.iloc[i].station_id
        print('Station: ' + str(station_list.iloc[i].name) + ', dist: '+ str(station_list.iloc[i].dist))

        # Get data from server
        x = requests.get(urlObs, params = {"station_id": station_id, "fromDate":fromDate.strftime('%Y-%m-%d'), "toDate":toDate.strftime('%Y-%m-%d'), "tmax":"1", "tmin":"1", "tavg":"1", "prec":"1"})

        obs_df = pd.DataFrame.from_records(x.json())
        availableDays = obs_df.shape[0]

        if (1 - availableDays/expectedDays) < tol:
            # acceptable. return the weather data
            dataFound = True
            break
        else:
            # look for next station
            i = i + 1
            if station_list.shape[0] == i:
                break

    if dataFound == False:
        return -1, -1
    obs_df['obs_date'] = pd.to_datetime(obs_df['obs_date']).dt.date
    obs_df = obs_df.drop('station_id', 1)
    obs_df['tavg'] = pd.to_numeric(obs_df['tavg'])
    obs_df['tmax'] = pd.to_numeric(obs_df['tmax'])
    obs_df['tmin'] = pd.to_numeric(obs_df['tmin'])
    obs_df['prec'] = pd.to_numeric(obs_df['prec'])

    return obs_df, station_list['dist'][i]

def find_index(var_array, var_value):
    return min(range(len(var_array)), key=lambda i: abs(var_array[i]-var_value))

def load_nc_dataset(var_name, year):
    # var_name: air, rhum, pres, pr_wtr
    directory = INPUT_DATA_DIR + 'rean/'
    dir_part = ''

    if var_name == 'air':
        dir_part = 'surf_temp_daily_avg/air.sig995.'
    elif var_name == 'rhum':
        dir_part = 'surf_rel_hum_daily_avg/rhum.sig995.'
    elif var_name == 'pres':
        dir_part = 'surf_press_daily_avg/pres.sfc.'
    elif var_name == 'pr_wtr':
        dir_part = 'prec_wat_daily_avg/pr_wtr.eatm.'

    file = directory + dir_part + str(year) + '.nc'
#     print(file)

    return nc.Dataset(file)

def read_rean_data(lat, lon, date_array, var_name):
    # var_name: air, rhum, pres, pr_wtr
#     print(var_name)
    data = []
    date_indices = []

    if lon < 0:
        lon = lon + 360

    year = -1
    for date in date_array:
        if year != date.year:
            year = date.year
            # load file whenver a the year has changed
            dataset = load_nc_dataset(var_name, year)

            data_values = dataset[var_name][:]
            lat_values = dataset['lat'][:]
            lon_values = dataset['lon'][:]
            lat_idx =find_index(lat_values, lat)
            lon_idx =find_index(lon_values, lon)

        day = date.dayofyear
        time_idx = day - 1
        date_indices.append(date)
        data.append(data_values[time_idx, lat_idx, lon_idx])

    return data, date_indices

def get_rean_data(lat, lon, date_array):

    rean_vars = ['air', 'rhum', 'pres', 'pr_wtr']
    col_names = ['rean_avg_temp', 'rean_rhum', 'rean_pres', 'rean_pr_wtr']

    rean_df = pd.DataFrame()
    for v_id, var_name in enumerate(rean_vars):
        col_name = col_names[v_id]
        data, date_indices = read_rean_data(lat, lon, date_array, var_name)
        rean_df[col_name] = data

    rean_df['date'] = date_indices
    return rean_df

# Convert daily data to weekly data (convert to Sunday, accumulate from prev Monday to this Sunday)
def convert_weather_to_weekly(weeklyDates, weather_df):

    tavg = []
    tmax = []
    tmin = []
    prec = []
    rainy_days = []

    t_range_min = []
    t_range_max = []
    t_range_avg = []

    for i in range(0,len(weeklyDates)):
        lastDayOfWeek = weeklyDates[i]
        firstDayOfWeek = lastDayOfWeek-pd.to_timedelta(6, unit='d')
        weekIndices = (weather_df['obs_date'] <= lastDayOfWeek) & (weather_df['obs_date'] >= firstDayOfWeek)

        n_days = sum(weekIndices)

        if(n_days == 0):
            tavg.append(float('nan'))
            tmax.append(float('nan'))
            tmin.append(float('nan'))
            prec.append(float('nan'))
            rainy_days.append(float('nan'))

            t_range_min.append(float('nan'))
            t_range_max.append(float('nan'))
            t_range_avg.append(float('nan'))
        else:
            tavg.append(weather_df['tavg'][weekIndices].mean())
            tmax.append(weather_df['tmax'][weekIndices].max())
            tmin.append(weather_df['tmin'][weekIndices].min())

            weeks_prec = weather_df['prec'][weekIndices]

            rainy_days.append(sum(weeks_prec > 0)*(7/n_days))
            prec.append(weeks_prec.sum()*(7/n_days))

            t_ranges = weather_df['tmax'][weekIndices] - weather_df['tmin'][weekIndices]

            t_range_min.append(t_ranges.min())
            t_range_max.append(t_ranges.max())
            t_range_avg.append(t_ranges.mean())


    W_df = pd.DataFrame()

    W_df['date'] = weeklyDates
    W_df['tavg'] = tavg
    W_df['tmax'] = tmax
    W_df['tmin'] = tmin
    W_df['prec'] = prec
    W_df['rainy_days'] = rainy_days
    W_df['t_range_avg'] = t_range_avg
    W_df['t_range_max'] = t_range_max
    W_df['t_range_min'] = t_range_min


    # interpolate missing data
    W_df['tavg'] = W_df['tavg'].interpolate()
    W_df['tmax'] = W_df['tmax'].interpolate()
    W_df['tmin'] = W_df['tmin'].interpolate()

    W_df['t_range_avg'] = W_df['t_range_avg'].interpolate()
    W_df['t_range_max'] = W_df['t_range_max'].interpolate()
    W_df['t_range_min'] = W_df['t_range_min'].interpolate()

    W_df['prec'] = W_df['prec'].interpolate(method='spline', order = 2)
    W_df['rainy_days'] = W_df['rainy_days'].interpolate(method='spline', order = 2)

    return W_df

def get_population_data(location_lists):
    sqlQuery = 'SELECT location_id, ST_Y(location::geometry) AS lat, ST_X(location::geometry) AS lon, population FROM local_population WHERE location_id IN '
    sqlQuery = sqlQuery + '('

    for location_id in location_lists:
        sqlQuery = sqlQuery + '\'' +  str(location_id) + '\'' + ','

    sqlQuery = sqlQuery[:-1]
    sqlQuery = sqlQuery + ');'

    records = read_from_db(sqlQuery)
    loc_id = []
    pop_data = []
    lats = []
    lons = []
    for a in np.asarray(records):
        loc_id.append(int(a[0]))
        lats.append(float(a[1]))
        lons.append(float(a[2]))
        pop_data.append(int(a[3]))

    pop_df = pd.DataFrame()

    pop_df['loc_id'] = loc_id
    pop_df['lat'] = lats
    pop_df['lon'] = lons
    pop_df['pop'] = pop_data

    pop_df.set_index('loc_id', inplace=True)

    return pop_df

def normalize_case_counts(inc_df, pop_df):
    inc_df = inc_df.copy(deep=True)
    loc_list = inc_df.columns[1:]
    for loc_id in loc_list:
        pop = pop_df.loc[int(loc_id)].values[2]
        inc_df[loc_id] = inc_df[loc_id]*(100000/pop)
    return inc_df

def add_case_counts(pop_df, outbreak_sizes, outbreak_ratios):
    case_tot = []
    case_frac = []
    for index, row in pop_df.iterrows():
        case_tot.append(outbreak_sizes[str(index)])
        case_frac.append(outbreak_ratios[str(index)])
    pop_df['case_tot'] = case_tot
    pop_df['case_frac'] = case_frac
    case_tot = ((case_tot - min(case_tot))/(max(case_tot)-min(case_tot)))*100+5
    case_frac = ((case_frac - min(case_frac))/(max(case_frac)-min(case_frac)))*100+5
    pop_df['case_tot_n'] = case_tot
    pop_df['case_frac_n'] = case_frac

def get_color_gradient(val, min_val, max_val):
    r = 0
    g = 0
    b = 0

    half_point = (max_val + min_val)/2

    if val < half_point:
        g = 255
        r = int((val/half_point)*255)
    else:
        r = 255
        g = 255-int(((val-half_point)/half_point)*255)

    return '{:02x}'.format(r) + '{:02x}'.format(g) + '{:02x}'.format(b)

def get_combined_df(pop_series, inc_df):
    inc_ratio = []
    for row in inc_df.iterrows():
        inc_counts = row[1][1:]

        na_count = sum(inc_counts.isna())

        total_inc = inc_counts.sum()

        locs_with_data = inc_counts[~inc_counts.isna()].index
        locs_with_data = [int(numeric_string) for numeric_string in locs_with_data]
        total_pop = pop_series.loc[locs_with_data].sum()

        inc_ratio.append((total_inc*100000)/total_pop)

    combined_df = pd.DataFrame()
    combined_df['date'] = inc_df['date']
    combined_df['inc_ratio'] = inc_ratio
    return combined_df

def get_locs_below_threshold(min_frac, inc_df):
    inc_nulls = inc_df.isna().sum()
    inc_nulls = inc_nulls[1:] # drop the date entry
    data_points = inc_df.shape[0]
    locs_to_drop = inc_nulls[(inc_nulls/data_points > (1-min_frac)).values].index
    return locs_to_drop

def get_ecoregion_boundaries(ecoregion_name):
    sqlQuery = f'SELECT ST_AsGeoJSON(ST_SetSRID(geom,4326)), ST_GeometryType(geom), ST_NDims(geom), ST_SRID(geom) FROM tnc_terr_ecoregions WHERE eco_name = \'{ecoregion_name}\''

    records = read_from_db(sqlQuery)
    return np.asarray(records)[0][0]
