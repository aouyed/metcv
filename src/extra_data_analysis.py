#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 15:47:22 2020

@author: amirouyed
,"""

import pdb
from viz import amv_analysis as aa
from viz import dataframe_calculators as dfc
import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
from tqdm import tqdm
from joblib import dump, load
from global_land_mask import globe
import reverse_geocoder
import extra_data_plotter as edp
import ml_functions as mlf

R = 6373.0


def distance(s_lat, s_lng, e_lat, e_lng):

    # approximate radius of earth in km

    s_lat = s_lat*np.pi/180.0
    s_lng = np.deg2rad(s_lng)
    e_lat = np.deg2rad(e_lat)
    e_lng = np.deg2rad(e_lng)

    d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat) * \
        np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2

    return 2 * R * np.arcsin(np.sqrt(d))


dict_path = '../data/interim/dictionaries/dataframes.pkl'
dataframes_dict = pickle.load(open(dict_path, 'rb'))


start_date = datetime.datetime(2006, 7, 1, 6, 0, 0, 0)
end_date = datetime.datetime(2006, 7, 1, 7, 0, 0, 0)
df = aa.df_concatenator(dataframes_dict, start_date,
                        end_date, False, True, False)


df = df.dropna()
df['land'] = globe.is_land(df.lat, df.lon)

df = df.reset_index(drop=True)
europe_coord = (54.5260, 15.2551)
us_coord = (37.0902, -95.7129)
aus_coord = (-25.2744, 133.7751)
bra_coord = (-14.2350, -51.9253)

df['europe_lat'] = europe_coord[0]
df['europe_lon'] = europe_coord[1]
df['us_lat'] = us_coord[0]
df['us_lon'] = us_coord[1]

df['aus_lat'] = aus_coord[0]
df['aus_lon'] = aus_coord[1]

df['bra_lat'] = bra_coord[0]
df['bra_lon'] = bra_coord[1]

df['distance'] = np.minimum(distance(df.lat, df.lon, df.europe_lat,

                                     df.europe_lon), distance(df.lat, df.lon, df.us_lat, df.us_lon))

df['distance'] = np.minimum(df.distance, distance(
    df.lat, df.lon, df.aus_lat, df.aus_lon))
df['distance'] = np.minimum(df.distance, distance(
    df.lat, df.lon, df.bra_lat, df.bra_lon))
#df['distance'] = np.maximum(df.distance-1000, 0)
print(df['distance'].max())
df['sample_weight'] = np.exp(-df.distance/(np.pi*R))
exp_distance = np.exp(df.distance/(np.pi*R))
sigma_u = abs(2*exp_distance)
sigma_v = abs(0.2*exp_distance)
df['stdev'] = np.sqrt(sigma_u**2+sigma_v**2)

edp.map_plotter(df, 'stdev', "exponential part of stdev", ' ')
e_u = np.random.normal(scale=sigma_u)
e_v = np.random.normal(scale=sigma_v)
e_u = np.sign(e_u)*np.minimum(2*sigma_u, abs(e_u))
e_v = np.sign(e_v)*np.minimum(2*sigma_v, abs(e_v))

df['error_mag'] = np.sqrt(e_u**2+e_v**2)

edp.map_plotter(df, 'error_mag', "exponential part of stdev", ' ')


print('done plotting')
dft = aa.df_concatenator(dataframes_dict, start_date,
                         end_date, True, True, False)
f = open("errors.txt", "w+")

dft = dft.dropna()
dft['land'] = globe.is_land(dft.lat, dft.lon)
category = []
rmse = []
latlon = []
test_sizes = []
exp_list = []
only_land = False

latdowns = [-30, 30, 60, -60, -90]
latups = [30, 60, 90, -30, -60]
#latdowns = [-90]
#latups = [90]
test_size = 0.9
exp_filters = ['rand']
#exp_filters = ['rand2']
print('process data...')


for exp_filter in exp_filters:
    print('fitting with filter ' + str(exp_filter))
    regressor, X_test0, y_test0 = mlf.ml_fitter('uv', f, df,
                                                'rf', rmse, test_size, only_land, -90, 90, exp_filter)
    print('predicting..')
    for i, latdown in enumerate(tqdm(latdowns)):
        mlf.latitude_selector(f, df, dft, latdown, latups[i], category,  rmse, latlon,  test_size,
                              test_sizes, only_land, exp_filter, exp_list, regressor, X_test0, y_test0)

d = {'latlon': latlon,  'categories': category,
     'rmse': rmse, 'exp_filter': exp_list}

df_results = pd.DataFrame(data=d)

df_results.to_pickle("df_results.pkl")

print('done!')
print(df_results)
# print(df_results[df_results.exp_filter])
