#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:17:00 2020

@author: aouyed
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df_dof_ms=pd.read_pickle('../data/processed/dof_ms/df_mean_lat_speed_approx.pkl')
#df_dof=pd.read_pickle('../data/processed/dof/df_mean_lat_u_scaled_approx.pkl')

df_s=pd.read_pickle('../data/processed/dof_ms/df_mean_lat_speed.pkl')

df_jpl=pd.read_pickle('../data/processed/jpl/df_mean_lat_speed_approx.pkl')

#df_dof3u_ss=pd.read_pickle('../data/processed/dof_3/df_mean_lat_u_scaled_approx.pkl')

#df_dofu_ss=pd.read_pickle('../data/processed/dof/df_mean_lat_u_scaled_approx.pkl')
#df_jpl=pd.read_pickle('../data/processed/jpl/df_mean_lat_u_scaled_approx.pkl')



fig, ax = plt.subplots()
ax=sns.lineplot(x=df_s['lat'], y=df_s['speed'], label='truth',linewidth=1)
ax=sns.lineplot(x=df_dof_ms['lat'], y=df_dof_ms['speed_approx'], label='ms',linewidth=1)
ax=sns.lineplot(x=df_jpl['lat'], y=df_jpl['speed_approx'], label='jpl',linewidth=1)
#ax=sns.lineplot(x=df_cc_t10['lat'], y=df_cc_t10['u_scaled_approx'], label='cct10',linewidth=1)
#ax=sns.lineplot(x=df_cc_t5['lat'], y=df_cc_t5['u_scaled_approx'], label='cct5',linewidth=1)
#ax=sns.lineplot(x=df_cc_t30['lat'], y=df_cc_t30['u_scaled_approx'], label='cct30',linewidth=1)

