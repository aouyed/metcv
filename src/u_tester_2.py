#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:17:00 2020

@author: aouyed
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df_dof_ms=pd.read_pickle('../data/processed/dof_ms/df_mean_lat_u_scaled_approx.pkl')
df_dof=pd.read_pickle('../data/processed/dof/df_mean_lat_u_scaled_approx.pkl')
df_cc_t10=pd.read_pickle('../data/processed/cc_t10/df_mean_lat_u_scaled_approx.pkl')
df_cc_t5=pd.read_pickle('../data/processed/cc_t5/df_mean_lat_u_scaled_approx.pkl')
df_cc_t30=pd.read_pickle('../data/processed/cc_t30/df_mean_lat_u_scaled_approx.pkl')

df_u=pd.read_pickle('../data/processed/dof_ms/df_mean_lat_u.pkl')

df_jpl=pd.read_pickle('../data/processed/jpl/df_mean_lat_u_scaled_approx.pkl')

#df_dof3u_ss=pd.read_pickle('../data/processed/dof_3/df_mean_lat_u_scaled_approx.pkl')

#df_dofu_ss=pd.read_pickle('../data/processed/dof/df_mean_lat_u_scaled_approx.pkl')
#df_jpl=pd.read_pickle('../data/processed/jpl/df_mean_lat_u_scaled_approx.pkl')



fig, ax = plt.subplots()
ax=sns.lineplot(x=df_dof_ms['lat'], y=df_dof_ms['u_scaled_approx'], label='ms',linewidth=1)
ax=sns.lineplot(x=df_jpl['lat'], y=df_jpl['u_scaled_approx'], label='jpl',linewidth=1)
ax=sns.lineplot(x=df_dof['lat'], y=df_dof['u_scaled_approx'], label='dof',linewidth=1)
#ax=sns.lineplot(x=df_cc_t10['lat'], y=df_cc_t10['u_scaled_approx'], label='cct10',linewidth=1)
#ax=sns.lineplot(x=df_cc_t5['lat'], y=df_cc_t5['u_scaled_approx'], label='cct5',linewidth=1)
ax=sns.lineplot(x=df_cc_t30['lat'], y=df_cc_t30['u_scaled_approx'], label='cct30',linewidth=1)

ax=sns.lineplot(x=df_u['lat'], y=df_u['u'], label='truth',linewidth=1)
