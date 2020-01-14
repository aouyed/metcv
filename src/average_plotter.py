#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 09:52:58 2020

@author: aouyed
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df_jpl=pd.read_pickle('../data/processed/jpl/df_mean_lat_speed_error.pkl')
df_jpl_s=pd.read_pickle('../data/processed/jpl/df_mean_speed_speed_error.pkl')
df_jpl_u=pd.read_pickle('../data/processed/jpl/df_mean_u_u_scaled_approx.pkl')
df_jpl_v=pd.read_pickle('../data/processed/jpl/df_mean_v_v_scaled_approx.pkl')
###

df_dof=pd.read_pickle('../data/processed/dof/df_mean_lat_speed_error.pkl')
df_dof_s=pd.read_pickle('../data/processed/dof/df_mean_speed_speed_error.pkl')
df_dof_u=pd.read_pickle('../data/processed/dof/df_mean_u_u_scaled_approx.pkl')
df_dof_v=pd.read_pickle('../data/processed/dof/df_mean_v_v_scaled_approx.pkl')


df_cc=pd.read_pickle('../data/processed/cc/df_mean_lat_speed_error.pkl')
df_cc_s=pd.read_pickle('../data/processed/cc/df_mean_speed_speed_error.pkl')
df_cc_u=pd.read_pickle('../data/processed/cc/df_mean_u_u_scaled_approx.pkl')
df_cc_v=pd.read_pickle('../data/processed/cc/df_mean_v_v_scaled_approx.pkl')



fig, ax = plt.subplots()
ax=sns.lineplot(x=df_jpl['lat'], y=df_jpl['speed_error'], label='jpl')
#ax=sns.lineplot(x=df_dof['lat'], y=df_dof['speed_error'], label='dof')
ax=sns.lineplot(x=df_cc['lat'], y=df_dof['speed_error'], label='cc')


l=np.arange(-30,30,1)
fig, ax = plt.subplots()
ax=sns.lineplot(x=df_jpl_s['speed'], y=df_jpl_s['speed_error'], label='jpl')
ax=sns.lineplot(x=df_dof_s['speed'], y=df_dof_s['speed_error'], label='dof')
ax=sns.lineplot(x=df_cc_s['speed'], y=df_cc_s['speed_error'], label='cc')


fig, ax = plt.subplots()
ax=sns.lineplot(x=df_jpl_u['u'], y=df_jpl_u['u_scaled_approx'], label='jpl')
ax=sns.lineplot(x=df_dof_u['u'], y=df_dof_u['u_scaled_approx'], label='dof')
ax=sns.lineplot(x=df_cc_u['u'], y=df_cc_u['u_scaled_approx'], label='cc')
ax=sns.lineplot(x=df_cc_u['u'], y=(-df_cc_u['u_scaled_approx']+df_dof_u['u_scaled_approx']), label='h')
ax=sns.lineplot(x=l, y=l, label='y=x')


fig, ax = plt.subplots()
ax=sns.lineplot(x=df_jpl_v['v'], y=df_jpl_v['v_scaled_approx'], label='jpl')
ax=sns.lineplot(x=df_dof_v['v'], y=df_dof_v['v_scaled_approx'], label='dof')
ax=sns.lineplot(x=df_cc_v['v'], y=df_cc_v['v_scaled_approx'], label='cc')
ax=sns.lineplot(x=df_cc_v['v'], y=(-df_cc_v['v_scaled_approx']+df_dof_v['v_scaled_approx']), label='h')
ax=sns.lineplot(x=l, y=l, label='y=x')
