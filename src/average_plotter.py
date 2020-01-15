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
df_jpl_ss=pd.read_pickle('../data/processed/jpl/df_mean_lat_speed_approx.pkl')
df_jpl_s=pd.read_pickle('../data/processed/jpl/df_mean_speed_speed_error.pkl')
df_jpl_u=pd.read_pickle('../data/processed/jpl/df_mean_u_u_scaled_approx.pkl')
df_jpl_v=pd.read_pickle('../data/processed/jpl/df_mean_v_v_scaled_approx.pkl')
###

df_dof=pd.read_pickle('../data/processed/dof/df_mean_lat_speed_error.pkl')
df_dof_ss=pd.read_pickle('../data/processed/dof/df_mean_lat_speed_approx.pkl')
df_dof_s=pd.read_pickle('../data/processed/dof/df_mean_speed_speed_error.pkl')
df_dof_u=pd.read_pickle('../data/processed/dof/df_mean_u_u_scaled_approx.pkl')
df_dof_v=pd.read_pickle('../data/processed/dof/df_mean_v_v_scaled_approx.pkl')


df_cc=pd.read_pickle('../data/processed/cc/df_mean_lat_speed_error.pkl')
#df_cc_ss=pd.read_pickle('../data/processed/cc/df_mean_lat_speed_approx.pkl')
df_cc_s=pd.read_pickle('../data/processed/cc/df_mean_speed_speed_error.pkl')
df_cc_u=pd.read_pickle('../data/processed/cc/df_mean_u_u_scaled_approx.pkl')
df_cc_v=pd.read_pickle('../data/processed/cc/df_mean_v_v_scaled_approx.pkl')

df_cct20=pd.read_pickle('../data/processed/cct20/df_mean_lat_speed_error.pkl')
df_cct20_s=pd.read_pickle('../data/processed/cct20/df_mean_speed_speed_error.pkl')
df_cct20_u=pd.read_pickle('../data/processed/cct20/df_mean_u_u_scaled_approx.pkl')
df_cct20_v=pd.read_pickle('../data/processed/cct20/df_mean_v_v_scaled_approx.pkl')

df_h=pd.read_pickle('../data/processed/hybrid/df_mean_lat_speed_error.pkl')
df_h_ss=pd.read_pickle('../data/processed/hybrid/df_mean_lat_speed_approx.pkl')
df_h_sss=pd.read_pickle('../data/processed/hybrid/df_mean_lat_speed.pkl')

df_h_s=pd.read_pickle('../data/processed/hybrid/df_mean_speed_speed_error.pkl')
df_h_u=pd.read_pickle('../data/processed/hybrid/df_mean_u_u_scaled_approx.pkl')
df_h_v=pd.read_pickle('../data/processed/hybrid/df_mean_v_v_scaled_approx.pkl')


fig, ax = plt.subplots()
ax=sns.lineplot(x=df_jpl['lat'], y=df_jpl['speed_error'], label='jpl')
ax=sns.lineplot(x=df_dof['lat'], y=df_dof['speed_error'], label='dof')
ax=sns.lineplot(x=df_cc['lat'], y=df_cc['speed_error'], label='cc')
ax=sns.lineplot(x=df_h['lat'], y=df_h['speed_error'], label='h')

fig, ax = plt.subplots()
ax=sns.lineplot(x=df_h_ss['lat'], y=df_h_ss['speed_approx'], label='ht')
ax=sns.lineplot(x=df_dof_ss['lat'], y=df_dof_ss['speed_approx'], label='dof')
ax=sns.lineplot(x=df_h_sss['lat'], y=df_h_sss['speed'], label='truth')
ax=sns.lineplot(x=df_jpl_ss['lat'], y=df_jpl_ss['speed_approx'], label='jpl')

#ax=sns.lineplot(x=df_dof_ss['lat'], y=df_dof['speed_approx'], label='dof')
#ax=sns.lineplot(x=df_jpl_sss['lat'], y=df_jpl['speed'], label='ht')



l=np.arange(-30,30,1)
fig, ax = plt.subplots()
ax=sns.lineplot(x=df_jpl_s['speed'], y=df_jpl_s['speed_error'], label='jpl')
ax=sns.lineplot(x=df_dof_s['speed'], y=df_dof_s['speed_error'], label='dof')
ax=sns.lineplot(x=df_cc_s['speed'], y=df_cc_s['speed_error'], label='cc')
ax=sns.lineplot(x=df_h_s['speed'], y=df_h_s['speed_error'], label='ht')


l=np.arange(-30,30,1)
fig, ax = plt.subplots()
ax=sns.lineplot(x=df_h_s['speed'], y=df_h_s['speed_error_count'], label='ht')



fig, ax = plt.subplots()
ax=sns.lineplot(x=df_jpl_u['u'], y=df_jpl_u['u_scaled_approx'], label='jpl')
ax=sns.lineplot(x=df_dof_u['u'], y=df_dof_u['u_scaled_approx'], label='dof')
#ax=sns.lineplot(x=df_cc_u['u'], y=df_cc_u['u_scaled_approx'], label='cc')
ax=sns.lineplot(x=df_cc_u['u'], y=(df_cc_u['u_scaled_approx']+df_dof_u['u_scaled_approx']), label='h')
#ax=sns.lineplot(x=df_cct20_u['u'], y=(df_cct20_u['u_scaled_approx']+df_dof_u['u_scaled_approx']), label='h')
ax=sns.lineplot(x=df_h_u['u'], y=df_h_u['u_scaled_approx'], label='ht')

ax=sns.lineplot(x=l, y=l, label='y=x')

l=np.arange(-30,30,1)
fig, ax = plt.subplots()
ax=sns.lineplot(x=df_h_u['u'], y=df_h_u['u_scaled_approx_count'], label='ht')

fig, ax = plt.subplots()
ax=sns.lineplot(x=df_h_u['u'], y=df_h_u['u_scaled_approx_std'], label='ht')


fig, ax = plt.subplots()
ax=sns.lineplot(x=df_jpl_v['v'], y=df_jpl_v['v_scaled_approx'], label='jpl')
ax=sns.lineplot(x=df_dof_v['v'], y=df_dof_v['v_scaled_approx'], label='dof')
#ax=sns.lineplot(x=df_cc_v['v'], y=df_cc_v['v_scaled_approx'], label='cc')
#ax=sns.lineplot(x=df_cc_v['v'], y=(df_cc_v['v_scaled_approx']+df_dof_v['v_scaled_approx']), label='h')
ax=sns.lineplot(x=df_cc_v['v'], y=(df_cc_v['v_scaled_approx']+df_dof_v['v_scaled_approx']), label='h')
ax=sns.lineplot(x=df_h_v['v'], y=df_h_v['v_scaled_approx'], label='h1')

ax=sns.lineplot(x=l, y=l, label='y=x')
