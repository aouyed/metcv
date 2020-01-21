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


df_h=pd.read_pickle('../data/processed/hybrid/df_mean_lat_speed_error.pkl')
df_h_ss=pd.read_pickle('../data/processed/hybrid/df_mean_lat_speed_approx.pkl')
df_h_sss=pd.read_pickle('../data/processed/hybrid/df_mean_lat_speed.pkl')


df_h_s=pd.read_pickle('../data/processed/hybrid/df_mean_speed_speed_error.pkl')
df_h_u=pd.read_pickle('../data/processed/hybrid/df_mean_u_u_scaled_approx.pkl')
df_h_v=pd.read_pickle('../data/processed/hybrid/df_mean_v_v_scaled_approx.pkl')

df_hg05=pd.read_pickle('../data/processed/hybrid_g0.5/df_mean_lat_speed_error.pkl')
df_hg05_ss=pd.read_pickle('../data/processed/hybrid_g0.5/df_mean_lat_speed_approx.pkl')
df_hg05_sss=pd.read_pickle('../data/processed/hybrid_g0.5/df_mean_lat_speed.pkl')

df_hg05_s=pd.read_pickle('../data/processed/hybrid_g0.5/df_mean_speed_speed_error.pkl')
df_hg05_u=pd.read_pickle('../data/processed/hybrid_g0.5/df_mean_u_u_scaled_approx.pkl')
df_hg05_v=pd.read_pickle('../data/processed/hybrid_g0.5/df_mean_v_v_scaled_approx.pkl')


df_hg05h_s=pd.read_pickle('../data/processed/hybrid_g0.5_h/df_mean_speed_speed_error.pkl')
df_hg05h_u=pd.read_pickle('../data/processed/hybrid_g0.5_h/df_mean_u_u_scaled_approx.pkl')
df_hg05h_v=pd.read_pickle('../data/processed/hybrid_g0.5_h/df_mean_v_v_scaled_approx.pkl')


df_dof=pd.read_pickle('../data/processed/dof/df_mean_lat_speed_error.pkl')
df_dof2=pd.read_pickle('../data/processed/dof_2/df_mean_lat_speed_error.pkl')
df_dof3=pd.read_pickle('../data/processed/dof_3/df_mean_lat_speed_error.pkl')



df_jpl=pd.read_pickle('../data/processed/jpl/df_mean_lat_speed_error.pkl')
df_jpl_sss=pd.read_pickle('../data/processed/jpl/df_mean_lat_speed.pkl')

df_jpl_ss=pd.read_pickle('../data/processed/jpl/df_mean_lat_speed_approx.pkl')
df_jpl_s=pd.read_pickle('../data/processed/jpl/df_mean_speed_speed_error.pkl')
df_jpl_u=pd.read_pickle('../data/processed/jpl/df_mean_u_u_scaled_approx.pkl')
df_jpl_v=pd.read_pickle('../data/processed/jpl/df_mean_v_v_scaled_approx.pkl')

df_dof=pd.read_pickle('../data/processed/dof/df_mean_lat_speed_error.pkl')
df_dofx=pd.read_pickle('../data/processed/dof/df_mean_lon_speed_error.pkl')
df_dof2x=pd.read_pickle('../data/processed/dof_2/df_mean_lon_speed_error.pkl')
df_dof3x=pd.read_pickle('../data/processed/dof_3/df_mean_lon_speed_error.pkl')
df_jplx=pd.read_pickle('../data/processed/jpl/df_mean_lon_speed_error.pkl')

df_dof_ss=pd.read_pickle('../data/processed/dof/df_mean_lat_speed_approx.pkl')
df_dof_s=pd.read_pickle('../data/processed/dof/df_mean_speed_speed_error.pkl')
df_dof_st=pd.read_pickle('../data/processed/dof/df_mean_lat_speed.pkl')

df_dof2_ss=pd.read_pickle('../data/processed/dof_2/df_mean_lat_speed_approx.pkl')
df_dof2_s=pd.read_pickle('../data/processed/dof_2/df_mean_speed_speed_error.pkl')
df_dof2_st=pd.read_pickle('../data/processed/dof_2/df_mean_lat_speed.pkl')


df_dof3_ss=pd.read_pickle('../data/processed/dof_3/df_mean_lat_speed_approx.pkl')
df_dof3_s=pd.read_pickle('../data/processed/dof_3/df_mean_speed_speed_error.pkl')
df_dof3_st=pd.read_pickle('../data/processed/dof_3/df_mean_lat_speed.pkl')

df_dof3_ssx=pd.read_pickle('../data/processed/dof_3/df_mean_lon_speed_approx.pkl')
df_dof3_stx=pd.read_pickle('../data/processed/dof_3/df_mean_lon_speed.pkl')
df_dof2u_su=pd.read_pickle('../data/processed/dof_2/df_mean_lat_u.pkl')

df_dof2u_ss=pd.read_pickle('../data/processed/dof_2/df_mean_lat_u_scaled_approx.pkl')
df_dof2u_s=pd.read_pickle('../data/processed/dof_2/df_mean_u_error_u.pkl')


df_dof3u_ss=pd.read_pickle('../data/processed/dof_3/df_mean_lat_u_scaled_approx.pkl')
df_dof3u_s=pd.read_pickle('../data/processed/dof_3/df_mean_u_error_u.pkl')

df_dofu_ss=pd.read_pickle('../data/processed/dof/df_mean_lat_u_scaled_approx.pkl')
df_dofu_s=pd.read_pickle('../data/processed/dof/df_mean_u_error_u.pkl')

df_jplu_ss=pd.read_pickle('../data/processed/jpl/df_mean_lat_u_scaled_approx.pkl')
df_jplu_s=pd.read_pickle('../data/processed/jpl/df_mean_u_error_u.pkl')


df_dof3_ss=pd.read_pickle('../data/processed/dof_3/df_mean_lat_speed_approx.pkl')
df_dof3_s=pd.read_pickle('../data/processed/dof_3/df_mean_speed_speed_error.pkl')
df_dof3_st=pd.read_pickle('../data/processed/dof_3/df_mean_lat_speed.pkl')



df_dof_u=pd.read_pickle('../data/processed/dof/df_mean_u_u_scaled_approx.pkl')
df_dof_v=pd.read_pickle('../data/processed/dof/df_mean_v_v_scaled_approx.pkl')
df_dof2_u=pd.read_pickle('../data/processed/dof_2/df_mean_u_u_scaled_approx.pkl')
df_dof2_v=pd.read_pickle('../data/processed/dof_2/df_mean_v_v_scaled_approx.pkl')
df_dof3_u=pd.read_pickle('../data/processed/dof_3/df_mean_u_u_scaled_approx.pkl')
df_dof3_v=pd.read_pickle('../data/processed/dof_3/df_mean_v_v_scaled_approx.pkl')


fig, ax = plt.subplots()

ax=sns.lineplot(x=df_dof['lat'], y=df_dof['speed_error'], label='dof')
ax=sns.lineplot(x=df_dof2['lat'], y=df_dof2['speed_error'], label='dof_2')
ax=sns.lineplot(x=df_dof3['lat'], y=df_dof3['speed_error'], label='dof_3')
ax=sns.lineplot(x=df_jpl['lat'], y=df_jpl['speed_error'], label='jpl')




fig, ax = plt.subplots()

ax=sns.lineplot(x=df_dofx['lon'], y=df_dofx['speed_error'], label='dof')
ax=sns.lineplot(x=df_dof2x['lon'], y=df_dof2x['speed_error'], label='dof_2')
ax=sns.lineplot(x=df_dof3x['lon'], y=df_dof3x['speed_error'], label='dof_3')
ax=sns.lineplot(x=df_jplx['lon'], y=df_jplx['speed_error'], label='jpl')

####
fig, ax = plt.subplots()

ax=sns.lineplot(x=df_h_ss['lat'], y=df_h_ss['speed_approx_std'], label='h')
ax=sns.lineplot(x=df_jpl_ss['lat'], y=df_jpl_ss['speed_approx_std'], label='jpl')


####
fig, ax = plt.subplots()
ax=sns.lineplot(x=df_dof_ss['lat'], y=df_dof_ss['speed_approx'], label='dof')
ax=sns.lineplot(x=df_dof2_ss['lat'], y=df_dof2_ss['speed_approx'], label='dof2',linewidth=3)
ax=sns.lineplot(x=df_dof3_ss['lat'], y=df_dof3_ss['speed_approx'], label='dof3')

ax=sns.lineplot(x=df_jpl_ss['lat'], y=df_jpl_ss['speed_approx'], label='jpl', linewidth=3)
ax=sns.lineplot(x=df_jpl_sss['lat'], y=df_jpl_sss['speed'], label='truth', linewidth=3)

fig, ax = plt.subplots()
ax=sns.lineplot(x=df_jplu_ss['lat'], y=df_jplu_ss['u_scaled_approx'], label='jpl')
ax=sns.lineplot(x=df_dofu_ss['lat'], y=df_dofu_ss['u_scaled_approx'], label='dof')
ax=sns.lineplot(x=df_dof2u_ss['lat'], y=df_dof2u_ss['u_scaled_approx'], label='dof2',linewidth=3)
ax=sns.lineplot(x=df_dof3u_ss['lat'], y=df_dof3u_ss['u_scaled_approx'], label='dof3')
ax=sns.lineplot(x=df_dof2u_su['lat'], y=df_dof2u_su['u'], label='truth',linewidth=5)

l=np.arange(-30,30,1)
fig, ax = plt.subplots()
ax=sns.lineplot(x=df_h_s['speed'], y=df_h_s['speed_error'], label='ht')
ax=sns.lineplot(x=df_hg05_s['speed'], y=df_hg05_s['speed_error'], label='hg0.5')
ax=sns.lineplot(x=df_hg05h_s['speed'], y=df_hg05h_s['speed_error'], label='hg0.5h')

ax=sns.lineplot(x=df_jpl_s['speed'], y=df_jpl_s['speed_error'], label='jpl')

l=np.arange(-30,30,1)
fig, ax = plt.subplots()
ax=sns.lineplot(x=df_h_s['speed'], y=df_h_s['speed_error_count'], label='ht')
ax=sns.lineplot(x=df_jpl_s['speed'], y=df_jpl_s['speed_error_count'], label='jpl')


fig, ax = plt.subplots()

#ax=sns.lineplot(x=df_hg05_u['u'], y=df_hg05_u['u_scaled_approx'], label='h g0.5')
#ax=sns.lineplot(x=df_hg05h_u['u'], y=df_hg05h_u['u_scaled_approx'], label='h g0.5h')
ax=sns.lineplot(x=df_dof_u['u'], y=df_dof_u['u_scaled_approx'], label='dof')
ax=sns.lineplot(x=df_dof2_u['u'], y=df_dof2_u['u_scaled_approx'], label='dof2')
ax=sns.lineplot(x=df_dof3_u['u'], y=df_dof3_u['u_scaled_approx'], label='dof3')

ax=sns.lineplot(x=df_jpl_u['u'], y=df_jpl_u['u_scaled_approx'], label='jpl')

ax=sns.lineplot(x=l, y=l, label='y=x')

l=np.arange(-30,30,1)

fig, ax = plt.subplots()
ax=sns.lineplot(x=df_h_u['u'], y=df_h_u['u_scaled_approx_std'], label='ht')

ax=sns.lineplot(x=df_jpl_u['u'], y=df_jpl_u['u_scaled_approx_std'], label='jpl')
ax=sns.lineplot(x=df_dof2_u['u'], y=df_dof2_u['u_scaled_approx_std'], label='dof2')
ax=sns.lineplot(x=df_dof3_u['u'], y=df_dof3_u['u_scaled_approx_std'], label='dof3')


fig, ax = plt.subplots()

#ax=sns.lineplot(x=df_h_v['v'], y=df_h_v['v_scaled_approx'], label='h1')
#ax=sns.lineplot(x=df_hg05_v['v'], y=df_hg05_v['v_scaled_approx'], label='h g0.5')
#ax=sns.lineplot(x=df_hg05h_v['v'], y=df_hg05h_v['v_scaled_approx'], label='h g0.5h')
ax=sns.lineplot(x=df_dof_v['v'], y=df_dof_v['v_scaled_approx'], label='dof')
ax=sns.lineplot(x=df_dof2_v['v'], y=df_dof2_v['v_scaled_approx'], label='dof2')
ax=sns.lineplot(x=df_dof3_v['v'], y=df_dof3_v['v_scaled_approx'], label='dof3')

ax=sns.lineplot(x=df_jpl_v['v'], y=df_jpl_v['v_scaled_approx'], label='jpl')

ax=sns.lineplot(x=l, y=l, label='y=x')

print("Done!")