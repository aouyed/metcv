#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 21:17:00 2020

@author: aouyed
"""

import seaborn as sns
import matplotlib.pyplot as plt



df_dofw3=pd.read_pickle('../data/processed/dof_w3/df_mean_lat_u_scaled_approx.pkl')
df_dofw9=pd.read_pickle('../data/processed/dof_w9/df_mean_lat_u_scaled_approx.pkl')
df_dofw27=pd.read_pickle('../data/processed/dof_w27/df_mean_lat_u_scaled_approx.pkl')
df_dofw81=pd.read_pickle('../data/processed/dof_w81/df_mean_lat_u_scaled_approx.pkl')
df_dofw243=pd.read_pickle('../data/processed/dof_w243/df_mean_lat_u_scaled_approx.pkl')
df_dofw720=pd.read_pickle('../data/processed/dof_w720/df_mean_lat_u_scaled_approx.pkl')


#df_dofw81=pd.read_pickle('../data/processed/dof_w81/df_mean_lat_u_scaled_approx.pkl')

df_u=pd.read_pickle('../data/processed/dof_w3/df_mean_lat_u.pkl')
#df_dof3u_ss=pd.read_pickle('../data/processed/dof_3/df_mean_lat_u_scaled_approx.pkl')

#df_dofu_ss=pd.read_pickle('../data/processed/dof/df_mean_lat_u_scaled_approx.pkl')
df_jpl=pd.read_pickle('../data/processed/jpl/df_mean_lat_u_scaled_approx.pkl')



fig, ax = plt.subplots()
ax=sns.lineplot(x=df_jpl['lat'], y=df_jplu_ss['u_scaled_approx'], label='jpl')
ax=sns.lineplot(x=df_dofw3['lat'], y=df_dofw3['u_scaled_approx'], label='w3',linewidth=3)
ax=sns.lineplot(x=df_dofw9['lat'], y=df_dofw9
                ['u_scaled_approx'], label='w9',linewidth=3)
ax=sns.lineplot(x=df_dofw27['lat'], y=df_dofw27
                ['u_scaled_approx'], label='w27',linewidth=3)
ax=sns.lineplot(x=df_dofw81['lat'], y=df_dofw81
                ['u_scaled_approx'], label='w81',linewidth=3)
ax=sns.lineplot(x=df_dofw243['lat'], y=df_dofw243
                ['u_scaled_approx'], label='w243',linewidth=3)
ax=sns.lineplot(x=df_dofw720['lat'], y=df_dofw720
                ['u_scaled_approx'], label='w720',linewidth=3)

ax=sns.lineplot(x=df_u['lat'], y=df_dof2u_su['u'], label='truth',linewidth=5)
