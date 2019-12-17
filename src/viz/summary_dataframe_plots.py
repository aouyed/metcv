#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:19:51 2019

@author: amirouyed
"""

import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df_path='../../data/interim/dataframes/2019-12-16/cross_correlation.pkl'
df_path_fine='../../data/interim/dataframes/2019-12-16/cFalse.pkl'
df = pd.read_pickle(df_path)
dff = pd.read_pickle(df_path_fine)

sns.set()
sns.set_context('notebook')
print(df[['max', 'corr_speed','mean_speed_error','initial_count','ratio_count']])
#print(dff[['max', 'corr_speed','mean_speed_error','initial_count','ratio_count']])

fig, ax = plt.subplots()


sns.lineplot(x=df['cutoff'],y=df['corr_speed'], ax=ax,label='cross_correlation')
sns.lineplot(x=dff['cutoff'],y=dff['corr_speed'],ax=ax,label='DOF')

ax.set(xlim=(2.5,10))
#ax.set(ylim=(0.7,1))
ax.set_xlabel('maxiumum AMV vector difference [m/s]')
ax.set_ylabel('correlation coefficient')

################################
fig, ax = plt.subplots()


sns.lineplot(x=df['cutoff'],y=df['corr_speed'], ax=ax,label='cross_correlation')
sns.lineplot(x=dff['cutoff'],y=dff['corr_speed'],ax=ax,label=' DOF')
#ns.lineplot(x=df['cutoff'],y=0.487145,ax=ax, label='Δθ = 0.5 deg, unfiltered')
#sns.lineplot(x=dff['cutoff'],y=0.174652,ax=ax, label='Δθ = 0.0625 deg, unfiltered')


ax.set(xlim=(2.5,10))
#ax.set(ylim=(-0.3,1))

ax.set_xlabel('maxiumum vector difference [m/s]')
ax.set_ylabel('correlation coefficient')
################################

fig, ax = plt.subplots()

scale=100
sns.lineplot(x=df['ratio_count']*scale,y=df['corr_speed'], ax=ax, label='cross_correlation')
sns.lineplot(x=dff['ratio_count']*scale,y=dff['corr_speed'], ax=ax,label='DOF')
ax.set_xlabel('vector count normalized by 10x10 pixel box')
ax.set_ylabel('correlation coefficient')

fig, ax = plt.subplots()


sns.lineplot(x=df['cutoff'],y=df['ratio_count']*scale, ax=ax,label='cross correlation')
sns.lineplot(x=dff['cutoff'],y=dff['ratio_count']*scale,ax=ax,label='DOF')

ax.set(xlim=(2.5,10))

ax.set_xlabel('maxiumum vector difference [m/s]')
ax.set_ylabel('normalized vector count')


################################

fig, ax = plt.subplots()

scale=100
sns.lineplot(x=df['ratio_count']*scale,y=df['mean_speed_error'], ax=ax, label='cross correlation')
sns.lineplot(x=dff['ratio_count']*scale,y=dff['mean_speed_error'], ax=ax,label='DOF')
ax.set_xlabel('vector count normalized by 10x10 pixel box')
ax.set_ylabel('mean vector difference [m/s]')