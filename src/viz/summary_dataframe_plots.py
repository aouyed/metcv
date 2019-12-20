#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:19:51 2019

@author: amirouyed
"""

import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#df_path='../../data/interim/dataframes/2019-12-17/cc.pkl'
#df_path_fine='../../data/interim/dataframes/2019-12-17/dof.pkl'

df_path='../../data/interim/dataframes/2019-12-18/qvdens.pkl'
df_path_fine='../../data/interim/dataframes/2019-12-18/qv.pkl'
df = pd.read_pickle(df_path)
dff = pd.read_pickle(df_path_fine)

sns.set()
sns.set_context('notebook')
#print(dff[['max', 'corr_speed','mean_speed_error','initial_count','ratio_count']])

fig, ax = plt.subplots()


sns.lineplot(x=df['cutoff'],y=df['corr_speed'], ax=ax,label=label1)
sns.lineplot(x=dff['cutoff'],y=dff['corr_speed'],ax=ax,label=label2)

ax.set(xlim=(2.5,10))
#ax.set(ylim=(0.7,1))
ax.set_xlabel('maxiumum AMV vector difference [m/s]')
ax.set_ylabel('correlation coefficient')

################################
fig, ax = plt.subplots()

label1='water vapor density'
label2='specific humidity'


sns.lineplot(x=df['cutoff'],y=df['corr_speed'], ax=ax,label=label1)
sns.lineplot(x=dff['cutoff'],y=dff['corr_speed'],ax=ax,label=' DOF')
#ns.lineplot(x=df['cutoff'],y=0.487145,ax=ax, label='Δθ = 0.5 deg, unfiltered')
#sns.lineplot(x=dff['cutoff'],y=0.174652,ax=ax, label='Δθ = 0.0625 deg, unfiltered')


ax.set(xlim=(2.5,10))
#ax.set(ylim=(-0.3,1))

ax.set_xlabel('maxiumum vector difference [m/s]')
ax.set_ylabel('correlation coefficient')
################################

fig, ax = plt.subplots()

scale=1
sns.lineplot(x=df['ratio_count']*scale,y=df['corr_speed'], ax=ax, label=label1)
sns.lineplot(x=dff['ratio_count']*scale,y=dff['corr_speed'], ax=ax,label=label2)
ax.set_xlabel('fraction of raw AMVs preserved')
ax.set_ylabel('correlation coefficient')

fig, ax = plt.subplots()


sns.lineplot(x=df['cutoff'],y=df['ratio_count']*scale, ax=ax,label=label1)
sns.lineplot(x=dff['cutoff'],y=dff['ratio_count']*scale,ax=ax,label=label2)

ax.set(xlim=(2.5,10))

ax.set_xlabel('maxiumum vector difference [m/s]')
ax.set_ylabel('fraction of raw AMVs preserved')


################################

fig, ax = plt.subplots()

sns.lineplot(x=df['ratio_count']*scale,y=df['mean_speed_error'], ax=ax, label=label1)
sns.lineplot(x=dff['ratio_count']*scale,y=dff['mean_speed_error'], ax=ax,label=label2)
ax.set_xlabel('fraction of raw AMVs preserved')
ax.set_ylabel('mean vector difference [m/s]')

print(df[])
print(dff)