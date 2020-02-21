#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 11:44:19 2020

@author: amirouyed
"""

#y_pred = regressor.predict(X_test)
#print(y_pred.shape)
from joblib import dump, load

#dump(regressor, 'rf.joblib') 
regressor=load('rf.joblib') 


y_pred = regressor.predict(X_test)

#dft = aa.df_concatenator(dataframes_dict, start_date, end_date, True, True,False)
dft=dft.dropna()

error_u=y_test['u'] -y_pred[:,0]
error_v=y_test['v'] -y_pred[:,1]

speed_error=np.sqrt(error_u**2-error_v**2)
print(speed_error.mean())

error_ut=y_test['u'] -X_test['u_scaled_approx']
error_vt=y_test['v'] -X_test['v_scaled_approx']

speed_errort=np.sqrt(error_ut**2-error_vt**2)
print(speed_errort.mean())


error_uj=df['u'] -df['u_scaled_approx']
error_vj=df['v'] -df['v_scaled_approx']
speed_errorj=np.sqrt(error_uj**2-error_vj**2)
print(speed_errorj.mean())

error_ujt=dft['u'] -dft['u_scaled_approx']
error_vjt=dft['v'] -dft['v_scaled_approx']
speed_errorjt=np.sqrt(error_ujt**2-error_vjt**2)
print(speed_errorjt.mean())
