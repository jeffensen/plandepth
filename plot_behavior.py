#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

from os import walk

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(context = 'talk', style = 'white', color_codes = True)

from bayesian_linear_regression import BayesLinRegress

path = '../../../Dropbox/Experiments/Data/Plandepth/'
fnames = ['part_1_23-Mar-2018.mat',
          'part_2_23-Mar-2018.mat',
          'part_3_27-Mar-2018.mat',
          'part_4_27-Mar-2018.mat',
          'part_5_27-Mar-2018.mat',
          'part_6_27-Mar-2018.mat',
          'part_7_27-Mar-2018.mat',
          'part_8_27-Mar-2018.mat',
          'part_9_28-Mar-2018.mat',
          'part_10_28-Mar-2018.mat',
          'part_11_28-Mar-2018.mat',
          'part_12_28-Mar-2018.mat',
          'part_13_28-Mar-2018.mat',
          'part_14_28-Mar-2018.mat',
          'part_15_28-Mar-2018.mat',
          'part_16_28-Mar-2018.mat',
          'part_17_29-Mar-2018.mat',
          'part_18_29-Mar-2018.mat',
          'part_19_29-Mar-2018.mat',
          'part_20_29-Mar-2018.mat']

import scipy as scp

data = pd.DataFrame(columns = ['log_rt', 'Gain', 'Subject', 'Phase', 'Order'])

order = np.tile(range(1,5), (25,1)).flatten(order = 'F')
trials = np.arange(1,101)
color = []
for i,f in enumerate(fnames):
    parts = f.split('_')
    tmp = scp.io.loadmat(path+f)
    points = tmp['data']['Points'][0, 0]
    
    rts = np.nan_to_num(tmp['data']['Responses'][0,0]['RT'][0,0])
    notrials = tmp['data']['Conditions'][0,0]['notrials'][0,0].T
    
    #get points at the last trial of the miniblock
    points = points[range(100), (np.nan_to_num(notrials)-1).astype(int)][0]
    
    
            
    df = pd.DataFrame()
    
    df['Gain']= np.diff(np.hstack([990, points]))
    
    df['Points'] = np.hstack([990, points])[:-1]
    
    df['log_rt']= np.log(rts[:,:3].sum(axis=-1))

    df['Subject'] = i
    
    if int(parts[1]) > 10:
        df['Phase'] = np.hstack([order[50:], order[:50]])
        df['Order'] = 2
        color.append('r')

    else:
        df['Phase'] = order
        df['Order'] = 1
        color.append('b')
    
    if np.isnan(points[-1]):
        color[-1] = 'y'
                
    data = data.append(df)
    
nans = data.Gain.isnull()
data = data[~nans]

sub_idx = data.Subject.values.astype(int)
y = data.log_rt.values
X = np.ones((len(data),1))
X = np.hstack([X, np.log(data.index+1).values[:,None]])

for phase in [1,2,3,4]:
    X = np.hstack([X, (data.Phase == phase).values[:,None]])
    
X = np.hstack([X, data.Points.values[:,None]/990])
    
linreg = BayesLinRegress(X, y, sub_idx)

results = linreg.fit(n_iterations=40000)
mu_w = results['mean_beta']
sigma_w = results['sigma_beta']


#plot convergence
plt.plot(linreg.loss[-10000:]);
plt.axhline(y = np.mean(linreg.loss[-1000:]), xmin=0, xmax = 10000, c ='r');

#plot results
fig, ax = plt.subplots(2, 2, figsize = (10, 5), sharey = True, sharex = True)

for i in range(len(color)):
    ax[0,0].errorbar(mu_w[i,2], mu_w[i,3], 
      xerr=sigma_w[i,2], 
      yerr=sigma_w[i,3], 
      fmt='o',
      elinewidth = 1,
      c = color[i],
      alpha = .8);
    ax[0,1].errorbar(mu_w[i,5], mu_w[i,3], 
      xerr=sigma_w[i,5], 
      yerr=sigma_w[i,3], 
      fmt='o',
      elinewidth = 1,
      c = color[i],
      alpha = .8);
      
    ax[1,0].errorbar(mu_w[i,2], mu_w[i,4], 
      xerr=sigma_w[i,2], 
      yerr=sigma_w[i,4], 
      fmt='o',
      elinewidth = 1,
      c = color[i],
      alpha = .8);
    ax[1,1].errorbar(mu_w[i,5], mu_w[i,4], 
      xerr=sigma_w[i,5], 
      yerr=sigma_w[i,4], 
      fmt='o',
      elinewidth = 1,
      c = color[i],
      alpha = .8);

ax[0,0].set_ylabel(r'$\ln(rt(II))$')
ax[1,0].set_xlabel(r'$\ln(rt(I))$')
ax[1,0].set_ylabel(r'$\ln(rt(III))$')
ax[1,1].set_xlabel(r'$\ln(rt(IV))$')

x = np.arange(-.5, .5, .1)
ax[0,0].plot(x, x, 'k--', lw = 2)
ax[0,1].plot(x, x, 'k--', lw = 2)
ax[1,0].plot(x, x, 'k--', lw = 2)
ax[1,1].plot(x, x, 'k--', lw = 2)
 