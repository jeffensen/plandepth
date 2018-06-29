#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(context = 'talk', style = 'white', color_codes = True)

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

from scipy import io

data = pd.DataFrame(columns = ['log_rt', 'Gain', 'Subject', 'Phase', 'Order', 'NoTrials'])

order = np.tile(range(1,5), (25,1)).flatten(order = 'F')
trials = np.arange(1,101)
color = []
for i,f in enumerate(fnames):
    parts = f.split('_')
    tmp = io.loadmat(path+f)
    points = tmp['data']['Points'][0, 0]
    
    rts = np.nan_to_num(tmp['data']['Responses'][0,0]['RT'][0,0])
    notrials = tmp['data']['Conditions'][0,0]['notrials'][0,0][:,0]
    
    #get points at the last trial of the miniblock
    points = points[range(100), (np.nan_to_num(notrials)-1).astype(int)]
    
    
    df = pd.DataFrame()
    
    df['Gain']= np.diff(np.hstack([990, points]))
    
    df['Points'] = np.hstack([990, points])[:-1]
    
    df['NoTrials'] = notrials
    #normalize points
#    df['Points'] = (df['Points'] - df['Points'].mean())/df['Points'].std()
    
    df['log_rt']= np.log(rts[:,:].sum(axis=-1)) 

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
N = len(data)

n_subs = len(fnames)
sub_idx = data.Subject.values.astype(int)
observed = data.log_rt.values

X = np.zeros((N,2))
loc = data.NoTrials == 2
X[loc,0] = (data.index+1).values[loc]
X[~loc,1]= (data.index+1).values[~loc]

max_reward = np.load('max_reward.npy').reshape(-1)

for phase in [1,2,3,4]:
    X = np.hstack([X, (data.Phase == phase).values[:,None]])

X = np.hstack([X, max_reward[~nans][:,None]])
Q, R = np.linalg.qr(X)
Q = Q*np.sqrt(N-1)
R = R/np.sqrt(N-1)
R_inv = np.linalg.inv(R).T

#reward_diff = np.load('reward_diff.npy')
#model_matrix = np.load('model_matrix.npy')


d = X.shape[-1]

import theano#    prior_mu = gtheta[None,:].repeat(n_subs, axis = 0)
#    theta = pm.Normal('theta', 
#                       mu = prior_mu, 
#                       sd = 0.001, 
#                       shape = (n_subs,d))
Q = theano.shared(Q)
observed = theano.shared(observed)
idx = theano.shared(sub_idx)    

with pm.Model() as model:
    #model error
    std = pm.InverseGamma('std', alpha=2, beta = 1, shape=(n_subs,))
    
    phi = pm.InverseGamma('phi', alpha=2, beta=1, shape = (d,))
#    eta = pm.HalfCauchy('eta', beta=1, shape=(n_subs,d+1))
    lam = pm.InverseGamma('lam', alpha = 2, beta=0.1, shape=(n_subs,d))
   
    gtheta = pm.Normal('gtheta',
                       mu=0,
                       sd = phi,
                       shape = (d,))
    
    prior_mu = gtheta[None,:].repeat(n_subs, axis = 0)
    
    theta = pm.Normal('theta', 
                       mu = prior_mu, 
                       sd = lam, 
                       shape = (n_subs,d))
    
    alpha = pm.Normal('alpha',
                      mu=0,
                      sd = 10,
                      shape=(n_subs,))
    
#    mu = alpha[idx] + Q.dot(gtheta)
    mu = alpha[idx] + pm.math.sum(theta[idx]*Q, axis = -1)
    sd = std[idx]
    
    #Data likelihood
    y = pm.Normal('y', mu = mu, sd=sd, observed=observed)
    
with model:
    approx = pm.fit(method = 'advi', n = 50000)

trace = approx.sample(100000)

#with model:
#    trace = pm.sample(draws = 25000, tune = 10000, njobs = 2, nuts_kwargs=dict(target_accept=.8))    

alpha = np.mean(trace['alpha'], axis = 0)
group_beta = trace['gtheta'].dot(R_inv)
mu_group = group_beta.mean(axis = 0)
sigma_group = group_beta.std(axis = 0)
beta = trace['theta'].dot(R_inv)
mu_beta = beta.mean(axis = 0)
sigma_beta = beta.std(axis = 0)


#plot results
fig, ax = plt.subplots(2, 2, figsize = (10, 5), sharey = True, sharex = True)

for i in range(len(color)):
    
    ax[0,0].errorbar(mu_beta[i,2], mu_beta[i,3], 
      xerr = sigma_beta[i,2], 
      yerr= sigma_beta[i,3], 
      fmt='o',
      elinewidth = 1,
      c = 'r',
      alpha = .8);
    ax[0,1].errorbar(mu_beta[i,5], mu_beta[i,3], 
      xerr = sigma_beta[i,4], 
      yerr= sigma_beta[i,5], 
      fmt='o',
      elinewidth = 1,
      c = 'r',
      alpha = .8);
      
    ax[1,0].errorbar(mu_beta[i,2], mu_beta[i,4], 
      xerr = sigma_beta[i,2], 
      yerr= sigma_beta[i,4], 
      fmt='o',
      elinewidth = 1,
      c = 'r',
      alpha = .8);
    ax[1,1].errorbar(mu_beta[i,5], mu_beta[i,4], 
      xerr = sigma_beta[i,3], 
      yerr= sigma_beta[i,5], 
      fmt='o',
      elinewidth = 1,
      c = 'r',
      alpha = .8);

ax[0,0].scatter(mu_group[2], mu_group[3], color = 'k', zorder = 10)
ax[0,1].scatter(mu_group[5], mu_group[3], color = 'k', zorder = 10)
ax[1,0].scatter(mu_group[2], mu_group[4], color = 'k', zorder = 10)
ax[1,1].scatter(mu_group[5], mu_group[4], color = 'k', zorder = 10)

ax[0,0].set_ylabel(r'$\ln(rt(II))$')
ax[1,0].set_xlabel(r'$\ln(rt(I))$')
ax[1,0].set_ylabel(r'$\ln(rt(III))$')
ax[1,1].set_xlabel(r'$\ln(rt(IV))$')

x = np.arange(0, 2, .1)
ax[0,0].plot(x, x, 'k--', lw = 2)
ax[0,1].plot(x, x, 'k--', lw = 2)
ax[1,0].plot(x, x, 'k--', lw = 2)
ax[1,1].plot(x, x, 'k--', lw = 2)
 