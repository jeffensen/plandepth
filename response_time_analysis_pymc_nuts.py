#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""
import os
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(context = 'talk', style = 'white', color_codes = True)

def get_failures(states, responses):
    transitions = np.array([4., 3., 4., 5., 1., 1.])
    n_subs, n_blocks = states.shape[:2]
    
    failures = np.zeros((n_subs, n_blocks))    
    for i in range(n_subs):
        for j in range(2):
            nans = np.isnan(states[i,:,j+1])
            sts = states[i,~nans,j].astype(int)
            matching_trans = transitions[sts] == states[i,~nans, j+1]
            resp = responses[i,~nans,j].astype(bool)
            failures[i,~nans] += np.logical_xor(matching_trans, resp)
        nans = np.isnan(states[i,:,-1])
        failures[i] -= nans
    
    failures[failures < 0] = 0
    return failures

path = '../../../Dropbox/Experiments/Data/Plandepth/Main/'
fnames = []
for root, dirs, files in os.walk(path):
    fnames.extend(files)

files = []
for f in fnames:
    if f.split('_')[0] == 'Training':
        pass
    else:
        files.append(f)

fnames = np.sort(files)
    
from scipy import io

data = pd.DataFrame(columns = ['log_rt', 'Gain', 'Subject', 'Phase', 'Order', 'NoTrials', 'BlockIndex'])

order = np.tile(range(1,5), (25,1)).flatten(order = 'F')
blocks = np.arange(1,101)
color = []
states = []
transitions = {0:4, 1:3, 2:4, 3:5, 4:2, 5:2}
responses = []
n_subs = 0
for i,f in enumerate(fnames):
    parts = f.split('_')
    tmp = io.loadmat(path+f)
    points = tmp['data']['Points'][0, 0]
    rts = tmp['data']['Responses'][0,0]['RT'][0,0]
    notrials = tmp['data']['Conditions'][0,0]['notrials'][0,0][:,0]
    
    #get points at the last trial of the miniblock
    points = points[range(100), np.nan_to_num(notrials-1).astype(int)]
    
    if points[-1] > 0:
        states.append(tmp['data']['States'][0,0] - 1)
        responses.append(tmp['data']['Responses'][0,0]['Keys'][0,0]-1)
    
        df = pd.DataFrame()
    
        df['Gain']= np.diff(np.hstack([990, points]))
    
        df['Points'] = np.hstack([990, points])[:-1]
    
        df['NoTrials'] = notrials
    
        df['log_rt']= np.log(np.nanmean(rts, axis = -1))

        df['Subject'] = n_subs
    
        if notrials[0] == 3:
            df['Phase'] = np.hstack([order[50:], order[:50]])
            df['Order'] = 2
            df['BlockIndex'] = np.hstack([blocks[50:], blocks[:50]])
            color.append('r')

        else:
            df['Phase'] = order
            df['Order'] = 1
            df['BlockIndex'] = blocks
            color.append('b')
    
        data = data.append(df)
        n_subs += 1

N = len(data)

states = np.array(states)
responses = np.array(responses)

sub_idx = data.Subject.values.astype(int)
observed = data.log_rt.values

failures = get_failures(states, responses).reshape(-1)
success = np.sum(np.nan_to_num(responses), axis = -1).reshape(-1) - failures

max_reward = np.load('max_reward.npy').reshape(-1)

X = data['BlockIndex'].values[:,None].astype(int)

for phase in [1,2,3,4]:
    X = np.hstack([X, (data.Phase == phase).values[:,None]])
    
X = np.hstack([X, failures[:,None]])

#adding points as regressor reduces model evidence and leads to a zero 
#group mean
#X = np.hstack([X, data.Points.values[:,None]])

Q, R = np.linalg.qr(X)
Q = Q*np.sqrt(N-1)
R = R/np.sqrt(N-1)
R_inv = np.linalg.inv(R).T

d = X.shape[-1]

import theano
Q = theano.shared(Q)
idx = theano.shared(sub_idx)

with pm.Model() as model:
    #model error
    std = pm.InverseGamma('std', alpha=2, beta = 1, shape=(n_subs,))
    
    #noise hyperpriors
    phi = pm.InverseGamma('phi', alpha=2, beta=1, shape = (d,))
    lam = pm.InverseGamma('lam', alpha = 2, beta=0.1, shape=(n_subs,d))
    
    #group level parameter
    prior = pm.Normal('prior',
                       mu=0,
                       sd = 1,
                       shape = (d,))
    
    prior_mu = (phi*prior)[None,:].repeat(n_subs, axis = 0)
    
    #subject level parameter
    noise = pm.Normal('noise', 
                       mu = 0, 
                       sd = 1, 
                       shape = (n_subs,d))
    
    theta = pm.Deterministic('theta', prior_mu + lam*noise)
    mu = pm.math.sum(theta[idx]*Q, axis = -1)
    sd = std[idx]
    
    #Data likelihood
    y = pm.Normal('y', mu = mu, sd=sd, observed=observed)
    
#with model:
#    approx = pm.fit(method = 'advi', n = 50000)
#trace = approx.sample(100000)
    
with model:
    trace = pm.sample(draws = 2000, njobs = 4)


group_beta = (trace['prior']*trace['phi']).dot(R_inv)
mu_group = group_beta.mean(axis = 0)
sigma_group = group_beta.std(axis = 0)
beta = trace['theta'].dot(R_inv)
mu_beta = beta.mean(axis = 0)
sigma_beta = beta.std(axis = 0)


#plot results
fig, ax = plt.subplots(2, 2, figsize = (10, 5), sharey = True)

for i in range(len(color)):
    
    ax[0,0].errorbar(mu_beta[i,1], mu_beta[i,2], 
      xerr = sigma_beta[i,1], 
      yerr= sigma_beta[i,2], 
      fmt='o',
      elinewidth = 1,
      c = 'r',
      alpha = .8);
      
    ax[0,1].errorbar(mu_beta[i,4], mu_beta[i,2], 
      xerr = sigma_beta[i,4], 
      yerr= sigma_beta[i,2], 
      fmt='o',
      elinewidth = 1,
      c = 'r',
      alpha = .8);
      
    ax[1,0].errorbar(mu_beta[i,1], mu_beta[i,3], 
      xerr = sigma_beta[i,1], 
      yerr= sigma_beta[i,3], 
      fmt='o',
      elinewidth = 1,
      c = 'r',
      alpha = .8);
      
    ax[1,1].errorbar(mu_beta[i,4], mu_beta[i,3], 
      xerr = sigma_beta[i,4], 
      yerr= sigma_beta[i,3], 
      fmt='o',
      elinewidth = 1,
      c = 'r',
      alpha = .8);

ax[0,0].scatter(mu_group[1], mu_group[2], color = 'k', zorder = 10)
ax[0,1].scatter(mu_group[4], mu_group[2], color = 'k', zorder = 10)
ax[1,0].scatter(mu_group[1], mu_group[3], color = 'k', zorder = 10)
ax[1,1].scatter(mu_group[4], mu_group[3], color = 'k', zorder = 10)

ax[0,0].set_ylabel(r'$\ln(rt(II))$')
ax[1,0].set_xlabel(r'$\ln(rt(I))$')
ax[1,0].set_ylabel(r'$\ln(rt(III))$')
ax[1,1].set_xlabel(r'$\ln(rt(IV))$')

x1 = np.arange(1., 3., .1)
x2 = np.arange(0.5, 4.5, .1)
ax[0,0].plot(x1, x1, 'k--', lw = 2)
ax[0,1].plot(x2, x2, 'k--', lw = 2)
ax[1,0].plot(x1, x1, 'k--', lw = 2)
ax[1,1].plot(x2, x2, 'k--', lw = 2)
ax[0,0].set_xlim([1.,3.]); ax[1,0].set_xlim([1.,3.]);
ax[0,1].set_xlim([.5, 4.5]); ax[1,1].set_xlim([.5, 4.5]);

fig.savefig('fig1.png', bbox_tight=True, transparent = True)

residual = observed - np.sum(mu_beta[sub_idx,:]*X, axis = -1)
residual = residual.reshape(n_subs,-1)

plt.figure()
plt.plot(blocks, np.median(residual,axis = 0))
plt.xlim([1,100])
plt.xlabel('block index')
plt.ylabel(r'median $\ln(RT)$ residual')
plt.savefig('fig2.pdf', bbox_tight=True, transparent = True)


fig, ax = plt.subplots(8, 5, figsize = (20, 15), sharex = True, sharey=True)
ax = ax.flatten()
max_reward = np.load('max_reward.npy')
block_index = np.arange(1,101)
phases = np.ones((25,4))
for i in range(n_subs):
    T = len(data.loc[data['Subject']==i, 'BlockIndex'])
    ax[i].plot(data.loc[data['Subject']==i, 'BlockIndex'], data.loc[data['Subject']==i, 'log_rt'], 'ob')
    ax[i].plot(block_index, block_index*mu_beta[i,0] + (phases*mu_beta[i,1:5].T).reshape(-1, order = 'F'), 'r')

#failures = get_failures(states, responses)
#fails_per_phase = failures.reshape(n_subs, 4, -1).sum(axis = -1)
#fail0 = np.tile([16, 10], (n_subs,1))
#
#n_jumps = np.sum(np.nan_to_num(responses), axis = -1).reshape(20,4,-1)
#jumps_per_phase = n_jumps.sum(axis = -1)
#
#fail1 = fail0+fails_per_phase[:,:2]
#jpp1 = jumps_per_phase[:,:2]+20
#
#p1 = 1-fail1/jpp1
#p2 = 1-(fail1+fails_per_phase[:,2:])/(jpp1+jumps_per_phase[:,2:])
#
#plt.figure()
#plt.scatter(fail0[:,0]/20, p1[:,0]); 
#plt.scatter(fail0[:,1]/20, p1[:,1]);
#x = np.arange(.5, .9, .05)
#plt.plot(x,x, 'k--')
#plt.xlabel('prior')
#plt.xlabel('first half posterior mean')
#plt.savefig('fig4.pdf', bbox_tight=True, transparent = True)
#
#plt.figure()
#plt.scatter(p1[:,0], p2[:,0]);
#plt.scatter(p1[:,1], p2[:,1]);
#plt.plot(x,x, 'k--')
#plt.xlabel('first half posterior mean')
#plt.ylabel('second half posterior mean')
#plt.savefig('fig5.pdf', bbox_tight=True, transparent = True)
#
#report_low = np.array([10,7,8,8,8,9,8,9,8,9,9,9,9,8,9,9,8,9,9,3])/10
#report_high = np.array([7,4,3,6,6,6,5,4,4,4,3,7,7,3,8,2,5,7,7,7])/10
#
#p2 = 1-(np.array([8,5])[None,:]+fails_per_phase[:,2:])/(10+jumps_per_phase[:,2:])
#
#plt.figure()
#plt.scatter(p2[:,0], report_low);
#plt.scatter(p2[:,1], report_high);
#plt.plot(x,x, 'k--')
#plt.xlabel('final')
#plt.ylabel('report')
#plt.savefig('fig6.pdf', bbox_tight=True, transparent = True)
