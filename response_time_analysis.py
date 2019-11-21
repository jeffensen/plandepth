#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use('seaborn-notebook')
sns.set(style = 'white')

from bayesian_linear_regression import BayesLinRegress

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

from pathlib import Path
home = str(Path.home())

path = home + '/tudcloud/Shared/Experiments/Plandepth/Main-Healthy/main/'
fnames = os.listdir(path)

from scipy import io

data = pd.DataFrame()

T = 100
order = np.tile(range(1,5), (25,1)).flatten(order = 'F')
blocks = np.arange(1, T + 1)
transitions = {0:4, 1:3, 2:4, 3:5, 4:2, 5:2}
states = []
responses = []

n_subs = 0
for i,f in enumerate(fnames):
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
    
        df['gain']= np.diff(np.hstack([990, points]))
    
        df['start_points'] = np.hstack([990, points])[:-1]
        df['end_points'] = points
    
        df['n_trials'] = notrials
    
        df['log_rt_1'] = np.log(rts[:, 0])
        df['log_rt_sum'] = np.log(np.nansum(rts, -1)) 

        df['subject'] = n_subs
        df['block_number'] = blocks
        
        if notrials[0] == 3:
            df['phase'] = np.hstack([order[50:], order[:50]])
            df['order'] = 2
            df['block_index'] = np.hstack([blocks[50:], blocks[:50]])
        
        else:
            df['phase'] = order
            df['order'] = 1
            df['block_index'] = blocks
            
        data = data.append(df, ignore_index=True)
        n_subs += 1


y1 = data['log_rt_1'].values.reshape(n_subs, T).T

X1 = np.expand_dims(data['block_number'].values.astype(int).reshape(n_subs, T).T, -1)
X2 = np.log(X1)

X1 = X1 - X1.mean(0)
X2 = X2 - X2.mean(0)

X3 = np.concatenate([np.ones((T, n_subs, 1)), X2], -1)

phases = data.phase.values.reshape(n_subs, T).T - 1

X1 = np.concatenate([X1, np.eye(4)[phases]], -1)
X2 = np.concatenate([X2, np.eye(4)[phases]], -1)

start_points = data.start_points.values.reshape(n_subs, T).T
start_points -= start_points.mean(0)

X1 = np.concatenate([X1, np.expand_dims(start_points, -1)], -1)
X2 = np.concatenate([X2, np.expand_dims(start_points, -1)], -1)
X3 = np.concatenate([X3, np.expand_dims(start_points, -1)], -1)


m1 = BayesLinRegress(X1, y1)
m2 = BayesLinRegress(X2, y1)
m3 = BayesLinRegress(X3, y1)

samples_1 = []
for i, m in enumerate([m1, m2, m3]):
    samples_1.append(m.fit(num_samples=5000, warmup_steps=5000, summary=False))
    # print posterior predictive log-likelihood
    print('m{} ppll'.format(i+1), m.post_pred_log_likelihood())

# We get approximately the following values for ppll
# ppll(m1) = -3412, ppll(m2) = -3412, ppll(m3) = 3619
# In conclusion removing phase basef dependency of response times reduces model evidence.

    
y2 = data['log_rt_sum'].values.reshape(n_subs, T).T

states = np.array(states)
responses = np.array(responses)
failures = get_failures(states, responses).T
failures -= failures.mean(0)

X1 = np.concatenate([X1, np.expand_dims(failures, -1)], -1)
X2 = np.concatenate([X2, np.expand_dims(failures, -1)], -1)
X3 = np.concatenate([X3, np.expand_dims(failures, -1)], -1)

m4 = BayesLinRegress(X1, y2)
m5 = BayesLinRegress(X2, y2)
m6 = BayesLinRegress(X3, y2)

samples_2 = []
for i, m in enumerate([m4, m5, m6]):
    samples_2.append(m.fit(num_samples=5000, warmup_steps=5000, summary=False))
    # print posterior predictive log-likelihood
    print('m{} ppll'.format(i+4), m.post_pred_log_likelihood())

# For the second response variable we get the following values for ppll
# ppll(m4) = -3167, ppll(m5) = -3153, ppll(m6) = -3357
# We again see the need for separating response times on phases, and furthermore now a stronger 
# evidence that predictor should contain log(block_number) instead of block_number. This results 
# in a power-law reduction of response times with experiment duration.
    
# Potential additional predictors to consider:
#   - number of jumps
#   - number of miniblocks since the usage of the same strategy (action sequence)
#   - max gain
#   - cumulative gain on x previous trials
    
betas1 = samples_1[1]['beta']
betas2 = samples_2[1]['beta']

mu_beta1 = betas1.mean(0)
std_beta1 = betas1.std(0)

mu_beta2 = betas2.mean(0)
std_beta2 = betas2.std(0)

#plot results of fitting log_rt_1 response
fig, ax = plt.subplots(2, 2, figsize = (10, 5), sharey = True)

ax[0,0].errorbar(mu_beta1[:, 1], mu_beta1[:, 2], 
                 xerr = std_beta1[:,1], 
                 yerr= std_beta1[:,2], 
                 fmt='o',
                 elinewidth = 1,
                 c = 'r',
                 alpha = .8);
  
ax[0,1].errorbar(mu_beta1[:,4], mu_beta1[:, 2], 
                 xerr = std_beta1[:, 4], 
                 yerr= std_beta1[:, 2], 
                 fmt='o',
                 elinewidth = 1,
                 c = 'r',
                 alpha = .8);
  
ax[1,0].errorbar(mu_beta1[:, 1], mu_beta1[:, 3],
                 xerr = std_beta1[i, 1],
                 yerr= std_beta1[i, 3],
                 fmt='o',
                 elinewidth = 1,
                 c = 'r',
                 alpha = .8);
  
ax[1,1].errorbar(mu_beta1[:, 4], mu_beta1[:, 3],
                 xerr = std_beta1[:, 4],
                 yerr= std_beta1[:, 3],
                 fmt='o',
                 elinewidth = 1,
                 c = 'r',
                 alpha = .8);
  
mu_group = m2.samples['mt'][:, 1:5].mean(0)

ax[0,0].scatter(mu_group[1], mu_group[2], color = 'k', zorder = 10)
ax[0,1].scatter(mu_group[4], mu_group[2], color = 'k', zorder = 10)
ax[1,0].scatter(mu_group[1], mu_group[3], color = 'k', zorder = 10)
ax[1,1].scatter(mu_group[4], mu_group[3], color = 'k', zorder = 10)

ax[0,0].set_ylabel(r'$\ln(rt[$ two x high $])$')
ax[1,0].set_xlabel(r'$\ln(rt[$ two x low $])$')
ax[1,0].set_ylabel(r'$\ln(rt[$ three x low $])$')
ax[1,1].set_xlabel(r'$\ln(rt[$ three x high $])$')

x1 = np.arange(1.5, 3.5, .1)
x2 = np.arange(0.5, 3.5, .1)
ax[0,0].plot(x1, x1, 'k--', lw = 2)
ax[0,1].plot(x2, x2, 'k--', lw = 2)
ax[1,0].plot(x1, x1, 'k--', lw = 2)
ax[1,1].plot(x2, x2, 'k--', lw = 2)

#plot results of fitting log_rt_sum response
fig, ax = plt.subplots(2, 2, figsize = (10, 5), sharey = True)

ax[0,0].errorbar(mu_beta2[:, 1], mu_beta2[:, 2], 
                 xerr = std_beta2[:,1], 
                 yerr= std_beta2[:,2], 
                 fmt='o',
                 elinewidth = 1,
                 c = 'r',
                 alpha = .8);
  
ax[0,1].errorbar(mu_beta2[:,4], mu_beta2[:, 2], 
                 xerr = std_beta2[:, 4], 
                 yerr= std_beta2[:, 2], 
                 fmt='o',
                 elinewidth = 1,
                 c = 'r',
                 alpha = .8);
  
ax[1,0].errorbar(mu_beta2[:, 1], mu_beta2[:, 3],
                 xerr = std_beta2[i, 1],
                 yerr= std_beta2[i, 3],
                 fmt='o',
                 elinewidth = 1,
                 c = 'r',
                 alpha = .8);
  
ax[1,1].errorbar(mu_beta2[:, 4], mu_beta2[:, 3],
                 xerr = std_beta2[:, 4],
                 yerr= std_beta2[:, 3],
                 fmt='o',
                 elinewidth = 1,
                 c = 'r',
                 alpha = .8);
  
mu_group2 = m5.samples['mt'][:, 1:5].mean(0)

ax[0,0].scatter(mu_group2[1], mu_group2[2], color = 'k', zorder = 10)
ax[0,1].scatter(mu_group2[4], mu_group2[2], color = 'k', zorder = 10)
ax[1,0].scatter(mu_group2[1], mu_group2[3], color = 'k', zorder = 10)
ax[1,1].scatter(mu_group2[4], mu_group2[3], color = 'k', zorder = 10)

ax[0,0].set_ylabel(r'$\ln(rt[$ two x high $])$')
ax[1,0].set_xlabel(r'$\ln(rt[$ two x low $])$')
ax[1,0].set_ylabel(r'$\ln(rt[$ three x low $])$')
ax[1,1].set_xlabel(r'$\ln(rt[$ three x high $])$')

x1 = np.arange(1.5, 3.5, .1)
x2 = np.arange(0.5, 3.5, .1)
ax[0,0].plot(x1, x1, 'k--', lw = 2)
ax[0,1].plot(x2, x2, 'k--', lw = 2)
ax[1,0].plot(x1, x1, 'k--', lw = 2)
ax[1,1].plot(x2, x2, 'k--', lw = 2)

#fig.savefig('fig1.png', bbox_tight=True, transparent = True)

fig, axes = plt.subplot()

# test residuals for stationarity and normality

pred1 = np.einsum('ijk,ljk->ilj',betas1, X2[..., :-1])
residuals1 = y1 - pred1.mean(0)

pred2 = np.einsum('ijk,ljk->ilj',betas2, X2)
residuals2 = y2 - pred2.mean(0)

from statsmodels.tsa.stattools import adfuller

for ns in range(n_subs):
    print(adfuller(residuals1[:, ns])[:2])
    print(adfuller(residuals2[:, ns])[:2])

# We do not see any nonstationarity in the residuals
    
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for ns in range(n_subs):
    sns.kdeplot(np.array(residuals1[:, ns]).astype(np.double), color='b', alpha=.6, ax=axes[0])
    sns.kdeplot(np.array(residuals2[:, ns]).astype(np.double), color='b', alpha=.6, ax=axes[1])
    
# The plots show that we are not explaining all structure in the response times as not all distributions 
# are unimodal that is gaussian. 


#residual = observed - np.sum(mu_beta[sub_idx,:]*X, axis = -1)
#residual = residual.reshape(n_subs,-1)

failures = get_failures(states, responses)
fails_per_phase = failures.reshape(n_subs, 4, -1).sum(axis = -1)

n_jumps = np.nansum(responses, -1).reshape(n_subs, 4, -1)
jumps_per_phase = n_jumps.sum(axis = -1)

order = data['order'].values.astype(int).reshape(n_subs, T).T[0]
rdr1 = order == 1
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
for i in range(4):
    axes[0].scatter(fails_per_phase[rdr1, i]/jumps_per_phase[rdr1, i], jumps_per_phase[rdr1, i], label=i+1)
    axes[1].scatter(fails_per_phase[~rdr1, i]/jumps_per_phase[~rdr1, i], jumps_per_phase[~rdr1, i])

axes[0].legend(title='phase')
axes[0].set_title('order 1')
axes[1].set_title('order 2')

# Subjects experienced more failures in the last phase of the experiment (phase 4, order 1 and phase 2 order 2), 
# but they were also using jumps more often.