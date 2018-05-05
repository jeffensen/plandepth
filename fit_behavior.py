#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

import torch

import scipy as scp

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = 'talk', style = 'white', color_codes = True)

import agents

def map_noise_to_values(strings, blocks):
    def value(s):
        if s[0] == 'high':
            return .5
        else:
            return .9
        
    a = np.zeros(blocks)*np.nan
    for i,s in enumerate(strings):
        a[i] = value(s)
    return a

path = '/home/markovic/Dropbox/Experiments/Data/Plandepth/'
filenames = ['part_1_23-Mar-2018.mat',
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

n_subjects = len(filenames) # number of runs/subjects
blocks = 100 # number of mini blocks in each run
max_trials = 3 # maximal number of trials within a mini block
max_depth = 3 # maximal planning depth

responses = np.zeros((n_subjects, blocks, max_trials))
states = np.zeros((n_subjects, blocks, max_trials+1))
scores = np.zeros((n_subjects, blocks))
conditions = np.zeros((n_subjects, blocks))
confs = np.zeros((n_subjects, blocks, 6))
notrials = np.zeros((n_subjects, blocks))
for i,f in enumerate(filenames):
    parts = f.split('_')
    tmp = scp.io.loadmat(path+f)
    responses[i] = tmp['data']['Responses'][0,0]['Keys'][0,0]-1
    states[i] = tmp['data']['States'][0,0] - 1
    confs[i] = tmp['data']['PlanetConf'][0,0] - 1
    points = tmp['data']['Points'][0,0]
    scores[i] = points[:,-1]
    snans = np.isnan(points[:,-1])
    scores[i, snans] = points[snans, -2]
    strings = tmp['data']['Conditions'][0,0]['noise'][0,0][0]
    conditions[i] = map_noise_to_values(strings, blocks)
    notrials[i] = tmp['data']['Conditions'][0,0]['notrials'][0,0][:,0]

scores = np.nan_to_num(scores)

states = states.reshape(-1,max_trials+1)[:,:-1]
nans = np.any(np.isnan(states), axis = -1)
states = torch.LongTensor(states[~nans]).view(-1, max_trials)    

confs = confs.reshape(-1, 6)
confs = torch.LongTensor(confs[~nans]).view(-1, 6)

responses = responses.reshape(-1, max_trials)
responses = torch.from_numpy(responses[~nans]).view(-1, max_trials)

conditions = conditions.reshape(-1)
conditions = torch.from_numpy(conditions[~nans])

runs = len(states)
agent = agents.Informed(confs,
                        transition_probability=conditions,
                        responses = responses, 
                        states = states, 
                        runs = runs, 
                        trials = max_trials, 
                        planning_depth = max_depth)

responses = responses.numpy()
rnotnans = ~np.isnan(responses)
responses = np.nan_to_num(responses).astype(int)
model_responses = np.tile(np.zeros_like(responses)[None,:,:], [max_depth,1,1]).astype(int)
hamming_dist = np.zeros((max_depth, len(responses)))
for d in range(3):
    model_responses[d, rnotnans] = agent.model(d+1).numpy().astype(int)
    hamming_dist[d] = np.count_nonzero(model_responses[d] != responses, axis = -1)

#model = np.zeros((n_subjects, blocks), dtype = int) - 1
#model = model.reshape(-1)    
#model[~nans] = np.argmin(hamming_dist, axis = 0)
#model = model.reshape(n_subjects, blocks)+1

post_prob = np.exp(-100*(hamming_dist - hamming_dist.max(axis = 0)))
post_prob /= post_prob.sum(axis = 0)
model_prob = np.zeros((n_subjects, blocks, 3))
model_prob = model_prob.reshape(-1, 3)
model_prob[~nans, :] = post_prob.T
model_prob = model_prob.reshape(n_subjects, blocks, 3) 
#results = agent.fit(n_iterations = 50000)

model = np.sum(np.tile(np.arange(1,4), [20, 100,1])*model_prob,axis = -1)


colours = np.array(['#000000', '#0000ff', '#ff0000', '#00ff00'])
time = np.arange(1,blocks+1)

prob_max_depth = np.zeros((n_subjects, 4))
fig, ax = plt.subplots(1,2, figsize = (10,4))
for n in range(n_subjects):
    if n <11:
        ax[0].scatter(time, scores[n], c = model[n]/3, s = 20, cmap = 'viridis')
        mp = model_prob[n].reshape(4,-1,3)[[0,1,2,3],:,[1,1,2,2]]
        mp[:2] += model_prob[n].reshape(4,-1,3)[[0,1],:, [2,2]]
        prob_max_depth[n] = mp.sum(axis = -1)/25
    else:
        ax[1].scatter(time, scores[n], c = model[n]/3, s = 20, cmap = 'viridis')
        mp = model_prob[n].reshape(4,-1,3)[[2,3,1,2],:,[1,1,2,2]]
        mp[:2] += model_prob[n].reshape(4,-1,3)[[2,3],:, [2,2]]
        prob_max_depth[n] = mp.sum(axis = -1)/25

for i in range(2):
    ax[i].vlines([25, 50, 75], 0, 1000, color = 'k', linestyle = '--')

fig.savefig('per_trial_model.pdf')
import pandas as pd

subject = np.tile(np.arange(1, n_subjects+1)[:, None], [1,4])
phase = np.tile(np.arange(1,5)[None,:], [n_subjects, 1])
df = pd.DataFrame({r'$Pr(d=d_{max})$': prob_max_depth.reshape(-1), 
                   'subject': subject.reshape(-1), 
                   'order': ((subject>10).astype(int)+1).reshape(-1),
                   'phase': phase.reshape(-1)})

fig = plt.figure()
sns.boxplot(x = 'phase', y = r'$Pr(d=d_{max})$', data = df, hue = 'order');
fig.savefig('depth_percentage.pdf')

fig, ax = plt.subplots(1,2, figsize = (10,4))
ax[0].plot(model[:10, :50].mean(axis = 0), label = '1')
ax[0].plot(model[10:, 50:].mean(axis = 0), label = '2')
ax[0].vlines(25, 1, 3, linestyle = '--')
ax[0].legend(title = 'order')

ax[1].plot(model[:10, 50:].mean(axis = 0), label = '1')
ax[1].plot(model[10:, :50].mean(axis = 0), label = '2')
ax[1].vlines(25, 1, 3, linestyle = '--')

fig.savefig('mean_model_prob.pdf')

model_ent = -np.sum(model_prob*np.log(model_prob+1e-10), axis = -1)
fig, ax = plt.subplots(1,2, figsize = (10,4))
ax[0].plot(model_ent[:10, :50].mean(axis = 0), label = '1')
ax[0].plot(model_ent[10:, 50:].mean(axis = 0), label = '2')

ax[1].plot(model_ent[:10, 50:].mean(axis = 0), label = '1')
ax[1].plot(model_ent[10:, :50].mean(axis = 0), label = '2')

model_matrix = np.empty_like(model_prob)
for n in range(n_subjects):
    model_matrix[n] = (model_prob[n] > 0.01).astype(int)
    if n > 10:
        model_matrix[n, 50:, -1] = 0
    else:
        model_matrix[n, :50, -1] = 0
np.save('model_matrix.npy', model_matrix)

no_trials = torch.LongTensor(notrials.reshape(-1)[~nans].astype(int)-1)
max_reward = np.zeros((n_subjects, blocks))
max_reward = max_reward.reshape(-1)
max_reward[~nans] = agent.Vs[range(1950), no_trials, states[:,0]].numpy()
max_reward = max_reward.reshape(n_subjects, blocks)

np.save('max_reward.npy', max_reward)

reward_diff = np.zeros((n_subjects, blocks))
reward_diff = reward_diff.reshape(-1)
count = 0
for i in range(len(reward_diff)):
    if not nans[i]:
        notr = no_trials[count]
        state = states[count,0]
        if notr == 2:
            reward_diff[i] = agent.Vs[count,1, state] - agent.Vs[count,0,state]
        else:
            reward_diff[i] = agent.Vs[count,2,state] - agent.Vs[count,1,state]
        count += 1
reward_diff = reward_diff.reshape(n_subjects, -1)

np.save('reward_diff.npy', reward_diff)