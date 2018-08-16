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
sns.set(context = 'paper', style = 'white', color_codes = True)

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

def map_to_labels(array):
    string = np.array(['two x low', 'two x high', 'three x low', 'three x high']);
    
    return string[array-1]

#path = '/home/markovic/Dropbox/Experiments/Data/Plandepth/Pilot/'
#filenames = ['part_1_23-Mar-2018.mat',
#             'part_2_23-Mar-2018.mat',
#             'part_3_27-Mar-2018.mat',
#             'part_4_27-Mar-2018.mat',
#             'part_5_27-Mar-2018.mat',
#             'part_6_27-Mar-2018.mat',
#             'part_7_27-Mar-2018.mat',
#             'part_8_27-Mar-2018.mat',
#             'part_9_28-Mar-2018.mat',
#             'part_10_28-Mar-2018.mat',
#             'part_11_28-Mar-2018.mat',
#             'part_12_28-Mar-2018.mat',
#             'part_13_28-Mar-2018.mat',
#             'part_14_28-Mar-2018.mat',
#             'part_15_28-Mar-2018.mat',
#             'part_16_28-Mar-2018.mat',
#             'part_17_29-Mar-2018.mat',
#             'part_18_29-Mar-2018.mat',
#             'part_19_29-Mar-2018.mat',
#             'part_20_29-Mar-2018.mat']
    
path = '/home/markovic/Dropbox/Experiments/Data/Plandepth/Main/'
f1 = ['part_23_20-Jun-2018.mat',
      'part_25_20-Jun-2018.mat',
      'part_27_21-Jun-2018.mat',
      'part_29_21-Jun-2018.mat',
      'part_31_21-Jun-2018.mat',
      'part_33_21-Jun-2018.mat',
      'part_35_22-Jun-2018.mat',
      'part_37_22-Jun-2018.mat',
      'part_39_22-Jun-2018.mat',
      'part_43_22-Jun-2018_17-54.mat',
      'part_45_25-Jun-2018_15-33.mat',
      'part_47_25-Jun-2018_17-21.mat',
      'part_49_26-Jun-2018.mat',
      'part_51_26-Jun-2018_18-31.mat',
      'part_55_03-Jul-2018_09-33.mat',
      'part_57_03-Jul-2018_11-17.mat',
      'part_61_03-Jul-2018_16-16.mat',
      'part_63_03-Jul-2018_18-48.mat',
      'part_65_04-Jul-2018_10-42.mat',
      'part_67_13-Jul-2018_09-48.mat']

f2 = ['part_24_20-Jun-2018.mat',
      'part_28_21-Jun-2018.mat',
      'part_30_21-Jun-2018.mat',
      'part_32_21-Jun-2018.mat',
      'part_34_21-Jun-2018.mat',
      'part_38_22-Jun-2018.mat',
      'part_40_22-Jun-2018_15-20.mat',
      'part_42_22-Jun-2018_16-57.mat',
      'part_44_25-Jun-2018_14-30.mat',
      'part_46_25-Jun-2018_16-34.mat',
      'part_48_25-Jun-2018_18-28.mat',
      'part_50_26-Jun-2018_17-38.mat',
      'part_52_27-Jun-2018_09-53.mat',
      'part_54_27-Jun-2018_12-21.mat',
      'part_56_03-Jul-2018_10-32.mat',
      'part_58_03-Jul-2018_12-34.mat',
      'part_60_03-Jul-2018_14-42.mat',
      'part_62_03-Jul-2018_17-34.mat',
      'part_64_04-Jul-2018_10-10.mat',
      'part_66_04-Jul-2018_11-41.mat']

filenames = f1 + f2
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

final_score = scores[:,-1].reshape(2,-1)
sns.boxplot(data = final_score.T)
from scipy import stats
stats.wilcoxon(np.diff(final_score, axis = 0)[0])
stats.mannwhitneyu(final_score[0], final_score[1])

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

distance = np.ones((n_subjects, blocks, 3))*3
distance = distance.reshape(-1,3)
distance[~nans,:] = hamming_dist.T
distance[nans,0] = 0
distance = distance.reshape(n_subjects, blocks, 3)
distance[:10,:50,-1] = 3
distance[10:,50:,-1] = 3

model_prob = np.exp(-100*(distance - distance.max(axis = -1)[:,:,None]))
model_prob /= model_prob.sum(axis = -1)[:,:,None]

#results = agent.fit(n_iterations = 50000)

model = np.sum(np.tile(np.arange(1,4), [len(filenames), 100,1])*model_prob,axis = -1)

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
        mp = model_prob[n].reshape(4,-1,3)[[2,3,0,1],:,[1,1,2,2]]
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
                   'condition': map_to_labels(phase.reshape(-1))})

fig = plt.figure()
sns.boxplot(x = 'condition', y = r'$Pr(d=d_{max})$', data = df, color = 'b', hue = 'order');
fig.savefig('depth_percentage.png', transparent = True, bbox_inches = 'tight', dpi = 300)

fig, ax = plt.subplots(1,1, figsize = (10,4), sharey = True, sharex=True)
ax.plot(np.arange(1,51), np.median(np.vstack([model[:10, :50], model[10:, 50:]]), axis = 0));
ax.plot(np.arange(50, 101), np.median(np.vstack([model[:10, 49:], model[10:, :51]]), axis = 0), 'b');
ax.vlines([25,50,75], 1, 3, linestyle = '--');
ax.set_ylabel('median planning depth');
ax.set_xlabel('mini-block index');
ax.set_xlim([1,100]);
ax.text(5, 2.5, 'two x low')
ax.text(30, 2.5, 'two x high')
ax.text(55, 1., 'three x low')
ax.text(80, 1., 'three x high')

fig.savefig('median_model_prob.png',transparent = True, bbox_inches = 'tight', dpi = 300)

model_ent = -np.sum(model_prob*np.log(model_prob+1e-10), axis = -1)
fig, ax = plt.subplots(1,2, figsize = (10,4))
ax[0].plot(model_ent[:10, :50].mean(axis = 0), label = '1')
ax[0].plot(model_ent[10:, 50:].mean(axis = 0), label = '2')

ax[1].plot(model_ent[:10, 50:].mean(axis = 0), label = '1')
ax[1].plot(model_ent[10:, :50].mean(axis = 0), label = '2')

#model_matrix = np.empty_like(model_prob)
#for n in range(n_subjects):
#    model_matrix[n] = (model_prob[n] > 0.01).astype(int)
#    if n > 10:
#        model_matrix[n, 50:, -1] = 0
#    else:
#        model_matrix[n, :50, -1] = 0
#np.save('model_matrix.npy', model_matrix)
#
#no_trials = torch.LongTensor(notrials.reshape(-1)[~nans].astype(int))
#max_reward = np.zeros((n_subjects, blocks))
#max_reward = max_reward.reshape(-1)
#max_reward[~nans] = agent.Vs[range(1950), no_trials, states[:,0]].numpy()
#max_reward = max_reward.reshape(n_subjects, blocks)
#
#np.save('max_reward.npy', max_reward)
#
#reward_diff = np.zeros((n_subjects, blocks))
#reward_diff = reward_diff.reshape(-1)
#count = 0
#for i in range(len(reward_diff)):
#    if not nans[i]:
#        notr = no_trials[count]
#        state = states[count,0]
#        if notr == 2:
#            reward_diff[i] = agent.Vs[count,1, state] - agent.Vs[count,0,state]
#        else:
#            reward_diff[i] = agent.Vs[count,2,state] - agent.Vs[count,1,state]
#        count += 1
#reward_diff = reward_diff.reshape(n_subjects, -1)
#
#np.save('reward_diff.npy', reward_diff)

means = np.hstack([np.transpose(np.vstack([model[:10, :50], model[10:, 50:]]).reshape(-1,2,25), (0,2,1)).reshape(-1,2), 
                   np.transpose(np.vstack([model[:10, 50:], model[10:, :50]]).reshape(-1,2,25), (0,2,1)).reshape(-1,2)])
fig, ax = plt.subplots(1,1, figsize = (6,4)) 
sns.violinplot(data = means, ax = ax); 
ax.set_xlabel('conditions'); 
ax.set_ylabel('posterior expected planning depth'); 
ax.set_xticklabels(['two x low', 'two x high', 'three x low', 'three x high']);

fig.savefig('mean_dists.png', dpi = 300,  transparent = True)