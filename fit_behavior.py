#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

import torch

from scipy import io

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = 'paper', style = 'white', color_codes = True)

import agents

def logit(p):
    return torch.log(p/(1-p))

def logistic(x):
    return 1./(1. + torch.exp(-x))

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

from pathlib import Path
home = str(Path.home())    
path = home + '/Dropbox/Experiments/Data/Plandepth/Main/'
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
    tmp = io.loadmat(path+f)
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

final_score = pd.DataFrame(scores[:,-1].reshape(2,-1).T, columns = ['normal', 'reversed'])
sns.violinplot(data = final_score)
plt.title('final score distribution')
plt.savefig('score_dist.pdf', dpi=300, transparent=True)

from scipy import stats
print(stats.mannwhitneyu(final_score.loc[:, 'normal'], final_score.loc[:, 'reversed']))

states = states.reshape(-1,max_trials+1)[:,:-1]
nans = np.any(np.isnan(states), axis = -1)
states = torch.LongTensor(states[~nans]).view(-1, max_trials)

notrials = torch.LongTensor(notrials).view(-1)    

confs = confs.reshape(-1, 6)
confs = torch.LongTensor(confs[~nans]).view(-1, 6)

responses = responses.reshape(-1, max_trials)
responses = torch.tensor(responses[~nans], dtype=torch.uint8).view(-1, max_trials)

conditions = conditions.reshape(-1)
conditions = torch.from_numpy(conditions[~nans])

runs = len(states)
agent = agents.BackInduction(confs,
                             runs = runs, 
                             trials = max_trials, 
                             planning_depth = max_depth)

trans_pars = torch.stack([logit(conditions.float()), 
                          torch.ones(runs)*1e10]).transpose(dim0 = 1, dim1 = 0)
agent.set_parameters(trans_pars)

first_response = responses[:, 0]
init_state = states[:, 0]

model_res = logistic(-10*agent.D[range(runs), notrials-1, init_state]) > .5

diff_res = torch.tensor(np.uint8( 
                        np.sign(agent.D[range(runs), 0, init_state].numpy()) !=\
                        np.sign(agent.D[range(runs), notrials-1, init_state].numpy())
                                )
                       )
          
plan_depth_max = ((model_res == first_response)*diff_res).reshape(n_subjects, blocks)
diff_res = diff_res.reshape(n_subjects, blocks)

colours = np.array(['#000000', '#0000ff', '#ff0000', '#00ff00'])
time = np.arange(1,blocks+1)

n_jumps = responses.sum(dim=-1).reshape(n_subjects, -1)

prob_max_depth = np.zeros((n_subjects, 4))
prob_jump = np.zeros((n_subjects, 4))
optim_jump_prob = np.zeros((n_subjects, 4))

jump_count = torch.zeros(runs)
rem_trials = notrials.clone() - 1
for i in range(3):
    tmp = logistic(-10*agent.D[range(runs), rem_trials, states[:, i]]) > .5
    tmp *= rem_trials >= 0
    rem_trials -= 1
    
    jump_count += tmp.float()

jump_count = jump_count.reshape(n_subjects, blocks)

fig, ax = plt.subplots(1,2, figsize = (10,4))
for n in range(n_subjects):
    if n <n_subjects//2:
        ax[0].scatter(time, scores[n], c = plan_depth_max[n], s = 20, cmap = 'viridis')
        count = diff_res[n].reshape(4,-1).sum(dim=-1).float()
        tmp = plan_depth_max[n].reshape(4,-1)[[0,1,2,3]]
        prob_max_depth[n] = tmp.sum(dim = -1).float()/count
        tmp = n_jumps[n].reshape(4,-1)[[0,1,2,3]]
        prob_jump[n] = tmp.sum(dim = -1).float()/25.
        tmp = jump_count[n].reshape(4, -1)[[0, 1, 2, 3]]
        optim_jump_prob[n] = tmp.sum(dim = -1).float()/25.
    else:
        ax[1].scatter(time, scores[n], c = plan_depth_max[n], s = 20, cmap = 'viridis')
        count = diff_res[n].reshape(4,-1).sum(dim=-1).float()[[2, 3, 0, 1]]
        tmp = plan_depth_max[n].reshape(4,-1)[[2,3,0,1]]
        prob_max_depth[n] = tmp.sum(dim = -1).float()/count
        tmp = n_jumps[n].reshape(4,-1)[[2, 3, 0,1]]
        prob_jump[n] = tmp.sum(dim = -1).float()/25.
        tmp = jump_count[n].reshape(4, -1)[[2, 3, 0, 1]]
        optim_jump_prob[n] = tmp.sum(dim = -1).float()/25.
        
prob_jump /= np.array([2., 2., 3., 3.])
optim_jump_prob /= np.array([2., 2., 3., 3.])

for i in range(2):
    ax[i].vlines([25, 50, 75], 0, 1000, color = 'k', linestyle = '--')

fig.savefig('per_trial_model.pdf')
import pandas as pd

subject = np.tile(np.arange(1, n_subjects+1)[:, None], [1,4])
phase = np.tile(np.arange(1,5)[None,:], [n_subjects, 1])
df1 = pd.DataFrame({r'$Pr(d=d_{max})$': prob_max_depth.reshape(-1), 
                   'subject': subject.reshape(-1), 
                   'order': ((subject>20).astype(int)+1).reshape(-1),
                   'condition': map_to_labels(phase.reshape(-1))})
    
df2 = pd.DataFrame({r'$Pr(a=jump)$': prob_jump.reshape(-1), 
                   'subject': subject.reshape(-1), 
                   'order': ((subject>20).astype(int)+1).reshape(-1),
                   'condition': map_to_labels(phase.reshape(-1))})

fig = plt.figure()
sns.boxplot(x = 'condition', y = r'$Pr(d=d_{max})$', data = df1, color = 'b', hue = 'order');
fig.savefig('depth_percentage.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)

fig = plt.figure()
sns.boxplot(x = 'condition', y = r'$Pr(a=jump)$', data = df2, color = 'b', hue = 'order');
fig.savefig('jump_percentage.png', transparent = True, bbox_inches = 'tight', dpi = 300)


part_group1 = np.all([prob_jump[:, 0] > prob_jump[:, 1], prob_jump[:, 2] > prob_jump[:, 3]], axis=0)
part_group2 = np.all([prob_jump[:, 0] < prob_jump[:, 1], prob_jump[:, 2] < prob_jump[:, 3]], axis=0)

labels = ['phase I', 'phase II', 'phase III', 'phase IV']
fig, ax = plt.subplots(1,3, sharey=True, figsize = (10,4))
ax[0].plot(labels, prob_jump[part_group1].T, 'bo:' )
ax[1].plot(labels, prob_jump[part_group2].T, 'ro:' )
part_group3 = ~np.any([part_group1, part_group2], axis=0)
ax[2].plot(labels, prob_jump[part_group3].T, 'go:')

ax[0].set_ylabel('jump probability')

fig.savefig('jump_prob.pdf', transparent = True, bbox_inches = 'tight', dpi = 300)

plt.figure()
plt.plot(labels, optim_jump_prob.T, 'bo:')



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