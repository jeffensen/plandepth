#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

import torch
from tasks import MultiStage
from agents import Random, Informed

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = 'talk', style = 'white', color_codes = True)

def make_transition_matrix(transition_probability):
    p = transition_probability
    transition_matrix = torch.zeros(na, ns, ns)
    transition_matrix[0, :-1, 1:] = torch.eye(ns-1)
    transition_matrix[0,-1,0] = 1
    transition_matrix[1, -2:, 0:3] = (1-p)/2; transition_matrix[1, -2:, 1] = p
    transition_matrix[1, 2, 3:6] = (1-p)/2; transition_matrix[1, 2, 4] = p
    transition_matrix[1, 0, 3:6] = (1-p)/2; transition_matrix[1, 0, 4] = p
    transition_matrix[1, 3, 0] = (1-p)/2; transition_matrix[1, 3, -2] = (1-p)/2; 
    transition_matrix[1, 3, -1] = p
    transition_matrix[1, 1, 2:5] = (1-p)/2; transition_matrix[1, 1, 3] = p
    
    return transition_matrix

runs = 100
N = 3
trials = [N, N-1]
na = 2
ns = 6
no = 5

#confs = []
#pconfs = []
#starts = []
#pstarts = []
#for T in trials:
#    perms = np.random.permutation(60)
#    arrays = np.load('confsT%d.npy' % T)
#    confs.append(arrays[perms][:50])
#    pconfs.append(arrays[perms][50:])
#    
#    arrays = np.load('startsT%d.npy' % T)
#    starts.append(arrays[perms][:50])
#    pstarts.append(arrays[perms][50:])

#np.save('confsExp1.npy', np.vstack(confs))
#np.save('startsExp1.npy', np.hstack(starts))
#
#np.save('confsPractise1.npy', np.vstack(pconfs))
#np.save('startsPractise1.npy', np.hstack(pstarts))

vect = np.load('confsExp1.npy')
#outcome_likelihood = torch.from_numpy(vect)
outcome_likelihood = torch.from_numpy(np.vstack([vect[50:], vect[:50]]))
vect = np.load('startsExp1.npy')
#starts = torch.from_numpy(vect)
starts = torch.from_numpy(np.hstack([vect[50:], vect[:50]]))
    
#outcome_likelihood = torch.from_numpy(np.vstack(confs))
#starts = torch.from_numpy(np.hstack(starts))

noise = np.tile(np.array([.9, .5, .9, .5]), (25,1)).T.flatten()

costs = torch.FloatTensor([-.2, -.5]) #action costs
values = torch.arange(-2, 2+1) #outcome values

n_subs = 100 #number of subjects
rewards = torch.zeros(N, n_subs, runs)
for i in range(runs):
    if i < 50:
        T = trials[0]
    else:
        T = trials[1]
    p = noise[i]
    transition_matrix = make_transition_matrix(p)
    
    out_like = torch.from_numpy(np.tile(outcome_likelihood[i], (n_subs,1,1)))
    
    ms = [MultiStage(out_like, 
                    transition_matrix, 
                    runs = n_subs, 
                    trials = T) for n in range(N)]

    for n in range(N):
        ms[n].states[:,0] = starts[i]

#agent1 = Random(runs = runs, trials = trials, na = na)
    agent = [Informed(make_transition_matrix(.9), 
                      out_like,
                      runs = n_subs,
                      trials = T,
                      costs = costs,
                      planning_depth = d) for d in range(1,N+1)]

    outcomes = torch.zeros(N, n_subs, T+1, 2)
    responses = torch.zeros(N, n_subs, T)
    for n in range(N):
        outcomes[n,:,0] = ms[n].sample_outcomes(0)
        outcomes[n,:,0,0] = 2
    
    reward = torch.zeros(N, T, n_subs)
    for t in range(1,T+1):
        for n in range(N):
            agent[n].update_beliefs(t, outcomes[n,:,t-1])
            res = agent[n].sample_responses(t)
            ms[n].update_states(t, res)
            responses[n,:,t-1] = res
            outcomes[n, :, t] = ms[n].sample_outcomes(t)
            reward[n,t-1] = values[outcomes[n,:,t,0].long()]
            reward[n,t-1] += costs[responses[n,:,t-1].long()]

    rewards[:,:,i] = reward.sum(dim=1)    

crew = rewards.cumsum(dim = -1)

fig = plt.figure(figsize = (10,6))
style = ['-', '--', '-.', ':' ]
color = ['b', 'r', 'g', 'm']
for n in range(N):
    plt.plot(np.arange(1, runs+1), crew[n].numpy().T, color = color[n], alpha = .1);
    plt.plot(np.arange(1, runs+1), crew[n].mean(dim=0).numpy(), color = 'k', linestyle = style[n])
    

filepath = '/home/markovic/Dropbox/Experiments/Data/Plandepth/'
#filenames = ['part_1_23-Mar-2018.mat',
#             'part_2_23-Mar-2018.mat',
#             'part_3_27-Mar-2018.mat',
#             'part_4_27-Mar-2018.mat',
#             'part_5_27-Mar-2018.mat',
#             'part_6_27-Mar-2018.mat',
#             'part_7_27-Mar-2018.mat',
#             'part_8_27-Mar-2018.mat',
#             'part_9_28-Mar-2018.mat',
#             'part_10_28-Mar-2018.mat']

filenames = ['part_11_28-Mar-2018.mat',
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

for f in filenames:
    tmp = scp.io.loadmat(filepath+f)
    points = (tmp['data'][0][0][4]-990)/10
    nans = np.isnan(points[:,-1])
    points[:, -1][nans] = points[:,1][nans]
    points = points[:,-1]

    plt.plot(np.arange(1, len(points)+1), points, color = 'm', linewidth = 3)

plt.xlim([1,100])
plt.ylim([-100, 50])
plt.savefig('experiment.pdf')

for i in range(4):
    fig = plt.figure(figsize = (10,6))
    style = ['-', '--', '-.', ':' ]
    color = ['b', 'r', 'g', 'm']
    crew = rewards[:,:, i*25:(i+1)*25].cumsum(dim = -1)
    for n in range(N):
        plt.plot(np.arange(i*25+1, (i+1)*25+1), 
                 crew[n].numpy().T, color = color[n], alpha = .1);
        plt.plot(np.arange(i*25+1, (i+1)*25+1), 
                 crew[n].mean(dim=0).numpy(), color = 'k', linestyle = style[n])
    for f in filenames:
        tmp = scp.io.loadmat(filepath+f)
        points = (tmp['data'][0][0][4]-990)/10
        nans = np.isnan(points[:,-1])
        points[:, -1][nans] = points[:,1][nans]
        points = np.diff(np.hstack([0, points[:,-1]]))[i*25:(i+1)*25].cumsum()

        plt.plot(np.arange(i*25+1, (i+1)*25+1), points, color = 'm', linewidth = 3)
    
    plt.xlim([i*25+1, (i+1)*25])
    plt.ylim([-40, 20])
    plt.savefig('phase%d.pdf' % i)
        
 