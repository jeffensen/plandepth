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
N = 4
trials = [N-1, N]
na = 2
ns = 6
no = 5

confs = []
starts = []
for T in trials:
    perms = np.random.permutation(50)
    confs.append(np.load('confsT%d.npy' % T)[:50][perms])
    starts.append(np.load('startsT%d.npy' % T)[:50][perms])

#np.save('confsExp1.npy', np.vstack(confs))
#np.save('startsExp1.npy', np.hstack(starts))
    
outcome_likelihood = torch.from_numpy(np.vstack(confs))
starts = torch.from_numpy(np.hstack(starts))



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
    agent = [Informed(transition_matrix, 
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
plt.xlim([1,100])
plt.ylim([-80, 20])


#fig.savefig('performance.pdf', dpi = 300)