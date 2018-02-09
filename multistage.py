#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

import torch
from tasks import MultiStage
from agents import Random, Informed

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = 'talk', style = 'white', color_codes = True)

runs = 10000
trials = 3

na = 2
ns = 6
no = 5

mnom = torch.distributions.Multinomial(probs = torch.ones(ns, no))
outcome_likelihood = mnom.sample((runs,))

transition_matrix = torch.zeros(na, ns, ns)
transition_matrix[0, :-1, 1:] = torch.eye(ns-1)
transition_matrix[0,-1,0] = 1.
transition_matrix[1, :2,-2:] = torch.eye(2)
transition_matrix[1, 2:,:-2] = torch.eye(4)

ms = MultiStage(outcome_likelihood, 
                transition_matrix, 
                runs = runs, 
                trials = trials)

agent1 = Random(runs = runs, trials = trials, na = na)
agent2 = Informed(transition_matrix, 
                  outcome_likelihood,
                  runs = runs,
                  trials = trials, 
                  na = na)

cat = torch.distributions.Categorical(probs = torch.ones(na))
outcomes = torch.zeros(runs, trials+1, 2)
responses = torch.zeros(runs, trials)
outcomes[:,0] = ms.sample_outcomes(0)
res = None
for t in range(1,trials+1):
    agent1.update_beliefs(t, outcomes[:, t-1], responses = res)
    agent2.update_beliefs(t, outcomes[:,t-1], responses = res)
    res = agent1.sample_responses(t) 
    ms.update_states(t, res)
    responses[:,t-1] = res
    outcomes[:, t] = ms.sample_outcomes(t)
    
values = torch.arange(-2, 2+1)
costs = torch.FloatTensor([-.25, -.5])
reward = torch.zeros(runs)
for t in range(trials+1):
    reward += values[outcomes[:,t,0].long()]
    if t>0:
        reward += costs[responses[:,t-1].long()]
    
crew = reward.resize_((100,runs//100)).cumsum(dim = -1)
fig = plt.figure(figsize = (10,6))
plt.plot(crew.numpy().T, color = 'b', alpha = .1);
plt.xlim([0,100])
plt.ylim([-50, 100])