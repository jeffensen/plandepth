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

runs = 100000
trials = 4
na = 2
ns = 6
no = 5

mnom = torch.distributions.Multinomial(probs = torch.ones(ns, no))

outcome_likelihood = mnom.sample((runs,))

p = 1.
transition_matrix = torch.zeros(na, ns, ns)
transition_matrix[0, :-1, 1:] = torch.eye(ns-1)
transition_matrix[0,-1,0] = 1
transition_matrix[1, -2:, 0:3] = (1-p)/2; transition_matrix[1, -2:, 1] = p
transition_matrix[1, 2, 3:6] = (1-p)/2; transition_matrix[1, 2, 4] = p
transition_matrix[1, 0, 3:6] = (1-p)/2; transition_matrix[1, 0, 4] = p
transition_matrix[1, 3, 0] = (1-p)/2; transition_matrix[1, 3, -2] = (1-p)/2; transition_matrix[1, 3, -1] = p
transition_matrix[1, 1, 2:5] = (1-p)/2; transition_matrix[1, 1, 3] = p


ms = [MultiStage(outcome_likelihood, 
                transition_matrix, 
                runs = runs, 
                trials = trials) for x in range(trials)]

for i in range(trials):
    ms[i].states[:,0] = ms[0].states[:,0] #starts

costs = torch.FloatTensor([-.2, -.5])
#agent1 = Random(runs = runs, trials = trials, na = na)
agent = [Informed(transition_matrix, 
                  outcome_likelihood,
                  runs = runs,
                  trials = trials,
                  costs = costs,
                  planning_depth = d) for d in range(1,trials+1)]

outcomes = torch.zeros(trials, runs, trials+1, 2)
responses = torch.zeros(trials, runs, trials)
for d in range(trials):
    outcomes[d,:,0] = ms[d].sample_outcomes(0)
    outcomes[d,:,0,0] = 2
    
for t in range(1,trials+1):
    for d in range(trials):
        agent[d].update_beliefs(t, outcomes[d,:,t-1])
        res = agent[d].sample_responses(t)
        ms[d].update_states(t, res)
        responses[d,:,t-1] = res
        outcomes[d, :, t] = ms[d].sample_outcomes(t)

values = torch.arange(-2, 2+1)
reward = torch.zeros(trials, trials+1, runs)
for t in range(trials+1):
    for d in range(trials):
        if t>0:
            reward[d,t] = values[outcomes[d,:,t,0].long()]
            reward[d,t] += costs[responses[d,:,t-1].long()]

rs = reward.sum(dim=1)    
crew = reward.sum(dim=1).resize_((trials, 100,runs//100)).cumsum(dim = -1)

fig = plt.figure(figsize = (10,6))
style = ['-', '--', '-.', ':' ]
color = ['b', 'r', 'g', 'm']
for d in range(trials):
    plt.plot(crew[d].numpy().T, color = color[d], alpha = .1);
    plt.plot(crew[d].mean(dim=0).numpy(), color = 'k', linestyle = style[d])

plt.xlim([0,100])
plt.ylim([-50, 100])
#fig.savefig('performance.pdf', dpi = 300)

bs = torch.ByteTensor(runs)
bs[:] = True
for i in range(1, len(ms)):
    bs = bs*(rs[i-1] < rs[i])
diff = rs[-1] - rs[0]    
bss = (diff >= 2.5)*bs
inits = ms[0].states[:, 0][bss].numpy()
confs = outcome_likelihood.numpy()[bss.numpy().astype(bool)]
uniqs = np.unique(confs, axis = 0)
starts = {}
for i in range(len(uniqs)):
    for j in range(len(confs)):
        if np.all(uniqs[i] == confs[j]):
            if i in starts.keys():
                starts[i].append(inits[j])
            else:
                starts[i] = [inits[j]]
vec = []
for i in range(len(uniqs)):
    vec.append(np.unique(starts[i])[0])
starts = np.array(vec)
np.save('startsT%d.npy' % trials, starts)
np.save('confsT%d.npy' % trials, uniqs)