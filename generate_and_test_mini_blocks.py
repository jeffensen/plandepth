#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

import torch
from tasks import SpaceAdventure
from agents import BackInference
from simulate import Simulator

from torch.distributions import Categorical

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context = 'talk', style = 'white', color_codes = True)

runs = 1000000
trials = [2,3]
na = 2
ns = 6
no = 5

labels = ['low', 'high']
trans_prob = torch.ones(runs, 1)
trans_prob[:runs//2] = .9
trans_prob[runs//2:] = .5


costs = torch.FloatTensor([-2., -5.])

space_advent = []
agents = []

conditions = torch.zeros(2, runs, 1, dtype=torch.long)
conditions[0, runs//2:] = 1

res_divergence = []  # mini blocks for which max planning depth agent has a different response    
for T in trials:
    conditions[1] = T
    space_advent.append(SpaceAdventure(conditions, runs=runs, trials=T))
    outcome_likelihood = space_advent[-1].ol
    agents.append(BackInference(outcome_likelihood,
                       runs=runs,
                       trials=T,
                       costs=costs,
                       planning_depth=T))

    agents[-1].set_parameters()
    agents[-1].make_transition_matrix(trans_prob[:,0])
    agents[-1].compute_state_values(0)
    
    init_states = space_advent[-1].states[:, 0, 0]
    D_optim = agents[-1].D[range(runs), 0, -1, init_states].numpy()
    D_suboptim = agents[-1].D[range(runs), 0, :-1, init_states].numpy()
    
    # get the trials on which the choice sign for optimal planning depth
    # differs from all suboptimal planning depths.
    tmp = np.all(np.sign(D_suboptim) != np.sign(D_optim)[:,None], axis=-1)
    tmp *= (D_optim != 0)
    tmp *= np.all(D_suboptim != 0, axis=-1)
    
    if T == 3:
        res = torch.tensor(1.*(D_optim < 0), dtype=torch.long)
        second_state = space_advent[-1].tm[range(runs), 0, res, init_states].argmax(dim=-1)
           
        D_optim = agents[-1].D[range(runs), 0, -2, second_state].numpy()
        D_suboptim = agents[-1].D[range(runs), 0, :-2, second_state].numpy()
        tmp1 = np.all(np.sign(D_suboptim) != np.sign(D_optim)[:,None], axis=-1)
        tmp1 *= (D_optim != 0)
        tmp1 *= np.all(D_suboptim != 0, axis=-1)
        tmp *= tmp1

    
    # remove positive initial states
    negative = agents[-1].Vs[range(runs), 0, -1, init_states].numpy() <= 10
    tmp *= negative
    
    res_divergence.append(tmp.reshape(2, -1))

opt_unique = {}
for i in range(2):
    init_states = space_advent[i].states[:, 0, 0].numpy().reshape(2, -1)
    out_like = space_advent[i].ol.numpy()[:, 0].reshape(2, -1, 6, 5)
    
    for j in range(2):
        in_st = init_states[j]
        ol = out_like[j]
        rd = res_divergence[i][j]
        state_unique = {}
        for s in range(ns):
            state_loc = in_st == s
            tmp = np.unique(ol[state_loc][rd[state_loc]], axis=0)
            # remove confs with 3 or more planets of the same type
            dist_confs = np.all(tmp.sum(axis=1) < 3, axis=-1)
            
            tmp = tmp[dist_confs]
            # remove confs with more than one planet type missing
            dist_confs = np.sum(tmp.sum(axis=1) == 2, axis=-1) < 2
            state_unique[s] = tmp[dist_confs]            

        opt_unique[str(trials[i]) + ' ' + labels[j]] = state_unique
        

# construct experiment
cat = Categorical(logits=torch.zeros(ns))
confs = np.zeros((4, 25, 6, 5))
init_states = np.zeros((4, 25))
conditions = np.zeros((4, 25, 2))

for i, l in enumerate(opt_unique.keys()):
    conds = l.split(' ')
    conditions[i, :, 1] = int(conds[0])
    conditions[i, :, 0] = conds[1] == 'high'
    
    init_states[i] = cat.sample((25,)).numpy()
    for s in range(ns):
        locs = init_states[i] == s
        confs[i, locs] = opt_unique[l][s][:locs.sum()]

init_states = init_states.reshape(-1)
confs = confs.reshape(-1, 6, 5)
conditions = conditions.reshape(-1, 2)

# simulate behavior

runs = 100
mini_blocks = 100

tt_s0 = torch.tensor(init_states, dtype=torch.long).view(1, -1).repeat(runs, 1)

tt_conds = torch.tensor(conditions.T, dtype=torch.long).view(2, 1, -1)
tt_conds = tt_conds.repeat(1, runs, 1)

tt_confs = torch.tensor(confs, dtype=torch.float).view(1, mini_blocks, ns, no)
tt_confs = tt_confs.repeat(runs, 1, 1, 1)


costs = torch.tensor([-2., -5])
fuel = torch.arange(-20., 30., 10) 
agents = []
simulations = []
performance = []
for i in range(3):
    
    # define space adventure task with aquired configurations
    # set number of trials to the max number of actions
    space_advent = SpaceAdventure(tt_conds,
                                  outcome_likelihoods=tt_confs,
                                  init_states=tt_s0,
                                  runs=runs,
                                  mini_blocks=mini_blocks,
                                  trials=3)
    
    # define the optimal agent, each with a different maximal planning depth
    agent = BackInference(tt_confs,
                          runs=runs,
                          mini_blocks=mini_blocks,
                          trials=3,
                          planning_depth=i+1)
    
    agent.set_parameters()
    
    # simulate behavior
    sim = Simulator(space_advent, 
                    agent, 
                    runs=runs, 
                    mini_blocks=mini_blocks,
                    trials=3)
    sim.simulate_experiment()
    
    simulations.append(sim)
    agents.append(agent)
    
    points = costs[simulations[-1].responses] + fuel[simulations[-1].outcomes]
    points[simulations[-1].outcomes<0] = 0
    performance.append(points.sum(dim=-1))
    
for i in range(3):
    plt.figure()
    plt.plot(performance[i].numpy().cumsum(axis=-1).T, 'b')


plt.figure()
labels = [r'd=1', r'd=2', r'd=3']
for i in range(3):
    plt.hist(performance[i].numpy().cumsum(axis=-1)[...,-1], label=labels[i])
plt.legend()
    
# np.save('startsT%d.npy' % trials, np.array(inits))
# np.save('confsT%d.npy' % trials, np.array(confs))