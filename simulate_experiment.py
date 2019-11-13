#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: Dimitrije Markovic
"""

import torch
from tasks import SpaceAdventure
from agents import BackInduction
from simulate import Simulator

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='talk', style='white', color_codes=True)

runs = 100
mini_blocks = 100
na = 2
ns = 6
no = 5

import scipy.io as io
exp = io.loadmat('./experiment/experimental_variables_new.mat')
starts = exp['startsExp'][:, 0] - 1
planets = exp['planetsExp'] - 1
vect = np.eye(5)[planets]


ol1 = torch.from_numpy(vect)
ol2 = torch.from_numpy(np.vstack([vect[50:], vect[:50]]))

starts1 = torch.from_numpy(starts)
starts2 = torch.from_numpy(np.hstack([starts[50:], starts[:50]]))
    
noise = np.tile(np.array([0, 1, 0, 1]), (25,1)).T.flatten()
trials1 = np.tile(np.array([2, 2, 3, 3]), (25,1)).T.flatten()
trials2 = np.tile(np.array([3, 3, 2, 2]), (25,1)).T.flatten()

costs = torch.FloatTensor([-2, -5])  # action costs
fuel = torch.arange(-20., 30., 10.)  # fuel reward of each planet type

confs = torch.stack([ol1, ol2])
confs = confs.view(2, 1, mini_blocks, ns, no).repeat(1, runs//2, 1, 1, 1)\
        .reshape(-1, mini_blocks, ns, no).float()

starts = torch.stack([starts1, starts2])
starts = starts.view(2, 1, mini_blocks).repeat(1, runs//2, 1)\
        .reshape(-1, mini_blocks)
        
conditions = torch.zeros(2, runs, mini_blocks, dtype=torch.long)
conditions[0] = torch.tensor(noise, dtype=torch.long)[None,:]
conditions[1, :runs//2] = torch.tensor(trials1, dtype=torch.long)
conditions[1, runs//2:] = torch.tensor(trials2, dtype=torch.long)

agents = []
simulations = []
performance = []
for depth in range(3):
    
    # define space adventure task with aquired configurations
    # set number of trials to the max number of actions
    space_advent = SpaceAdventure(conditions,
                                  outcome_likelihoods=confs,
                                  init_states=starts,
                                  runs=runs,
                                  mini_blocks=mini_blocks,
                                  trials=3)
    
    # define the optimal agent, each with a different maximal planning depth
    agent = BackInduction(confs,
                          runs=runs,
                          mini_blocks=mini_blocks,
                          trials=3,
                          planning_depth=depth+1)
    
    agent.set_parameters()
    
    # simulate experiment
    sim = Simulator(space_advent, 
                    agent, 
                    runs=runs, 
                    mini_blocks=mini_blocks,
                    trials=3)
    sim.simulate_experiment()
    
    simulations.append(sim)
    agents.append(agent)
    
    responses = sim.responses.clone()
    responses[torch.isnan(responses)] = 0
    responses = responses.long()
    
    outcomes = sim.outcomes
    
    points = costs[responses] + fuel[outcomes]
    points[outcomes<0] = 0
    performance.append(points.sum(dim=-1))
    
start_points = 1000
for i in range(3):
    plt.figure()
    plt.plot(performance[i].numpy().cumsum(axis=-1).T + start_points, 'b')

plt.figure()
labels = [r'd=1', r'd=2', r'd=3']
for i in range(3):
    plt.hist(performance[i].numpy().cumsum(axis=-1)[...,-1] + start_points, label=labels[i])
plt.vlines(0, 0, 30, linestyle='--', lw=3)
plt.legend()