#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: Dimitrije Markovic
"""

import torch
from tasks import SpaceAdventure
from agents import BackInference
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

vect = np.load('confsExp1.npy')
ol1 = torch.from_numpy(vect)
ol2 = torch.from_numpy(np.vstack([vect[50:], vect[:50]]))
vect = np.load('startsExp1.npy')
starts1 = torch.from_numpy(vect)
starts2 = torch.from_numpy(np.hstack([vect[50:], vect[:50]]))
    
noise = np.tile(np.array([0, 1, 0, 1]), (25,1)).T.flatten()
trials1 = np.tile(np.array([2, 2, 3, 3]), (25,1)).T.flatten()
trials2 = np.tile(np.array([3, 3, 2, 2]), (25,1)).T.flatten()

costs = torch.FloatTensor([-2, -5])  # action costs
fuel = torch.arange(-20., 30., 10.)  # fuel reward of each planet type

confs = torch.stack([ol1, ol2])
confs = confs.view(2, 1, mini_blocks, ns, no).repeat(1, runs//2, 1, 1, 1)\
        .reshape(-1, mini_blocks, ns, no)

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
for i in range(3):
    
    # define space adventure task with aquired configurations
    # set number of trials to the max number of actions
    space_advent = SpaceAdventure(conditions,
                                  outcome_likelihoods=confs,
                                  init_states=starts,
                                  runs=runs,
                                  mini_blocks=mini_blocks,
                                  trials=3)
    
    # define the optimal agent, each with a different maximal planning depth
    agent = BackInference(confs,
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