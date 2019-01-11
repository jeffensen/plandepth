#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:23:10 2018

Test inference of model 

@author: Dimitrije Markovic
"""

import torch
from tasks import SpaceAdventure
from agents import BackInduction
from simulate import Simulator
from inference import Inferrer

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
exp = io.loadmat('./experiment/experimental_variables.mat')
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

costs = torch.FloatTensor([-.2, -.5])  # action costs
fuel = torch.arange(-2., 3., 1.)  # fuel reward of each planet type

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
planning_depth = 3
    
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
                      planning_depth=planning_depth)

trans_pars = torch.arange(-1, 1, 2/runs).view(-1, 1).repeat(1, 3)\
            + torch.tensor([1., 1., 3.]).view(1, -1)
agent.set_parameters(trans_pars)

# simulate behavior
sim = Simulator(space_advent, 
                agent, 
                runs=runs, 
                mini_blocks=mini_blocks,
                trials=3)
sim.simulate_experiment()

responses = sim.responses.clone()
mask = ~torch.isnan(responses)

stimuli = {'conditions': conditions,
           'states': space_advent.states.clone(), 
           'configs': confs}

agent = BackInduction(confs,
                      runs=runs,
                      mini_blocks=mini_blocks,
                      trials=3,
                      planning_depth=3)

infer = Inferrer(agent, stimuli, responses, mask)
infer.fit(num_iterations=500, parametrisation='horseshoe')

plt.figure()
plt.plot(infer.loss[-200:])

labels = [r'$\beta$', r'$\kappa$', r'$\epsilon$']

pars_df, scales_df, mg_df, sg_df, post_depth = infer.sample_from_posterior(labels)

pars_df = pars_df.melt(id_vars='subject', var_name='parameter')

g = sns.FacetGrid(pars_df, col="parameter", height=3);
g = (g.map(sns.lineplot, 'subject', 'value', ci='sd'));

for i in range(len(labels)):
    g.axes[0,i].plot(np.arange(1,runs+1), trans_pars[:,i].numpy(),'ro', markersize = 4, zorder=0);
    g.axes[0,i].set_ylim([0, 6])
    
g = sns.PairGrid(mg_df)
g = g.map_diag(sns.kdeplot)
g = g.map_offdiag(plt.scatter)

g = sns.PairGrid(sg_df)
g = g.map_diag(sns.kdeplot)
g = g.map_offdiag(plt.scatter)

def posterior_accuracy(labels, df, vals):
    for i, lbl in enumerate(labels):
        std = df.loc[df['parameter'] == lbl].groupby(by='subject').std()
        mean = df.loc[df['parameter'] == lbl].groupby(by='subject').mean()
        print(lbl, np.sum(((mean+2*std).values[:, 0] > vals[i])*((mean-2*std).values[:, 0] < vals[i]))/runs)

vals = [trans_pars[:,0].numpy(), trans_pars[:, 1].numpy(), trans_pars[:,2].numpy()]
posterior_accuracy(labels, pars_df, vals)
