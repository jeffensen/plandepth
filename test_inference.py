#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:23:10 2018

Test inference of model 

@author: Dimitrije Markovic
"""

import torch
import pandas as pd
from tasks import SpaceAdventure
from agents import BackInduction
from simulate import Simulator
from inference import Inferrer

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pyro
pyro.enable_validation()

sns.set(context='talk', style='white', color_codes=True)

runs = 40
mini_blocks = 100
max_trials = 3
max_depth = 3
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

# noise condition low -> 0, high -> 1     
noise = np.tile(np.array([0, 1, 0, 1]), (25,1)).T.flatten()

# max trials
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

for i in range(3):
    
    # define space adventure task with aquired configurations
    # set number of trials to the max number of actions
    space_advent = SpaceAdventure(conditions,
                                  outcome_likelihoods=confs,
                                  init_states=starts,
                                  runs=runs,
                                  mini_blocks=mini_blocks,
                                  trials=max_trials)
    
    # define the optimal agent, each with a different maximal planning depth
    agent = BackInduction(confs,
                          runs=runs,
                          mini_blocks=mini_blocks,
                          trials=3,
                          planning_depth=i+1)
    
    trans_pars = torch.arange(-1, 0, 1/runs).view(-1, 1).repeat(1, 3)\
            + torch.tensor([3., 0., 5.]).view(1, -1)
    agent.set_parameters(trans_pars)
    
    # simulate behavior
    sim = Simulator(space_advent, 
                    agent, 
                    runs=runs, 
                    mini_blocks=mini_blocks,
                    trials=3)
    sim.simulate_experiment()
    
    simulations.append(sim)
    agents.append(agent)
    
    responses = simulations[-1].responses.clone()
    responses[torch.isnan(responses)] = -1.
    responses = responses.long()
    points = 10*(costs[responses] + fuel[simulations[-1].outcomes])
    points[simulations[-1].outcomes<0] = 0
    performance.append(points.sum(dim=-1))
    
for i in range(3):
    plt.figure()
    plt.plot(performance[i].numpy().cumsum(axis=-1).T, 'b')


plt.figure(figsize=(10, 5))
labels = [r'd=1', r'd=2', r'd=3']
plt.hist(torch.stack(performance).numpy().cumsum(axis=-1)[...,-1].T, bins=30, stacked=True);
plt.legend(labels)
plt.ylabel('count')
plt.xlabel('score')
plt.savefig('finalscore_exp.pdf', bbox_inches='tight', transparent=True, dpi=600)
    
responses = simulations[-1].responses.clone()
mask = ~torch.isnan(responses)

stimuli = {'conditions': conditions,
           'states': space_advent.states.clone(), 
           'configs': confs}

agent = BackInduction(confs,
                      runs=runs,
                      mini_blocks=mini_blocks,
                      trials=max_trials,
                      planning_depth=max_depth)

infer = Inferrer(agent, stimuli, responses, mask)
infer.fit(num_iterations=400, parametrisation='dynamic')

plt.figure()
plt.plot(infer.loss[-200:])

labels = [r'$\beta$', r'$\theta$', r'$\epsilon$']

pars_df, scales_df, mg_df, sg_df  = infer.sample_from_posterior(labels)

pars_df = pars_df.melt(id_vars='subject', var_name='parameter')

g = sns.FacetGrid(pars_df, col="parameter", height=5);
g = (g.map(sns.lineplot, 'subject', 'value', ci='sd'));

for i in range(len(labels)):
    g.axes[0,i].plot(np.arange(1,runs+1), trans_pars[:,i].numpy(),'ro', markersize = 4, zorder=0);
    
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

vals = [trans_pars[:,0].numpy(), trans_pars[:, 1].numpy(), trans_pars[:, 2].numpy()]
posterior_accuracy(labels, pars_df, vals)

n_samples = 100
post_depth = infer.sample_posterior_marginal(n_samples=n_samples)

probs = np.zeros((mini_blocks, max_trials-1, n_samples, runs, 3))
for b in range(mini_blocks):
    for t in range(max_trials-1):
        tmp = post_depth['d_{}_{}'.format(b, t)]
        probs[b, t, :, :20] = tmp[:, :20]
        if b < 50:
            probs[b+50 , t, :, 20:] = tmp[:, 20:]
        else:
            probs[b-50, t, :, 20:] = tmp[:, 20:]

count = np.array([np.sum(probs.argmax(-1) == i, axis=-2) for i in range(3)])
trial_count = count.sum(-1)
trial_count = trial_count/trial_count.sum(0)            
for i in range(max_trials-1):
    plt.figure(figsize=(10, 5))
    sns.heatmap(trial_count[..., i], cmap='viridis')
    plt.ylabel('planning depth')
    plt.xlabel('mini-block')
    plt.savefig('exp_ep_depth_t{}.pdf'.format(i+1), bbox_inches='tight', transparent=True, dpi=600)
    
cond_count = count.reshape(3, 4, 25, 2, runs).sum(-3)
cond_probs = cond_count/cond_count.sum(0)

plt.figure(figsize=(10, 5))
sns.boxplot(data=pd.DataFrame(data=cond_probs[[1, 1, 2, 2], [0, 1, 2, 3], 0].T, 
                              columns=['1', '2', '3', '4']), 
                              color='b')
plt.ylim([0, 1.01])
plt.xlabel('phase')
plt.ylabel('exceedance probability')
plt.savefig('ep_experiment.pdf', bbox_inches='tight', transparent=True, dpi=600)