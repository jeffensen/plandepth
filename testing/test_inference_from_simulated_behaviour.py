

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:23:10 2018

Here we will test the validity of the inference procedure for estimating free parameters of the behavioural model. 
In a frist step we will simulate behaviour from the agents with a fixed planning depth and try to recover model 
parameters as mini-block dependent planning depth. In the second step, we will simulate behaviour from agents 
with varying planning depth and try to determine the estimation accuracy of the free model paramters and 
mini-block dependent planning depth.

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
pyro.enable_validation(True)

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

#exp = np.load('newexp.npz')
#vect = exp['arr_0']
#starts = exp['arr_1']


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
states = []
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
    
    m = torch.tensor([0., 0., 0.])
    trans_pars = torch.distributions.Normal(m, 1.).sample((runs,))
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
    states.append(space_advent.states.clone())
    
    responses = simulations[-1].responses.clone()
    responses[torch.isnan(responses)] = -1.
    responses = responses.long()
    points = 10*(costs[responses] + fuel[simulations[-1].outcomes])
    points[simulations[-1].outcomes < 0] = 0
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

sim_number = -1    
responses = simulations[sim_number].responses.clone()
mask = ~torch.isnan(responses)

stimuli = {'conditions': conditions,
           'states': states[sim_number], 
           'configs': confs}

agent = BackInduction(confs,
                      runs=runs,
                      mini_blocks=mini_blocks,
                      trials=max_trials,
                      planning_depth=max_depth)

infer = Inferrer(agent, stimuli, responses, mask)
infer.fit(num_iterations=200, parametrisation='static')

plt.figure()
plt.plot(infer.loss[-150:])

labels = [r'$\beta$', r'$\theta$', r'$\epsilon$']

pars_df, scales_df, mg_df, sg_df  = infer.sample_from_posterior(labels, n_samples=1000)

fig, axes = plt.subplots(3, 1, figsize=(15, 10))
m = pars_df.groupby('subject').mean()
s = pars_df.groupby('subject').std()
for i, l in enumerate(labels):
    tp = trans_pars[:, i].numpy()
    axes[i].errorbar(tp, m[l], 2*s[l], linestyle='', marker='o')
    axes[i].plot(tp, tp, 'k--')
    

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
posterior_accuracy(labels, pars_df.melt(id_vars='subject', var_name='parameter'), vals)

prd1=pyro.param('prd1').detach()
prd2=pyro.param('prd2').detach()

probs = torch.zeros(mini_blocks, max_trials - 1, runs, max_depth)
probs[50:, 0, :20] = prd1[:20, 50:].transpose(dim1=0, dim0=1)
probs[50:, 0, 20:] = prd1[20:, :50].transpose(dim1=0, dim0=1)

probs[:50, 0, :20, :2] = prd2[:20, :50].transpose(dim1=0, dim0=1)
probs[:50, 0, 20:, :2] = prd2[20:, 50:].transpose(dim1=0, dim0=1)

probs[:50, 1, :runs//2, 0] = 1.
probs[50:, 1, :runs//2, :2] = prd2[:runs//2, 50:].transpose(dim1=0, dim0=1)
    
probs[:50, 1, runs//2:, 0] = 1.
probs[50:, 1, runs//2:, :2] = prd2[runs//2:, :50].transpose(dim1=0, dim0=1)

probs = probs.numpy()

trial_count = np.array([np.sum(probs.argmax(-1) == i, axis=-1) for i in range(3)])
trial_count = trial_count/trial_count.sum(0)
            
for i in range(max_trials-1):
    plt.figure(figsize=(10, 5))
    sns.heatmap(trial_count[..., i], cmap='viridis')
    plt.ylabel('planning depth')
    plt.xlabel('mini-block')
    
    
depths = torch.stack(agents[-1].depths).numpy()
depths[0, :runs//2] = 1
depths[:, runs//2:] = np.vstack([depths[50:, runs//2:], depths[:50, runs//2:]])
diff = []
for sub in range(runs):
    diff.append(depths[:, sub] != probs[:, 0, sub].argmax(-1))

diff = np.vstack(diff)
diff[runs//2:] = np.hstack([diff[runs//2:, 50:], diff[runs//2:, :50]])
plt.figure()
plt.plot(diff.mean(0));

plt.figure()
sns.boxplot(data=diff.mean(0).reshape(4, 25).T)
    
sub = 0
plt.figure()
sns.heatmap(probs[:,0, sub].T, cmap='viridis')
plt.plot(range(100), depths[:, sub]+.5, 'wo');
#    plt.savefig('exp_ep_depth_t{}.pdf'.format(i+1), bbox_inches='tight', transparent=True, dpi=600)