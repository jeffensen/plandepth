#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

import torch
import pyro
from tasks import SpaceAdventure
from agents import BackInduction
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


space_advent = []
agents = []

conditions = torch.zeros(2, runs, 1, dtype=torch.long)
conditions[0, runs//2:] = 1

res_divergence = []  # mini blocks for which max planning depth agent has a different response    
for T in trials:
    conditions[1] = T
    space_advent.append(SpaceAdventure(conditions, runs=runs, trials=T))
    outcome_likelihood = space_advent[-1].ol
    agents.append(BackInduction(outcome_likelihood,
                       runs=runs,
                       trials=T,
                       planning_depth=T))
    
    trans_pars = torch.tensor([0., 0., 10.]).repeat(runs, 1)
    agents[-1].set_parameters(trans_pars)
    agents[-1].make_transition_matrix(trans_prob[:,0])
    agents[-1].compute_state_values(0)
    
    init_states = space_advent[-1].states[:, 0, 0]
    D_optim = agents[-1].D[-1][-1, range(runs), init_states].numpy()
    D_suboptim = agents[-1].D[-1][:-1, range(runs), init_states].numpy()
    
    # get the trials on which the choice sign for optimal planning depth
    # differs from all suboptimal planning depths.
    tmp = np.all(np.sign(D_suboptim) != np.sign(D_optim), axis = 0)
    tmp *= np.all(np.abs(D_suboptim - D_optim) > 1.3, axis=0)
    tmp *= (D_optim != 0)
    tmp *= np.all(D_suboptim != 0, axis=0)
    
    # remove to positive initial states
    negative = agents[-1].Vs[-1][-1, range(runs), init_states].numpy() <= 10
    tmp *= negative
    
    res_divergence.append(tmp.reshape(2, -1))

opt_unique = {}

# iterate over numbr of trials
for i in range(2):
    init_states = space_advent[i].states[:, 0, 0].numpy().reshape(2, -1)
    out_like = space_advent[i].ol.numpy()[:, 0].reshape(2, -1, 6, 5)
    
    # iterate over noise level
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


### good miniblocks from the first experiment
# array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 17, 18,
#        19, 20, 21, 22, 24, 25, 28, 29, 35, 41, 43, 44, 46, 48, 49, 52, 53, 
#        54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72,
#        73, 74, 75, 76, 81, 83, 84, 85, 87, 88, 89, 93, 94, 97])
       

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

runs = 40
mini_blocks = 100

tt_s0 = torch.tensor(init_states, dtype=torch.long).view(1, -1).repeat(runs, 1)
tt_s0[runs//2:] = torch.stack([tt_s0[runs//2:, 50:], tt_s0[runs//2:, :50]], 1).reshape(runs//2, -1)

tt_conds = torch.tensor(conditions.T, dtype=torch.long).view(2, 1, -1)
tt_conds = tt_conds.repeat(1, runs, 1)
tt_conds[:, runs//2:] = torch.stack([tt_conds[:, runs//2:, 50:], tt_conds[:, runs//2:, :50]], 2).reshape(2, runs//2, -1)

tt_confs = torch.tensor(confs, dtype=torch.float).view(1, mini_blocks, ns, no)
tt_confs = tt_confs.repeat(runs, 1, 1, 1)

tt_confs[runs//2:] = torch.stack([tt_confs[runs//2:, 50:], tt_confs[runs//2:, :50]], 1).reshape(runs//2, -1, ns, no)

costs = torch.tensor([-2., -5.])
fuel = torch.arange(-20., 30., 10) 
agents = []
simulations = []
performance = []
states = []
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
    agent = BackInduction(tt_confs,
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
    points = costs[responses] + fuel[simulations[-1].outcomes]
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

#plt.savefig('finalscore_opt.pdf', bbox_inches='tight', transparent=True, dpi=600)
    
# np.save('startsT%d.npy' % trials, np.array(inits))
# np.save('confsT%d.npy' % trials, np.array(confs))

# test parameter recovery
from inference import Inferrer
max_trials = 3
max_depth = 3

results = {1: {}, 2: {}, 3: {}}
for i in range(2,3):
    responses = simulations[i].responses.clone()
    mask = ~torch.isnan(responses)
    
    stimuli = {'conditions': tt_conds,
               'states': states[i], 
               'configs': tt_confs}
    
    agent = BackInduction(tt_confs,
                          runs=runs,
                          mini_blocks=mini_blocks,
                          trials=max_trials,
                          planning_depth=max_depth)
    
    infer = Inferrer(agent, stimuli, responses, mask)
    infer.fit(num_iterations=200, parametrisation='static')
    
    results[i+1]['inference'] = infer 
    
    plt.figure()
    plt.plot(infer.loss[-200:])
    
    results[i+1]['loss'] = infer.loss
    
    labels = [r'$\beta$', r'$\kappa$', r'$\epsilon$']
    
    pars_df, scales_df, mg_df, sg_df  = infer.sample_from_posterior(labels, n_samples=1000)
    
    results[i+1]['pars'] = [pars_df, scales_df, mg_df, sg_df]
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    m = pars_df.groupby('subject').mean()
    s = pars_df.groupby('subject').std()
    for i, l in enumerate(labels):
        tp = trans_pars[:, i].numpy()
        axes[i].errorbar(tp, m[l], 2*s[l], linestyle='', marker='o')
        axes[i].plot(tp, tp, 'k--')
        axes[i].set_title(l)
    
    g = sns.PairGrid(mg_df)
    g = g.map_diag(sns.kdeplot)
    g = g.map_offdiag(plt.scatter)
    
    g = sns.PairGrid(sg_df)
    g = g.map_diag(sns.kdeplot)
    g = g.map_offdiag(plt.scatter)
    
    probs = torch.zeros((mini_blocks, max_trials-1, runs, 3))
    prd1 = pyro.param('prd1').detach()
    prd2 = pyro.param('prd2').detach()

    probs[50:, 0, :20] = prd1[:20, 50:].transpose(dim1=0, dim0=1)
    probs[50:, 0, 20:] = prd1[20:, :50].transpose(dim1=0, dim0=1)
    
    probs[:50, 0, :20, :2] = prd2[:20, :50].transpose(dim1=0, dim0=1)
    probs[:50, 0, 20:, :2] = prd2[20:, 50:].transpose(dim1=0, dim0=1)
    
    probs[:50, 1, :runs//2, 0] = 1.
    probs[50:, 1, :runs//2, :2] = prd2[:runs//2, 50:].transpose(dim1=0, dim0=1)
        
    probs[:50, 1, runs//2:, 0] = 1.
    probs[50:, 1, runs//2:, :2] = prd2[runs//2:, :50].transpose(dim1=0, dim0=1)

    probs = probs.numpy()
    
    results[i+1]['post_depth'] = probs.copy()
    
    count = np.array([(probs.argmax(-1) == i).sum(-1) for i in range(3)])
    trial_count = count/count.sum(0)            
    for j in range(max_trials-1):
        plt.figure(figsize=(10, 5))
        sns.heatmap(trial_count[..., j], cmap='viridis')
        plt.yticks(ticks = [0.5, 1.5, 2.5], labels=range(1, 4))
        plt.ylabel('planning depth')
        plt.xticks(ticks= np.arange(.5, 100., 5), labels = range(1, 101, 5))
        plt.xlabel('mini-block')
        #plt.savefig('opt_ep_depth_t{}.pdf'.format(i+2), bbox_inches='tight', transparent=True, dpi=600)
    
    depths = torch.stack(agents[-1].depths).numpy()
    depths[0, :runs//2] = 1
    depths[:, runs//2:] = np.vstack([depths[50:, runs//2:], depths[:50, runs//2:]])

    diff = []
    for sub in range(runs):
        diff.append(depths[:, sub] != probs[:, 0, sub].argmax(-1))

    diff = np.vstack(diff)
    plt.figure()
    plt.plot(diff.mean(0));
    
    plt.figure()
    sns.boxplot(data=diff.mean(0).reshape(4, 25).T)
    
    sub = 0
    plt.figure()
    sns.heatmap(probs[:,0, sub].T, cmap='viridis')
    plt.plot(range(100), depths[:, sub]+.5, 'wo');