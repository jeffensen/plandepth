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
import pyro
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scipy.io as io

sys.path.append('../')
from tasks import SpaceAdventure
from agents import BackInduction
from simulate import Simulator
from inference import Inferrer


# set global variables ####################################
torch.manual_seed(16324)
pyro.enable_validation(True)

sns.set(context='talk', style='white', color_codes=True)

runs = 40
mini_blocks = 100
max_trials = 3
max_depth = 3
na = 2
ns = 6
no = 5
###########################################################

exp = io.loadmat('../experiment/experimental_variables_new.mat')
starts = exp['startsExp'][:, 0] - 1
planets = exp['planetsExp'] - 1
vect = np.eye(5)[planets]

ol1 = torch.from_numpy(vect)
ol2 = torch.from_numpy(np.vstack([vect[50:], vect[:50]]))

starts1 = torch.from_numpy(starts)
starts2 = torch.from_numpy(np.hstack([starts[50:], starts[:50]]))

# noise condition low -> 0, high -> 1
noise = np.tile(np.array([0, 1, 0, 1]), (25, 1)).T.flatten()

# max trials
trials1 = np.tile(np.array([2, 2, 3, 3]), (25, 1)).T.flatten()
trials2 = np.tile(np.array([3, 3, 2, 2]), (25, 1)).T.flatten()

costs = torch.FloatTensor([-.2, -.5])  # action costs
fuel = torch.arange(-2., 3., 1.)  # fuel reward of each planet type

confs = torch.stack([ol1, ol2])
confs = confs.view(2, 1, mini_blocks, ns, no).repeat(1, runs//2, 1, 1, 1)\
        .reshape(-1, mini_blocks, ns, no).float()

starts = torch.stack([starts1, starts2])
starts = starts.view(2, 1, mini_blocks).repeat(1, runs//2, 1)\
        .reshape(-1, mini_blocks)

conditions = torch.zeros(2, runs, mini_blocks, dtype=torch.long)
conditions[0] = torch.tensor(noise, dtype=torch.long)[None, :]
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

    m = torch.tensor([+2, 0., -2])
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
plt.hist(torch.stack(performance).numpy().cumsum(axis=-1)[..., -1].T, bins=30, stacked=True)
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
infer.fit(num_iterations=1000)

plt.figure()
plt.plot(infer.loss[-150:])

labels = [r'$\tilde{\beta}$', r'$\theta$', r'$\tilde{\alpha}$']

pars_df, mg_df, sg_df = infer.sample_from_posterior(labels, n_samples=1000)

# plot posterior parameter estimates in relation to true parameter values
fig, axes = plt.subplots(3, 1, figsize=(15, 15))
m = pars_df.groupby('subject').mean()
s = pars_df.groupby('subject').std()
for i, l in enumerate(labels):
    tp = trans_pars[:, i].numpy()
    axes[i].errorbar(tp, m[l], 2*s[l], linestyle='', marker='o')
    axes[i].plot(tp, tp, 'k--')
    axes[i].set_ylabel('estimated value')
    axes[i].text(1, 0.5, l, rotation=-90, transform=axes[i].transAxes)

axes[-1].set_xlabel('true value')

# plot correlation between group level mean
g = sns.PairGrid(mg_df)
g = g.map_diag(sns.kdeplot)
g = g.map_offdiag(plt.scatter)

# plot correlation between group level variance
g = sns.PairGrid(sg_df)
g = g.map_diag(sns.kdeplot)
g = g.map_offdiag(plt.scatter)

# print posterior accuracy of parameter estimates
def posterior_accuracy(labels, df, vals):
    for i, lbl in enumerate(labels):
        std = df.loc[df['parameter'] == lbl].groupby(by='subject').std()
        mean = df.loc[df['parameter'] == lbl].groupby(by='subject').mean()
        print(lbl, np.sum(((mean+2*std).values[:, 0] > vals[i])*((mean-2*std).values[:, 0] < vals[i]))/runs)

vals = [trans_pars[:,0].numpy(), trans_pars[:, 1].numpy(), trans_pars[:, 2].numpy()]
posterior_accuracy(labels, pars_df.melt(id_vars='subject', var_name='parameter'), vals)

n_samples = 100
post_marg = infer.sample_posterior_marginal(n_samples=n_samples)

post_depth = {0: np.zeros((n_samples, mini_blocks, runs, max_trials)),
              1: np.zeros((n_samples, mini_blocks, runs, max_trials))}
for pm in post_marg:
    b, t = np.array(pm.split('_')[1:]).astype(int)
    if t in post_depth:
        post_depth[t][:, b] = post_marg[pm]

# get sample mean over planning depth for the first and second choice
m_prob = [post_depth[d].mean(0) for d in range(2)]

# get sample plannign depth exceedance count of the first and second choice
# exceedance count => number of times planning depth d had highest posterior probability
exc_count = [np.array([np.sum(post_depth[t].argmax(-1) == i, 0) for i in range(3)]) for t in range(2)]

true_depths = torch.stack(agents[sim_number].depths).numpy()

# plot true planning depth and estimated mean posterior depth of the first choice for two runs (one from each group)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

for ns, ax in zip([10, 30], axes):
    sns.heatmap(m_prob[0][:, ns], cmap='viridis', ax=ax)
    ax.plot(true_depths[:, ns]+.5, range(mini_blocks), 'wo')

axes[0].set_title('normal order')
axes[1].set_title('reversed order')
fig.suptitle('mean probability')

# plot true planning depth and depth exceedance probability of the first choice for two runs (one from each group)
# plot true planning depth and estimated mean posterior depth of the first choice for two runs (one from each group)
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

for ns, ax in zip([10, 30], axes):
    sns.heatmap(exc_count[0][..., ns].T/n_samples, cmap='viridis', ax=ax)
    ax.plot(true_depths[:, ns]+.5, range(mini_blocks), 'wo')

axes[0].set_title('normal order')
axes[1].set_title('reversed order')
fig.suptitle('exceedance probability')

# count number of misclassified mini-blocks and plot distribution
# the misclassified mini-block corresponds to one where the highest
# exceedance probability is associated with the wrong planning depth
err_class = ~(exc_count[0].argmax(0) == true_depths)
plt.figure()
plt.hist(err_class.sum(-1))
plt.xlabel('number of errors')
plt.ylabel('count')
plt.title('misclassification distribution')

# plot group level exceedance probability
order = conditions[-1, :, 0].numpy() == 2
ordered_exc_count = np.concatenate([exc_count[0][..., order],
                                    np.concatenate([exc_count[0][:, 50:, ~order], exc_count[0][:, :50, ~order]], -2)],
                                    -1
                                   )
e_probs = ordered_exc_count.sum(-1)/(n_samples*runs)

plt.figure(figsize=(8, 5))
sns.heatmap(e_probs.T, cmap='viridis')
plt.xlabel('planning depth')
plt.ylabel('mini-block')

# def model(obs):
#     N = len(obs)

#     with plate('dim', 3):
#         with plate('rim', N):
#             logits = sample('logits', dist.Normal(0., 1.))

#     probs = ones(N, 3)/3
#     with plate('subs', N):
#         d = sample('d', dist.Categorical(probs), infer={"enumerate": "parallel"})

#         lgts = Vindex(logits)[..., d]

#         print(d.shape, lgts.shape)

#         sample('obs', dist.Bernoulli(logits=lgts), obs=obs)

# guide = AutoDiagonalNormal(model)
# svi = SVI(model=model,
#           guide=guide,
#           optim=Adam(optim_kwargs),
#           loss=TraceEnum_ELBO(num_particles=100,
#                               vectorize_particles=False))
# svi.step(obs)
# def guide(obs):
#     N = len(obs)
#     loc = param('loc', zeros(N, 3))

#     with plate('dim', 3):
#         with plate('rim', N):
#             sample('logits', dist.Normal(loc, 1.))

# svi = SVI(model=model,
#           guide=guide,
#           optim=Adam(optim_kwargs),
#           loss=TraceEnum_ELBO(num_particles=100,
#                               vectorize_particles=False))
# svi.step(obs)
# svi = SVI(model=model,
#           guide=guide,
#           optim=Adam(optim_kwargs),
#           loss=TraceEnum_ELBO(num_particles=100,
#                               vectorize_particles=True))
# svi.step(obs)