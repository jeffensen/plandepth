#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit models to behavioural data.

@author: Dimitrije Markovic
"""

import torch

from scipy import io

import pandas as pd

from torch import zeros, ones

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = 'talk', style = 'white', color_codes = True)

from agents import BackInduction
from inference import Inferrer

# function for plotting asymetric errorbars
def errorplot(*args, **kwargs):
    subjects = args[0]
    values = args[1].values
    
    unique_subjects = np.unique(subjects)
    nsub = len(unique_subjects)
    
    values = values.reshape(-1, nsub)
    
    quantiles = np.percentile(values, [5, 50, 95], axis=0)
    
    low_perc = quantiles[0]
    up_perc = quantiles[-1]
    
    x = unique_subjects
    y = quantiles[1]

    assert np.all(low_perc <= y)
    assert np.all(y <= up_perc)
    
    kwargs['yerr'] = [y-low_perc, up_perc-y]
    kwargs['linestyle'] = ''
    kwargs['marker'] = 'o'
    
    plt.errorbar(x, y, **kwargs)

def sigmoid(x):
    return 1./(1+np.exp(-x))

def map_noise_to_values(strings):
    for s in strings:
        if s[0] == 'high':
            yield 1
        elif s[0] == 'low':
            yield 0
        else:
            yield np.nap
        
from pathlib import Path
home = str(Path.home())    
#path = home + '/Dropbox/Experiments/Data/Plandepth/Main/'
#f1 = ['part_23_20-Jun-2018.mat',
#      'part_25_20-Jun-2018.mat',
#      'part_27_21-Jun-2018.mat',
#      'part_29_21-Jun-2018.mat',
#      'part_31_21-Jun-2018.mat',
#      'part_33_21-Jun-2018.mat',
#      'part_35_22-Jun-2018.mat',
#      'part_37_22-Jun-2018.mat',
#      'part_39_22-Jun-2018.mat',
#      'part_43_22-Jun-2018_17-54.mat',
#      'part_45_25-Jun-2018_15-33.mat',
#      'part_47_25-Jun-2018_17-21.mat',
#      'part_49_26-Jun-2018.mat',
#      'part_51_26-Jun-2018_18-31.mat',
#      'part_55_03-Jul-2018_09-33.mat',
#      'part_57_03-Jul-2018_11-17.mat',
#      'part_61_03-Jul-2018_16-16.mat',
#      'part_63_03-Jul-2018_18-48.mat',
#      'part_65_04-Jul-2018_10-42.mat',
#      'part_67_13-Jul-2018_09-48.mat']
#
#f2 = ['part_24_20-Jun-2018.mat',
#      'part_28_21-Jun-2018.mat',
#      'part_30_21-Jun-2018.mat',
#      'part_32_21-Jun-2018.mat',
#      'part_34_21-Jun-2018.mat',
#      'part_38_22-Jun-2018.mat',
#      'part_40_22-Jun-2018_15-20.mat',
#      'part_42_22-Jun-2018_16-57.mat',
#      'part_44_25-Jun-2018_14-30.mat',
#      'part_46_25-Jun-2018_16-34.mat',
#      'part_48_25-Jun-2018_18-28.mat',
#      'part_50_26-Jun-2018_17-38.mat',
#      'part_52_27-Jun-2018_09-53.mat',
#      'part_54_27-Jun-2018_12-21.mat',
#      'part_56_03-Jul-2018_10-32.mat',
#      'part_58_03-Jul-2018_12-34.mat',
#      'part_60_03-Jul-2018_14-42.mat',
#      'part_62_03-Jul-2018_17-34.mat',
#      'part_64_04-Jul-2018_10-10.mat',
#      'part_66_04-Jul-2018_11-41.mat']

path = home + '/mycloud/Shared/Advanced_Adventure/data/Main Experiment/'
f1 = ['part_2_29-Oct-2018_13-52.mat',
      'part_7_19-Nov-2018_15-43.mat',
      'part_10_30-Nov-2018_14-13.mat',
      'part_11_30-Nov-2018_14-34.mat',
      'part_13_04-Dec-2018_15-49.mat',
      'part_16_14-Dec-2018_14-20.mat',
      'part_17_14-Dec-2018_15-29.mat',
      'part_20_04-Feb-2019_15-47.mat',
      'part_23_11-Feb-2019_15-27.mat']

f2 = ['part_1_29-Oct-2018_13-05.mat',
      'part_4_07-Nov-2018_16-56.mat',
      'part_5_14-Nov-2018_14-49.mat',
      'part_9_19-Nov-2018_16-50.mat',
      'part_12_04-Dec-2018_13-46.mat',
      'part_14_12-Dec-2018_15-06.mat',
      'part_15_12-Dec-2018_16-30.mat',
      'part_18_10-Jan-2019_15-49.mat',
      'part_19_10-Jan-2019_16-55.mat',
      'part_21_04-Feb-2019_15-25.mat',
      'part_22_11-Feb-2019_13-30.mat']

filenames = f1 + f2
runs = len(filenames)  # number of subjects
mini_blocks = 100  # number of mini blocks in each run
max_trials = 3  # maximal number of trials within a mini block
max_depth = 3  # maximal planning depth

na = 2  # number of actions
ns = 6 # number of states/locations
no = 5 # number of outcomes/rewards

responses = zeros(runs, mini_blocks, max_trials)
states = zeros(runs, mini_blocks, max_trials+1, dtype=torch.long)
scores = zeros(runs, mini_blocks, max_depth)
conditions = zeros(2, runs, mini_blocks, dtype=torch.long)
confs = zeros(runs, mini_blocks, 6, dtype=torch.long)
for i,f in enumerate(filenames):
    parts = f.split('_')
    tmp = io.loadmat(path+f)
    responses[i] = torch.from_numpy(tmp['data']['Responses'][0,0]['Keys'][0,0]-1)
    states[i] = torch.from_numpy(tmp['data']['States'][0,0] - 1).long()
    confs[i] = torch.from_numpy(tmp['data']['PlanetConf'][0,0] - 1).long()
    scores[i] = torch.from_numpy(tmp['data']['Points'][0,0])
    strings = tmp['data']['Conditions'][0,0]['noise'][0,0][0]
    conditions[0, i] = torch.tensor(list(map_noise_to_values(strings)), dtype=torch.long)
    conditions[1, i] = torch.from_numpy(tmp['data']['Conditions'][0,0]['notrials'][0,0][:,0]).long()

states[states < 0] = -1
confs = torch.eye(no)[confs]

stimuli = {'conditions': conditions,
           'states': states, 
           'configs': confs}

mask = ~torch.isnan(responses)

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
pars_df, scales_df, mg_df, sg_df = infer.sample_from_posterior(labels)

pars_df[r'$\beta$'] = np.exp(pars_df[r'$\beta$'].values)
pars_df[r'$\epsilon$'] = sigmoid(pars_df[r'$\epsilon$'].values)

pars_df = pars_df.melt(id_vars='subject', var_name='parameter')

g = sns.FacetGrid(pars_df, col="parameter", height=3, sharey=False);
g = (g.map(errorplot, 'subject', 'value'));

g = sns.PairGrid(mg_df)
g = g.map_diag(sns.kdeplot)
g = g.map_offdiag(plt.scatter)

g = sns.PairGrid(sg_df)
g = g.map_diag(sns.kdeplot)
g = g.map_offdiag(plt.scatter)


ng = len(f1)  # number of participants in the first group

n_samples = 100
post_depth = infer.sample_posterior_marginal(n_samples=n_samples)

probs = np.zeros((mini_blocks, max_trials-1, n_samples, runs, 3))
for b in range(mini_blocks):
    for t in range(max_trials-1):
        tmp = post_depth['d_{}_{}'.format(b, t)]
        probs[b, t, :, :ng] = tmp[:, :ng]
        if b < 50:
            probs[b+50 , t, :, ng:] = tmp[:, ng:]
        else:
            probs[b-50, t, :, ng:] = tmp[:, ng:]

count = np.array([np.sum(probs.argmax(-1) == i, axis=-2) for i in range(3)])
trial_count = count.sum(-1)
trial_count = trial_count/trial_count.sum(0)            
for i in range(max_trials-1):
    plt.figure(figsize=(10, 5))
    sns.heatmap(trial_count[..., i], cmap='viridis')
    
cond_count = count.reshape(3, 4, 25, 2, runs).sum(-3)
cond_probs = cond_count/cond_count.sum(0)

plt.figure(figsize=(10, 5))
sns.boxplot(data=pd.DataFrame(data=cond_probs[[1, 1, 2, 2], [0, 1, 2, 3], 0].T, 
                              columns=['1', '2', '3', '4']), 
                              color='b')
plt.ylim([0, 1.01])
plt.xlabel('phase')
plt.ylabel('exceedance probability')
plt.savefig('ep_expdata_alcoholics.pdf', bbox_inches='tight', transparent=True ,dpi=600)