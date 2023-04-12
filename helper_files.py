# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:52:39 2023

@author: goenner
"""

#from scipy import io
#import pandas as pd

import torch
from torch import zeros#, ones
import numpy as np
import pylab as plt

#from pathlib import Path
from os.path import join # isdir, expanduser,
from os import walk # listdir, 
import json


def load_and_format_behavioural_data(local_path, filenames): #kann es nicht als komplette funktion, nur wenn ich alles einzeln abspiele und die paths per Hand ergänze
    
    # search local_path and all subdirectories for files named like filename
    #home = str(Path.home())    
    #path = home + local_path # muss man immer händisch machen 
    path = local_path # LG
    fnames = []
    for paths, dirs, files in walk(path): # hier kennt es immer die Filenames nicht, da sie unten erst definiert werden?
        for filename in [f for f in files if f in filenames]:
            fnames.append(join(paths, filename))  # get full paths of alle filename files
   
    # check for exclusion (gameover by negative points or incomplete data)
    for i,f in enumerate(fnames):
        read_file = open(f,"r", encoding='utf-8-sig')
        tmp = json.loads(json.load(read_file)['data']) # assume json file
        read_file.close()
        if all(flag <= 0 for (_, _, flag) in tmp['points']) or len(tmp['points']) != len(tmp['conditions']['noise']): fnames.remove(fnames[i])
        
    runs = len(fnames)  # number of subjects
    
    mini_blocks = 140  # number of mini blocks in each run
    max_trials = 3  # maximal number of trials within a mini block
    max_depth = 3  # maximal planning depth

    #na = 2  # number of actions
    #ns = 6 # number of states/locations
    no = 5 # number of outcomes/rewards

    responses = zeros(runs, mini_blocks, max_trials)
    states = zeros(runs, mini_blocks, max_trials+1, dtype=torch.long)
    scores = zeros(runs, mini_blocks, max_depth)
    conditions = zeros(2, runs, mini_blocks, dtype=torch.long)
    confs = zeros(runs, mini_blocks, 6, dtype=torch.long)
    balance_cond = zeros(runs)
    ids = []
    #subj_IDs = []    
    
    for i,f in enumerate(fnames):
        read_file = open(f,"r", encoding='utf-8-sig')
        # assume json file
        tmp = json.loads(json.load(read_file)['data'])
        read_file.close()
        
        responses[i] = torch.from_numpy(np.array(tmp['responses']['actions']) -1)
        states[i] = torch.from_numpy(np.array(tmp['states']) - 1).long()
        confs[i] = torch.from_numpy(np.array(tmp['planetConfigurations']) - 1).long()
        scores[i] = torch.from_numpy(np.array(tmp['points']))
        strings = tmp['conditions']['noise']
        
        conditions[0, i] = torch.tensor(np.unique(strings, return_inverse=True)[1]*(-1) + 1 , dtype=torch.long)  # "low" -> 0 | "high" -> 1
        conditions[1, i] = torch.from_numpy(np.array(tmp['conditions']['notrials'])).long() 
        
        balance_cond[i] = tmp['balancingCondition'] - 1
        
        # here, ids are just numbered starting from one
        # better would be to include IDs in .json file
        #ids.append(i+1)
        
        #ID = f.split('\\')[0].split('/')[-1]
        #ids.append(ID)        
        
        if f.split('\\')[0] != f:            
            ID = f.split('\\')[0].split('/')[-1]
        else:
            ID = f.split('/')[-2]
        ids.append(ID)                

    states[states < 0] = -1
    confs = torch.eye(no)[confs]

    # define dictionary containing information which participants recieved on each trial
    stimuli = {'conditions': conditions,
               'states': states, 
               'configs': confs}

    mask = ~torch.isnan(responses)
    
    return stimuli, mask, responses, conditions, ids


def get_posterior_stats(post_marg, mini_blocks=140):
    n_samples, runs, max_trials = post_marg['d_0_0'].shape
    post_depth = {0: np.zeros((n_samples, mini_blocks, runs, max_trials)),
              1: np.zeros((n_samples, mini_blocks, runs, max_trials))}
    for pm in post_marg:
        b, t = np.array(pm.split('_')[1:]).astype(int)
        if t in post_depth:
            post_depth[t][:, b] = post_marg[pm]

    # get sample mean over planning depth for the first and second choice
    m_prob = [post_depth[c].mean(0) for c in range(2)]

    # get sample plannign depth exceedance count of the first and second choice
    # exceedance count => number of times planning depth d had highest posterior probability
    exc_count = [np.array([np.sum(post_depth[t].argmax(-1) == i, 0) for i in range(3)]) for t in range(2)]
    
    return post_depth, m_prob, exc_count



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
    
# function for mapping strings ('high', 'low') to numbers 0, 1

def map_noise_to_values(strings):
    for s in strings:
        if s[0] == 'high':
            yield 1
        elif s[0] == 'low':
            yield 0
        else:
            yield np.nan