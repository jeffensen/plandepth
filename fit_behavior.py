#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

import torch
from tasks import MultiStage

import scipy as scp

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = 'talk', style = 'white', color_codes = True)

import agents

path = '/home/markovic/Dropbox/Experiments/Data/Plandepth/'
filenames = ['part_1_23-Mar-2018.mat',
             'part_2_23-Mar-2018.mat',
             'part_3_27-Mar-2018.mat',
             'part_4_27-Mar-2018.mat',
             'part_5_27-Mar-2018.mat',
             'part_6_27-Mar-2018.mat',
             'part_7_27-Mar-2018.mat',
             'part_8_27-Mar-2018.mat',
             'part_9_28-Mar-2018.mat',
             'part_10_28-Mar-2018.mat',
             'part_11_28-Mar-2018.mat',
             'part_12_28-Mar-2018.mat',
             'part_13_28-Mar-2018.mat',
             'part_14_28-Mar-2018.mat',
             'part_15_28-Mar-2018.mat',
             'part_16_28-Mar-2018.mat',
             'part_17_29-Mar-2018.mat',
             'part_18_29-Mar-2018.mat',
             'part_19_29-Mar-2018.mat',
             'part_20_29-Mar-2018.mat']

n_subjects = len(filenames) # number of runs/subjects
blocks = 100 # number of mini blocks in each run
max_trials = 3 # maximal number of trials within a mini block
max_depth = 3 # maximal planning depth

responses = np.zeros((n_subjects, blocks, max_trials))
states = np.zeros((n_subjects, blocks, max_trials+1))
confs = np.zeros((n_subjects, blocks, 6))
for i,f in enumerate(filenames):
    parts = f.split('_')
    tmp = scp.io.loadmat(path+f)
    responses[i] = tmp['data']['Responses'][0,0]['Keys'][0,0]-1
    states[i] = tmp['data']['States'][0,0] - 1
    confs[i] = tmp['data']['PlanetConf'][0,0] - 1

confs = confs.reshape(-1, 6)
nans = np.any(np.isnan(confs), axis = -1)
confs = torch.LongTensor(confs[~nans]).view(-1, 6)

responses = responses.reshape(-1, max_trials)
responses = torch.from_numpy(responses[~nans]).view(-1, max_trials)

states = states.reshape(-1,max_trials+1)[:,:-1]
states = torch.LongTensor(states[~nans]).view(-1, max_trials)    

agent = agents.Informed(confs, 
                        responses = responses, 
                        states = states, 
                        runs = len(confs), 
                        trials = max_trials, 
                        planning_depth = max_depth)
#results = agent.fit(n_iterations = 50000)