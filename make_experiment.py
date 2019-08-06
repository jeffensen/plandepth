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
confs = np.eye(5)[planets]

tmp = np.load('exp_conf_data.npz')
exp1_confs = tmp['arr_0'][0]
exp1_cond = tmp['arr_1'][:, 0]
exp1_init = tmp['arr_2'][0]
exp1_diff = tmp['arr_3']
exp1_diff[20:, :] = np.hstack( [exp1_diff[20:, 50:], exp1_diff[20:, :50]])

exp0_diff = np.load('exp0_clasiffication_diff.npy') 

e1c = exp1_confs.reshape(4, 25, ns, no)
e1i = exp1_init.reshape(4, 25)
sort = {0: {}, 1: {}, 2: {}, 3: {}}
for i, exp in enumerate(exp1_diff.sum(0).reshape(4, 25)):
    for j in range(10):
        loc = exp == j
        if sum(loc) > 0:
            sort[i][j] = [e1c[i, loc], e1i[i, loc]]

expnew_confs = confs.copy().reshape(4, 25, ns, no)
expnew_init = starts.copy().reshape(4, 25)
expnew_sum = exp0_diff.sum(0).reshape(4, 25)
for i, exp in enumerate(exp0_diff.sum(0).reshape(4, 25)):
    vals = np.unique(exp)
    for v in vals[::-1]:
        loc = exp == v
        n = loc.sum()
        count = 0
        while n > 0 and count < v:
            if count in sort[i].keys():
                m = sort[i][count][0].shape[0]
                print('replace value {} with new value {} in condition {}'.format(v, count, i))
                print('number of availible confs is {}'.format(m))
                print('number of required confs is {}'.format(n))

                if m >= n:
                    expnew_confs[i, loc] = sort[i][count][0][:n]
                    expnew_init[i, loc] = sort[i][count][1][:n]
                    expnew_sum[i, loc] = count
                    if m == n:
                        sort[i].pop(count)
                    else:
                        sort[i][count] = [sort[i][count][0][n:], sort[i][count][1][n:]]
                    n = 0
                else:
                    enc = expnew_confs[i, loc] 
                    eni = expnew_init[i, loc]
                    ens = expnew_sum[i, loc]
                    enc[:m] = sort[i][count][0]
                    eni[:m] = sort[i][count][1]
                    ens[:m] = count
                    expnew_confs[i, loc] = enc
                    expnew_init[i, loc] = eni
                    expnew_sum[i, loc] = ens
                    e = exp[loc]
                    e[:m] = count
                    exp[loc] = e
                    loc = exp == v
                    sort[i].pop(count)
                    count += 1
                    n -= m
            else:
                count += 1
                    
                

plt.figure()
plt.plot(exp1_diff.mean(0))
plt.plot(exp0_diff.mean(0))

plt.figure()
df = pd.DataFrame()
df['exp 1'] = exp1_diff.mean(0)
df['exp 0'] = exp0_diff.mean(0)
df['cond'] = np.tile(np.arange(1, 5)[:, None], 25).reshape(-1)
df = df.melt(id_vars=['cond'], value_vars=['exp 1', 'exp 0'])
sns.boxplot(x='cond', y='value', data=df, hue = 'variable')
    