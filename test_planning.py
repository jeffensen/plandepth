#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""
from tqdm import tqdm

import numpy as np

import torch
from torch.distributions import Categorical, constraints
from torch.autograd import Variable

from helpers import make_transition_matrix

import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam

zeros = torch.zeros
ones = torch.ones
var = Variable
randn = torch.randn
softplus = torch.nn.Softplus()

ftype = torch.FloatTensor
itype = torch.LongTensor
btype = torch.ByteTensor

na = 2
ns = 6

#state transition matrix
p = [.9, .5];
stm_forward = make_transition_matrix(p[0], ns, na)
stm_backward = stm_forward.transpose(dim0=2,dim1=1)
stm_backward = stm_backward/stm_backward.sum(dim=-1)[:,:,None]

#for plotting
colors = np.array(['red', 'pink', 'gray', 'blue', 'green'])
    
#load single planetary sistem
#vect = np.load('confsExp1.npy')
#outcome_likelihood = vect[0].astype(int)

outcomes_likelihood = np.array([[0,0,1.,0,0],[0,0,1.,0,0], [0,0,1.,0,0], [0,0,1.,0,0], [0,0,1.,0,0], [0,0,1.,0,0]])

#load starting condition
#vect = np.load('startsExp1.npy')
#starts = vect[0]

starts = 2

policies = itype([[0,0],[0,1],[1,0],[1,1]])
npi = len(policies)

p0 = zeros(ns)
p0[starts] = 1

values = ftype(outcomes_likelihood.argmax(axis = -1)- 2.)
pT = torch.exp(values-values.max())
pT /= pT.sum()

for depth in range(1,3):
    forward_beliefs = zeros(npi, ns, depth+1)
    forward_beliefs[:,:,0] = p0[None,:]
    backward_beliefs = zeros(npi, ns, depth+1)
    backward_beliefs[:,:,-1] = pT[None,:]
    for tau in range(1,depth+1):
        a = policies[:,tau-1]
        forward_beliefs[:,:,tau] = torch.sum(stm_forward[a]*forward_beliefs[:,:,tau-1,None], 1)
        backward_beliefs[:,:,-tau-1] = torch.sum(stm_backward[a]*backward_beliefs[:,:,-tau, None], 1)

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, npi, figsize = (16, 8))

    for i in range(npi):
        sns.heatmap(data = forward_beliefs[i], ax = axes[0,i], vmax = 1, vmin = 0, cmap = 'viridis')
        sns.heatmap(data = backward_beliefs[i], ax = axes[1,i], vmax = 1, vmin = 0, cmap = 'viridis')


    logp = torch.log(forward_beliefs)/2 + torch.log(backward_beliefs)/2

    q = torch.exp(logp)
    q /= q.sum(dim=1)[:,None,:]

    qlq = q*torch.log(q)
    qlq[torch.isnan(qlq)] = 0
    H = -qlq.reshape(4,-1).sum(dim=-1)

    qlp = q*logp
    qlp[torch.isnan(qlp)] = 0
    U = -qlp.reshape(4,-1).sum(dim=-1)

    F = U - H
    print('preidctive free energy: ', F)
    
    flf = forward_beliefs*torch.log(forward_beliefs)
    flf[torch.isnan(flf)] = 0
    H = -flf.reshape(4, -1).sum(dim=-1)
    
    flb = forward_beliefs*torch.log(backward_beliefs)
    flb[torch.isnan(flb)] = 0
    CrossH = -flb.reshape(4,-1).sum(dim=-1)
    
    print('expected free energy: ', -H+CrossH)