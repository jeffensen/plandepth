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

#state transition matrix c, a, s -> s'
p = [.9, .5];
stm_forward_low = make_transition_matrix(p[0], ns, na)
stm_forward_high = make_transition_matrix(p[1], ns, na)
stm_forward = torch.stack([stm_forward_low, stm_forward_high])
stm_backward = stm_forward.transpose(dim0 = -1, dim1 = -2).clone()
stm_backward /= stm_backward.sum(dim=-1, keepdim=True)

#for plotting
colors = np.array(['red', 'pink', 'gray', 'blue', 'green'])
    
#load single planetary sistem
#vect = np.load('confsExp1.npy')
#outcome_likelihood = vect[0].astype(int)

outcomes_likelihood = np.array([[1.,0,0.,0,0],[0,1.,0.,0,0], [0,0,1.,0,0], [0,0,0.,1.,0], [0,0,0,0,1.], [0,0,0,0,1.]])

#load starting condition
#vect = np.load('startsExp1.npy')
#starts = vect[0]

starts = 2

policies1 = itype([[0],[1]])
policies2 = itype([[0,0],[0,1],[1,0],[1,1]])
policies3 = itype([[0,0,0], 
                   [0,0,1],[0,1,0],[1,0,0],
                   [0,1,1],[1,0,1],[1,1,0],
                   [1,1,1]])

policies = policies2
npi = len(policies)

p0 = zeros(ns)
p0[starts] = 1

values = ftype(outcomes_likelihood.argmax(axis = -1)- 2.)
pT = torch.exp(values-values.max())
pT /= pT.sum() # preference over states

max_depth = policies.shape[-1]+1

for depth in range(1,max_depth):
    forward_beliefs = zeros(2,npi, ns, depth+1)
    forward_beliefs[...,0] = p0[None,None,:]
    backward_beliefs = zeros(2,npi, ns, depth+1)
    backward_beliefs[...,-1] = pT[None,None,:]
    for tau in range(1,depth+1):
        a = policies[:,tau-1]
        forward_beliefs[...,tau] = torch.sum(stm_forward[:,a]*forward_beliefs[...,tau-1,None], -2)
        backward_beliefs[...,-tau-1] = torch.sum(stm_backward[:,a]*backward_beliefs[...,-tau, None], -2)

    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, npi, figsize = (16, 8), sharex = True, sharey=True)

    for i in range(npi):
        sns.heatmap(data = forward_beliefs[0,i], ax = axes[0,i], vmax = 1, vmin = 0, cmap = 'viridis', cbar = False)
        sns.heatmap(data = backward_beliefs[0,i], ax = axes[1,i], vmax = 1, vmin = 0, cmap = 'viridis', cbar = False)
        sns.heatmap(data = forward_beliefs[1,i], ax = axes[2,i], vmax = 1, vmin = 0, cmap = 'viridis', cbar = False)
        sns.heatmap(data = backward_beliefs[1,i], ax = axes[3,i], vmax = 1, vmin = 0, cmap = 'viridis', cbar = False)


    logp = torch.log(forward_beliefs)/2 + torch.log(backward_beliefs)/2

    q = torch.exp(logp)
    q /= q.sum(dim=-2, keepdim=True)
#
#    qlq = q*torch.log(q)
#    qlq[torch.isnan(qlq)] = 0
#    H = -qlq.reshape(4,-1).sum(dim=-1)
#
#    qlp = q*logp
#    qlp[torch.isnan(qlp)] = 0
#    U = -qlp.reshape(4,-1).sum(dim=-1)
#
#    F = U - H
#    print('preidctive free energy: ', F)
#    
#    flf = forward_beliefs*torch.log(forward_beliefs)
#    flf[torch.isnan(flf)] = 0
#    H = -flf.reshape(4, -1).sum(dim=-1)
#    
#    flb = forward_beliefs*torch.log(backward_beliefs)
#    flb[torch.isnan(flb)] = 0
#    CrossH = -flb.reshape(4,-1).sum(dim=-1)
#    
#    print('expected free energy: ', -H+CrossH)
    

max_depth = 3

log_pd = -torch.log(torch.ones(1)*2)*torch.arange(1.,4.)

action_costs = torch.FloatTensor([-.2, -.5]).repeat(2,1) #action costs
tm = stm_forward

Q = torch.zeros(2, na, ns, max_depth)
R = torch.einsum('ijkl,l->ijk', (tm, values))

Q[...,0] = R + action_costs[..., None]        
for depth in range(1,max_depth):
    #compute response probability
    p = 1/(1+torch.exp((Q[:, 0, :, depth-1] - Q[:, 1, :, depth-1])/1e-10))
    
    #set state value
    V = p*Q[:, 1, :, depth-1] + (1-p)*Q[:, 0, :, depth-1]
    
    Q[...,depth] = torch.einsum('ijkl,il->ijk', (tm, V))
    Q[...,depth] += R + action_costs[..., None]

qa = torch.ones(2, ns, na)/na
qd = torch.ones(2, ns, max_depth)/max_depth

for i in range(10):
    log_qa = torch.einsum('ikl,ijkl->ikj', (qd, Q))
    log_qd = torch.einsum('ijk,ikjl->ijl', (qa, Q)) + log_pd
    
    qa = torch.exp(log_qa)
    qa /= qa.sum(dim=-1, keepdim=True)
    qd = torch.exp(log_qd)
    qd /= qd.sum(dim=-1, keepdim=True)

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(10, 6))

sns.heatmap(data=qa[0], ax=ax[0,0], cmap='viridis')
sns.heatmap(data=qa[1], ax=ax[0,1], cmap='viridis')

sns.heatmap(data=qd[0], ax=ax[1,0], cmap='viridis')
sns.heatmap(data=qd[1], ax=ax[1,1], cmap='viridis')

