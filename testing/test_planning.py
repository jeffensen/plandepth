#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tqdm import tqdm

import numpy as np

import torch
from torch.distributions import Categorical, constraints
from torch.autograd import Variable

from helpers import make_transition_matrix

zeros = torch.zeros
ones = torch.ones
var = Variable
randn = torch.randn
softplus = torch.nn.Softplus()

na = 2
ns = 6

# load planet cofigurations and starting positions
vect = np.load('confsExp1.npy')
ol_exp1 = torch.from_numpy(vect)
ol_exp2 = torch.from_numpy(np.vstack([vect[50:], vect[:50]]))

vect = np.load('startsExp1.npy')
starts_exp1 = torch.from_numpy(vect)
starts_exp2 = torch.from_numpy(np.hstack([vect[50:], vect[:50]]))

# state transition matrix c, a, s -> s'
p = [.9, .5]
stm_low = make_transition_matrix(p[0], ns, na)
stm_high = make_transition_matrix(p[1], ns, na)
stm = torch.stack([stm_low, stm_high])

# planning depths
pd_exp1 = torch.tensor([2, 3]).repeat(50, 1)\
    .transpose(dim0=1, dim1=0).contiguous().view(-1)
pd_exp2 = torch.stack([pd_exp1[50:], pd_exp1[:50]]).view(-1)

action_costs = torch.tensor([-.2, -.5]).repeat(2, 1)  # action costs

values = ol_exp1.argmax(dim=-1).float() - 2.

N = 100  # number of mini-blocks
optimal_depth = torch.zeros(N, 2, dtype=torch.long)
for i in range(N):

    start = starts_exp1[i]

    max_depth = pd_exp1[i]
    log_pd = 0.693*torch.arange(1., max_depth+1.)

    Q = torch.zeros(2, na, ns, max_depth)
    R = torch.einsum('ijkl,l->ijk', (stm, values[i]))

    Q[..., 0] = R + action_costs[..., None]        
    for depth in range(1, max_depth):
        # compute response probability
        p = 1/(1+torch.exp((Q[:, 0, :, depth-1] - Q[:, 1, :, depth-1])/1e-10))

        # set state value
        V = p*Q[:, 1, :, depth-1] + (1-p)*Q[:, 0, :, depth-1]

        Q[..., depth] = torch.einsum('ijkl,il->ijk', (stm, V))
        Q[..., depth] += R + action_costs[..., None]

    qa = torch.ones(2, ns, na)/na
    qd = torch.ones(2, ns, max_depth)/max_depth.item()

    for _ in range(10):
        log_qa = torch.einsum('ikl,ijkl->ikj', (qd, Q))
        log_qd = torch.einsum('ijk,ikjl->ijl', (qa, Q))
    
        qa = torch.exp(log_qa)
        qa /= qa.sum(dim=-1, keepdim=True)
        qd = torch.exp(log_qd)
        qd /= qd.sum(dim=-1, keepdim=True)
    
    optimal_depth[i] = qd[:, start].argmax(dim=-1)+1



# import seaborn as sns
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(2, 2, figsize=(10, 6))
#
# sns.heatmap(data=qa[0], ax=ax[0,0], cmap='viridis')
# sns.heatmap(data=qa[1], ax=ax[0,1], cmap='viridis')
#
# sns.heatmap(data=qd[0], ax=ax[1,0], cmap='viridis')
# sns.heatmap(data=qd[1], ax=ax[1,1], cmap='viridis')

