#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from tqdm import tqdm
import pandas as pd

import torch

ones = torch.ones
zeros = torch.zeros

softmax = torch.nn.functional.softmax

from torch.distributions import constraints

import pyro.distributions as dist
from pyro import sample, param, iarange, irange, clear_param_store, get_param_store
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.optim import Adam

from pyro.infer.enum import config_enumerate

import matplotlib.pyplot as plt

n1 = 10
n2 = 3

n = 100

w = ones(n1,n2)
w[n1//2:,-1] = 1e-3
w /= w.sum(dim=-1)[:,None]

logits = dist.Normal(0, 1).sample((n, 2, 3))
depths = dist.Categorical(logits=ones(n, n1, n2)).sample()
data = dist.Categorical(logits=logits[range(n), :, depths]).sample()

def model():
    w = sample('group', dist.Dirichlet(ones(n1, n2)).independent(1))
    d = sample('depth', dist.Categorical(w).independent(1))
    
    sample('obs', dist.Categorical(probs = logits[...,d]).independent(1), obs = data)

init = ones(n1,n2)
init[n1//2:,-1] = 1e-3
init /= init.sum(dim=-1)[:,None]
def guide():
    probs = param('probs', init, constraint=constraints.simplex)
    sample('depth', dist.Categorical(probs=probs).independent(1))
    
    alphas = param('alphas', ones(n1, n2), constraint=constraints.positive)
    sample('group', dist.Dirichlet(alphas).independent(1))
    
   
clear_param_store()

num_particles = 10
n_iterations = 5000

svi = SVI(model=model,
          guide=guide,
          optim=Adam({'lr':0.1}),
          loss=TraceEnum_ELBO(num_particles=num_particles, max_iarange_nesting=1))

losses = []
with tqdm(total = n_iterations, file = sys.stdout) as pbar:
    for step in range(n_iterations):
            losses.append(svi.step())
            pbar.set_description("Mean ELBO %6.2f" % torch.Tensor(losses[-20:]).mean())
            pbar.update(1)
    
results = {}
for name, value in get_param_store().named_parameters():
    results[name] = value
    
plt.plot(losses)