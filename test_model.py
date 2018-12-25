#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from tqdm import tqdm
import pandas as pd

import torch

ones = torch.ones
zeros = torch.zeros

softmax = torch.nn.functional.softmax

from torch.distributions import constraints, biject_to

import pyro.distributions as dist
from pyro import sample, param, markov, clear_param_store, get_param_store, plate, poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, JitTraceEnum_ELBO
from pyro.optim import Adam
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoDelta

from pyro.infer.enum import config_enumerate

import matplotlib.pyplot as plt

def model(data, num_blocks=90, num_subjects = 50, num_components=3):
    
    nu = sample("nu", dist.HalfCauchy(5.))
    mu = sample("mu", dist.Normal(0., 10.).expand([num_components]).to_event(1))
    sigma = sample("sigma", dist.HalfCauchy(1.).expand([num_components]).to_event(1))
    
    probs = sample("probs", dist.Dirichlet(ones(3)*nu).expand([num_subjects]).to_event(1))
    
    scales = sample("scales", dist.Gamma(1., 1.).expand([num_subjects]).to_event(1))
    
    locs = sample("locs", dist.Normal(0, 1).expand([num_subjects, num_components]).to_event(2))
    
    loc = locs * sigma + mu
    
    for b in markov(range(num_blocks)):
        with plate('subject_{}'.format(b), num_subjects) as ind:
            x = sample("x_{}".format(b), dist.Categorical(probs[ind]), infer={"enumerate": "parallel"})
            sample("y_{}".format(b), dist.Normal(loc[ind, x], scales[ind]), obs=data[:, b])
            
guide = AutoDiagonalNormal(poutine.block(model, expose=["nu", "mu", "sigma", "probs", "scales", "locs"]))

num_subjects = 50
num_components = 3

m = 5*torch.arange(-1., 1.1, 1)
s = 0.1*torch.ones(num_components)
data = dist.Normal(m, s).sample((50, 30)).reshape(50, -1) 

import pyro

pyro.set_rng_seed(0)
pyro.enable_validation(True)   
clear_param_store()

num_particles = 10
n_iterations = 1000

#lengths = torch.tensor([data.shape[-1]]*len(data))
#data = data.unsqueeze(dim=-1).float()

svi = SVI(model=model,
          guide=guide,
          optim=Adam({'lr':0.1}),
          loss=TraceEnum_ELBO(num_particles=num_particles, 
                              max_plate_nesting=1))

losses = []
with tqdm(total = n_iterations, file = sys.stdout) as pbar:
    for step in range(n_iterations):
            losses.append(svi.step(data))
            pbar.set_description("Mean ELBO %6.2f" % torch.Tensor(losses[-20:]).mean())
            pbar.update(1)
 
n_samples = 1000
pos_sample = {}
for n in range(n_samples):
    smpl = guide(data)
    for name in smpl:
        pos_sample.setdefault(name, [])
        pos_sample[name].append(smpl[name])

for name in pos_sample:
    pos_sample[name] = torch.stack(pos_sample[name])
    
plt.figure()
plt.plot(losses[-500:])

plt.figure()
for i in range(3):
    plt.hist(pos_sample['probs'][...,i].detach().numpy().reshape(-1), bins = 100)
    
plt.figure()
locs = pos_sample['locs'] * pos_sample['sigma'][:, None, :] + pos_sample['mu'][:, None, :]
for i in range(3):
    plt.hist(locs.detach().numpy().reshape(-1), bins = 100)
    
plt.figure()
plt.hist(pos_sample['scales'].detach().numpy().reshape(-1), bins = 100);
plt.xlim([0, 1])