#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Here we will test the inference procedure for the linear regression model. Use the same predictors as in the 
analysis of the behavioural data with a set of known regression coefficients to thest if we can 
infer the correct (non-zero) coefficients.
"""
import sys
from tqdm import tqdm
import pandas as pd

import torch

ones = torch.ones
zeros = torch.zeros

softmax = torch.nn.functional.softmax

from torch.distributions import constraints, biject_to

import pyro.distributions as dist
from pyro import sample, param, markov, clear_param_store, get_param_store, plate, poutine, iarange
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, JitTraceEnum_ELBO
from pyro.optim import Adam
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoDelta, AutoGuideList, AutoMultivariateNormal

from pyro.infer.enum import config_enumerate

import matplotlib.pyplot as plt
import seaborn as sns

sigma = dist.HalfCauchy(1.).expand([3]).sample()
mu = dist.Normal(0., 10.).expand([3]).sample()
locs = dist.Normal(0, 1).expand([10, 3]).sample()

loc = locs * sigma + mu

data = dist.Normal(loc.sum(-1), 1.).sample((100,))

def model(data, num_components=3, num_subjects=10):
    
    N = len(data)
    
    mu = sample("mu", dist.Normal(0., 10.).expand([num_components]).to_event(1))

    
    components = plate('components', num_components)
    subjects = plate("subjects", num_subjects)
    
    with components:
        sigma = sample("sigma", dist.HalfCauchy(1.))
    
        with subjects:
            locs = sample("locs", dist.Normal(0., 1.))
            assert locs.shape == (num_subjects, num_components)
    
    loc = (locs * sigma + mu).sum(-1)
    assert loc.shape == (num_subjects,)
    
    with plate('list', num_subjects) as ind:
        m = loc[ind]
        with plate('data', N):
            sample("y", dist.Normal(m, 1.), obs=data)


def guide(data, num_components=3, num_subjects=10):
    
    m_sigma = param('m_sigma', zeros(num_components))
    s_sigma = param('s_sigma', ones(num_components), constraint=constraints.positive)
    
    m_mu = param('m_mu', zeros(num_components))
    s_mu = param('s_mu', torch.eye(num_components), constraint=constraints.lower_cholesky)
    
    m_locs = param('m_locs', zeros(num_subjects, num_components))
    s_locs = param('s_locs', ones(num_subjects, num_components), constraint=constraints.positive)
    
    mu = sample("mu", dist.MultivariateNormal(m_mu, scale_tril=s_mu))

    with plate('components', num_components, dim=-1):
        sigma = sample("sigma", dist.LogNormal(m_sigma, s_sigma))
    
        with plate('subjects', num_subjects, dim=-2):
            locs = sample("locs", dist.Normal(m_locs, s_locs))
    
    return {"locs": locs, "mu": mu, "sigma": sigma}

import pyro

pyro.set_rng_seed(123)
pyro.enable_validation(True)   
clear_param_store()

num_particles = 20
n_iterations = 1000

svi = SVI(model=model,
          guide=guide,
          optim=Adam({'lr':0.1}),
          loss=Trace_ELBO(num_particles=num_particles, max_plate_nesting=3))

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
    
    pos_sample.setdefault('loc', [])
    pos_sample['loc'].append(pos_sample['locs'][-1] * pos_sample['sigma'][-1] + pos_sample['mu'][-1])

for name in pos_sample:
    pos_sample[name] = torch.stack(pos_sample[name])
    
plt.figure()
plt.plot(losses[-500:])

post_loc = pos_sample['loc']
plt.figure()
for i in range(3):
    plt.hist(post_loc[...,i].detach().numpy().reshape(-1), bins = 100)
    plt.vlines(loc[..., i], 0, 50, colors='k', linestyles='--')
    
plt.figure()
for i in range(3):
    plt.hist(post_loc.sum(-1).detach().numpy(), bins = 100)
    plt.vlines(loc.sum(-1), 0, 50, colors='k', linestyles='--')

mg_df = pd.DataFrame(data=pos_sample['mu'].detach().numpy())

g = sns.PairGrid(mg_df)
g = g.map_diag(sns.kdeplot)
g = g.map_offdiag(plt.scatter)
    
#plt.figure()
#locs = pos_sample['locs'] * pos_sample['sigma'][:, None, :] + pos_sample['mu'][:, None, :]
#for i in range(3):
#    plt.hist(locs.detach().numpy().reshape(-1), bins = 100)
#    
#plt.figure()
#plt.hist(pos_sample['scales'].detach().numpy().reshape(-1), bins = 100);
#plt.xlim([0, 1])