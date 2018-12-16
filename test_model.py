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
from pyro.contrib.autoguide import AutoDiagonalNormal

from pyro.infer.enum import config_enumerate

import matplotlib.pyplot as plt

import pyro

pyro.set_rng_seed(123)
torch.manual_seed(123)

n1 = 10
n2 = 3

n = 100

w = ones(n1,n2)
w[n1//2:,-1] = 1e-3
w /= w.sum(dim=-1)[:,None]

logits = dist.Normal(0, 1).sample((n, 2, 3))
depths = dist.Categorical(logits=ones(n, n1, n2)).sample()

lista = torch.tensor([range(n)]).reshape(-1, 1).repeat(1, n1)
data = dist.Categorical(logits=logits[lista, :, depths]).sample()

#def model():
#    #w = sample('group', dist.Dirichlet(ones(n1, n2)).independent(1))
#    with plate("data", n1):
#        d = sample('depth', dist.Categorical(logits=zeros(n2)))
#    print('model d = {}'.format(d))
    
#    with iarange('data', n*n1):
#        l = logits[lista.reshape(-1), :, d.reshape(-1)]
#        sample('obs', dist.Categorical(logits=l), obs = data.reshape(-1))

#init = ones(n, n1, n2)/n2
#def guide():
#    with plate("data", n1):
#        d = sample('depth', dist.Categorical(logits=zeros(n1, n2)))
#    print('guide d = {}'.format(d))

#    alphas = param('alphas', ones(n1, n2), constraint=constraints.positive)
#    sample('group', dist.Dirichlet(alphas).independent(1))

def model(data, num_blocks=90, num_subjects = 50, num_components=3):
    with plate("subjects", num_subjects):
        p = sample("p", dist.Dirichlet(ones(3) / 3.))

        scale = sample("scale", dist.Gamma(1., 1.))
        with plate("components", num_components):
            loc = sample("loc", dist.Normal(0, 10))
    
#        with plate("blocks", num_blocks):
#            x = sample("x", dist.Categorical(p))
##        print("loc.shape = {}".format(loc.shape))
##        print("scale.shape = {}".format(scale.shape))
#            sample("obs", dist.Normal(loc[x[:, ind], ind], scale), obs=data)
    for b in markov(range(num_blocks)):
        x = pyro.sample("x_{}".format(b), dist.Categorical(p), infer={"enumerate": "parallel"})
        pyro.sample("y_{}".format(b), dist.Normal(loc[x.squeeze()], scale), obs=data[b])
        print(x.shape)
        
guide = AutoDiagonalNormal(poutine.block(model, expose=["p", "scale", "loc"]))

m = 5*torch.arange(-1., 1.1, 1)
s = 0.1*torch.ones(3)
data = dist.Normal(m, s).sample((50, 30)).reshape(50, -1).transpose(dim0=1, dim1=0)    
   
clear_param_store()

num_particles = 10
n_iterations = 2000

svi = SVI(model=model,
          guide=guide,
          optim=Adam({'lr':0.1}),
          loss=TraceEnum_ELBO(num_particles=num_particles, max_plate_nesting=1))

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
        if n == 0:
            shape = smpl[name].shape
            tmp = zeros(shape).reshape(1, -1).repeat(n_samples, 1).reshape(-1, *shape)
            tmp[0] = smpl[name]
            pos_sample[name] = tmp
        else:
            pos_sample[name][n] = smpl[name]
    
plt.figure()
plt.plot(losses[-500:])

plt.figure()
for i in range(3):
    plt.hist(pos_sample['p'][:,i].detach().numpy(), bins = 100)
    
plt.figure()
for i in range(3):
    plt.hist(pos_sample['loc'][:,i].detach().numpy(), bins = 100)
    
plt.figure()
plt.hist(pos_sample['scale'].detach().numpy(), bins = 100);
