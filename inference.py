#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from tqdm import tqdm
import pandas as pd

import torch

ones = torch.ones
zeros = torch.zeros

from torch.distributions import constraints

import pyro.distributions as dist
from pyro import sample, param, iarange, irange, clear_param_store, get_param_store
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.optim import Adam

class Inferrer(object):
    def __init__(self,
                 agent,
                 stimuli,
                 responses,
                 mask):
        
        self.agent = agent
        self.blocks, self.n = responses.shape[:2] 
        
        self.responses = responses
        self.mask = mask
        
        self.trials = stimuli['trials']
        self.states = stimuli['states']
        self.configs = stimuli['config']
        self.scores = stimuli['scores']
        self.conditions = stimuli['conditions']
    
    def model(self):
        
        n = self.n #number of subjects
        agent = self.agent
        npars = agent.npars #number of parameters
        
        #hyperprior group
        mu0 = sample('mu0', dist.Cauchy(zeros(npars), ones(npars)).independent(1))
        
        #prior uncertainty
        sig0 = sample('sig0', dist.HalfCauchy(zeros(n,npars), ones(n,npars)). independent(2))
        x = sample('x', dist.Normal(mu0*ones(n,npars), sig0).independent(2))
        
        w = ones(self.blocks, n, 3)
        w[~self.mask] = 0.
        w /= w.sum(dim=-1)[:,:,None]
        #weights = param('weights', w, constraint = constraints.simplex)
        depth = sample('depth', dist.Categorical(w))
        
        agent.set_params(x)
        for b in range(self.blocks):
            trials = self.trials[b].clone()
            configs = self.configs[b]
            conditions = self.conditions[b]
            
            states = self.states[b]
            scores = self.scores[b]
            responses = self.responses[b]
            mask = self.mask[b]
            
            max_trial = trials.max().item()
            context = [trials, configs, conditions, states[:,0]]
            agent.set_context(context, max_trial)
            
            for t in range(max_trial):
                agent.plan_behavior(t, depth[b])
                agent.update_beliefs(t, states[:,t+1], scores[:,t], responses[t])

            with iarange('obs_loop'):
                sample('obs_{}'.format(b), 
                       dist.Categorical(probs = agent.probs[mask]), 
                       obs = responses[mask])

    def guide(self):
        n = self.n #number of subjects
        b = self.blocks #number of mini-blocks
        npars = self.agent.npars #number of parameters
        
        mu_mu0 = param('mu_mu0', zeros(npars))
        sig_mu0 = param('sig_mu0', ones(npars), constraint=constraints.positive)        
        sample('mu0', dist.Normal(mu_mu0, sig_mu0).independent(1))
        
        mu_sig0 = param('mu_sig0', zeros(n,npars))
        sig_sig0 = param('sig_sig0', ones(n, npars), constraint=constraints.positive)
        sample('sig0', dist.LogNormal(mu_sig0, sig_sig0).independent(2))
        
        mu_x = param('mu_x', zeros(n,npars))
        sig_x = param('sig_x', ones(n,npars), constraint=constraints.positive)
        sample('x', dist.Normal(mu_x, sig_x).independent(2))
        
        probs = param('probs', ones(b, n, 3)/3, constraint=constraints.simplex)
        sample('depth', dist.Categorical(probs))
    
    def fit(self, n_iterations = 1000, num_particles = 10, progressbar = True, optim_kwargs = {'lr':0.1}):
        
        clear_param_store()

        svi = SVI(model=self.model,
                  guide=self.guide,
                  optim=Adam(optim_kwargs),
                  loss=Trace_ELBO(num_particles=num_particles))
        
        losses = []
        if progressbar:
            with tqdm(total = n_iterations, file = sys.stdout) as pbar:
                for step in range(n_iterations):
                    losses.append(svi.step())
                    pbar.set_description("Mean ELBO %6.2f" % torch.Tensor(losses[-20:]).mean())
                    pbar.update(1)
        else:
            for step in range(n_iterations):
                    losses.append(svi.step())
            
    def get_posteriors(self):
        
        results = {}
        for name, value in get_param_store().named_parameters():
            results[name] = value.data.numpy()
        
        return results