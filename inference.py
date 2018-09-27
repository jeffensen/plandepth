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
        self.N = mask.sum().item()
        
        self.trials = stimuli['trials']
        self.states = stimuli['states']
        self.configs = stimuli['config']
        self.scores = stimuli['scores']
        self.conditions = stimuli['conditions']
    
    def model(self, depth):
        
        n = self.n #number of subjects
        agent = self.agent
        npars = agent.npars #number of parameters
        
        #prior uncertainty
        sig0 = sample('sig0', dist.HalfCauchy(zeros(n,npars), ones(n,npars)). independent(2))
        x = sample('x', dist.Normal(zeros(n,npars), sig0).independent(2))
        
        agent.set_params(x)
        
        for b in range(self.blocks):
            trials = self.trials[b]
            configs = self.configs[b]
            conditions = self.conditions[b]
            
            states = self.states[b]
            scores = self.scores[b]

            responses = self.responses[b]
            
            max_trial = trials.max().item()
            context = [trials, configs, conditions, states[:,0]]
            agent.set_context(context, max_trial)
            
            for t in range(max_trial):
                agent.plan_behavior(b, t, depth)
                agent.update_beliefs(t, states[:,t+1], scores[:,t], responses[t])
            
        responses = self.responses
        mask = self.mask
        sample('obs', 
               dist.Categorical(probs = agent.probs[mask]).independent(1), 
               obs = responses[mask])

    def guide(self, depth):
        n = self.n #number of subjects
        npars = self.agent.npars #number of parameters
        
        mu_sig0 = param('mu_sig0', zeros(n,npars))
        sig_sig0 = param('sig_sig0', ones(n, npars), constraint=constraints.positive)
        sample('sig0', dist.LogNormal(mu_sig0, sig_sig0).independent(2))
        
        mu_x = param('mu_x', zeros(n,npars))
        sig_x = param('sig_x', ones(n,npars), constraint=constraints.positive)
        sample('x', dist.Normal(mu_x, sig_x).independent(2))        
        
    
    def fit(self, depth, n_iterations = 1000, num_particles = 10, progressbar = True, optim_kwargs = {'lr':0.1}):
        
        clear_param_store()

        svi = SVI(model=self.model,
                  guide=self.guide,
                  optim=Adam(optim_kwargs),
                  loss=Trace_ELBO(num_particles=num_particles))
        
        losses = []
        if progressbar:
            with tqdm(total = n_iterations, file = sys.stdout) as pbar:
                for step in range(n_iterations):
                    losses.append(svi.step(depth))
                    pbar.set_description("Mean ELBO %6.2f" % torch.Tensor(losses[-20:]).mean())
                    pbar.update(1)
        else:
            for step in range(n_iterations):
                    losses.append(svi.step())
        
        self.losses = losses
            
    def get_posteriors(self):
        
        results = {}
        for name, value in get_param_store().named_parameters():
            if name == 'probs':
                results[name] = softmax(value, dim=-1)
            else:
                if name.find('sig_') == -1:
                    results[name] = value
                else:
                    results[name] = torch.exp(value)
        
        return results