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
            config = self.config[b]
            responses = self.responses[b]
            for t in range(trials.max().item()):
                trials -= t
                outcomes = [trials, states[t], config]
                agent.update_beliefs(outcomes, responses[t])
                agent.plan_behavior(depth[b])

        
        return pyro.sample('responses', dist.categorical, p)
    
    def guide(self):
        #hyperprior across subjects
        mu_t = pyro.param('mu_t', var(zeros(1), requires_grad = True))
        sigma_t = softplus(pyro.param('log_sigma_t', var(ones(1), requires_grad = True)))
        tau = pyro.sample('tau', dist.lognormal, mu_t, sigma_t)

        
        #subject specific hyper prior
        mu_l = pyro.param('mu_l', var(zeros(self.runs), requires_grad = True))
        sigma_l = softplus(pyro.param('log_sigma_l', var(ones(self.runs), requires_grad = True)))
        lam = pyro.sample('lam', dist.lognormal, mu_l, sigma_l)
        
        #subject specific response probability
        alphas = softplus(pyro.param('log_alphas', var(ones(self.runs, self.na), requires_grad = True)))
        p = pyro.sample('p', dist.dirichlet, alphas)
        
        return tau, lam, p
    
    def fit(self, n_iterations = 1000, progressbar = True):
        if progressbar:
            xrange = tqdm(range(n_iterations))
        else:
            xrange = range(n_iterations)
        
        pyro.clear_param_store()
        data = self.responses[self.notnans]
        condition = pyro.condition(self.model, data = {'responses': data})
        svi = SVI(model=condition,
                  guide=self.guide,
                  optim=Adam({"lr": 0.01}),
                  loss="ELBO")
        losses = []
        for step in xrange:
            loss = svi.step()
            if step % 100 == 0:
                losses.append(loss)
        
        fig = plt.figure()        
        plt.plot(losses)
        fig.suptitle('ELBO convergence')
        plt.xlabel('iteration')
        
        param_names = pyro.get_param_store().get_all_param_names();
        results = {}
        for name in param_names:
            if name[:3] == 'log':
                results[name[4:]] = softplus(pyro.param(name)).data.numpy()
            else:
                results[name] = pyro.param(name).data.numpy()
        
        return results