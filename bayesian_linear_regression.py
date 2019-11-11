#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch

import pyro
from pyro.distributions import Normal, HalfCauchy
from pyro.infer.mcmc import MCMC, NUTS

class BayesLinRegress(object):
    def __init__(self, X, y, idx):
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(y).squeeze()
        
        self.f, self.N = X.shape
        
        self.idx = torch.from_numpy(idx - 1)
        self.n = len(torch.unique(self.idx))

    def model(self):
        
        rho = pyro.sample('rho', HalfCauchy(1.))
        
        # Per subject model uncertainty
        with pyro.plate('subjects', self.n):
            sigma = pyro.sample('sigma', HalfCauchy(1))
            with pyro.plate('factors', self.f):
                # Factor level hyper prior for prior parameter uncertainty
                lam = pyro.sample('lam', HalfCauchy(1.))
        
                # Priors over the parameters
                weights = pyro.sample('weights', Normal(0., 1.))
        
        # Prediction
        pred = ((rho*lam*weights)[:, self.idx]*self.x_data).sum(-2)
           
        # Observation likelihood
        with pyro.plate('data', self.N):
            pyro.sample('obs', Normal(pred, sigma[self.idx]), obs=self.y_data)
        
    
    def fit(self, num_samples = 1500, num_chains=4, warmup_steps=500):
        pyro.clear_param_store()
        nuts_kernel = NUTS(self.model, adapt_step_size=True, jit_compile=True, ignore_jit_warnings=True)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_chains=num_chains, warmup_steps=warmup_steps)
        mcmc.run()
        samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
        
        return samples