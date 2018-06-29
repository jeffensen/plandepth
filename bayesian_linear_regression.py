#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm

import numpy as np

import torch
import torch.distributions.constraints as constraints 
positive = constraints.positive

import pyro
from pyro.distributions import Normal, HalfCauchy, LogNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

zeros = torch.zeros
ones = torch.ones

class BayesLinRegress(object):
    def __init__(self, X, y, idx):
        self.x_data = torch.Tensor(X)
        self.y_data = torch.Tensor(y).squeeze()
        
        self.N, self.f = X.shape
        
        self.loss = []
        self.n = len(np.unique(idx))
        self.idx = torch.LongTensor(idx)
        
    def model(self):
        # Group level hyper prior for sigma
        rho = pyro.sample('rho', HalfCauchy(zeros(1),ones(1)))
        
        # Per subject model uncertainty
        sigma = pyro.sample('sigma', HalfCauchy(zeros(self.n), rho*ones(self.n)).independent(1))    
        
        # Factor level hyper prior for prior parameter uncertainty
        lam = pyro.sample('lam', HalfCauchy(zeros(self.n, self.f), ones(self.n, self.f)).independent(2))
        
        #Per subject prior uncertainty over weights
        phi = pyro.sample('phi', HalfCauchy(zeros(self.f), ones(self.f)).independent(1))
        
        # Priors over the parameters
        mu0 = zeros(self.n, self.f) 
        sigma0 = phi[None,:]*lam*sigma[:,None]
        weights = pyro.sample('weights', Normal(mu0, sigma0).independent(2))
        
        
        #Prediction
        obs = self.y_data
        with pyro.iarange('map', len(obs)):
            pred = (weights[self.idx]*self.x_data).sum(dim=-1)
            pyro.sample('obs', Normal(pred, sigma[self.idx]), obs = obs)
        
    
    def guide(self):
        mu_r = pyro.param('mu_r', zeros(1,requires_grad = True))
        sigma_r = pyro.param('sigma_r', ones(1,requires_grad = True), constraint=positive)
        pyro.sample('rho', LogNormal(mu_r, sigma_r))

        mu_f = pyro.param('mu_f', zeros(self.f, requires_grad = True))
        sigma_f = pyro.param('sigma_f', ones(self.f,requires_grad = True), constraint=positive)
        pyro.sample('phi', LogNormal(mu_f, sigma_f).independent(1))
        
        mu_s = pyro.param('mu_s', zeros(self.n,requires_grad = True))
        sigma_s = pyro.param('sigma_s', ones(self.n,requires_grad = True), constraint=positive)
        pyro.sample('sigma', LogNormal(mu_s, sigma_s).independent(1))
        
        mu_l = pyro.param('mu_l', zeros(self.n, self.f,requires_grad = True))
        sigma_l = pyro.param('sigma_l', ones(self.n, self.f,requires_grad = True), constraint=positive)
        pyro.sample('lam', LogNormal(mu_l, sigma_l).independent(2))

        mu_w = pyro.param('mu_w', zeros(self.n, self.f, requires_grad = True))
        sigma_w = pyro.param('sigma_w',  ones(self.n, self.f,requires_grad = True), constraint=positive)
        pyro.sample('weights', Normal(mu_w, sigma_w).independent(2))
        
    
    def fit(self, n_iterations = 1000):
        pyro.clear_param_store()
        svi = SVI(model=self.model,
                  guide=self.guide,
                  optim=Adam({"lr": 0.01}),
                  loss=Trace_ELBO())

        for step in tqdm(range(n_iterations)):
            self.loss.append(svi.step())
        
        results = {}
        results['mean_beta'] = pyro.param('mu_w').data.numpy()
        results['sigma_beta'] = pyro.param('sigma_w').data.numpy()
        results['mean_std'] = pyro.param('mu_s').data.numpy()
        results['sigma_std'] = pyro.param('sigma_s').data.numpy()
        results['mean_rho'] = pyro.param('mu_r').data.numpy()
        results['sigma_rho'] = pyro.param('sigma_r').data.numpy()
        results['mean_lam'] = pyro.param('mu_l').data.numpy()
        results['sigma_lam'] = pyro.param('sigma_l').data.numpy()
        results['mean_phi'] = pyro.param('mu_f').data.numpy()
        results['sigma_phi'] = pyro.param('sigma_f').data.numpy()
        
        return results