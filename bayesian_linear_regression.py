#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

import pyro
from pyro.distributions import Normal, HalfCauchy, LogNormal
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

softplus = nn.Softplus()
zeros = torch.zeros
ones = torch.ones

class BayesLinRegress(object):
    def __init__(self, X, y, idx):
        self.x_data = Variable(torch.Tensor(X))
        self.y_data = Variable(torch.Tensor(y))
        
        self.N, self.f = X.shape
        
        self.n = len(np.unique(idx))
        self.idx = torch.LongTensor(idx)
        
    def model(self):
        # Group level hyper prior for sigma
        tau = pyro.sample('tau', 
                  HalfCauchy(Variable(zeros(1)),Variable(ones(1))))
        
        # Per subject model uncertainty
        sigma = pyro.sample('sigma', 
                            HalfCauchy(Variable(zeros(self.n)), tau*Variable(ones(self.n))))    
        
        # Factor level hyper prior for prior parameter uncertainty
        lam = pyro.sample('lam', 
                  HalfCauchy(Variable(zeros(self.n, self.f)), Variable(ones(self.n, self.f))))
        
        #Per subject prior uncertainty over weights
        phi = pyro.sample('phi', 
                          HalfCauchy(Variable(zeros(1)), Variable(ones(1))))
        
        # Priors over the parameters
        mu0 = Variable(zeros(self.n, self.f)) 
        sigma0 = phi*lam*sigma[:,None]
        weights = pyro.sample('weights', Normal(mu0, sigma0))
        
        
        #Prediction
        pred = (self.x_data*weights[self.idx]).sum(dim=-1)
       
        return pyro.sample('obs', Normal(pred, sigma[self.idx]))
        
    def guide(self):
        mu_t = pyro.param('mu_t', Variable(torch.randn(1), requires_grad = True))
        sigma_t = softplus(pyro.param('log_sigma_t', Variable(torch.randn(1), requires_grad = True)))
        tau = pyro.sample('tau', LogNormal(mu_t, sigma_t))

        mu_f = pyro.param('mu_f', Variable(torch.randn(1), requires_grad = True))
        sigma_f = softplus(pyro.param('log_sigma_f', Variable(torch.randn(1), requires_grad = True)))
        phi = pyro.sample('phi', LogNormal(mu_f, sigma_f))
        
        mu_s = pyro.param('mu_s', Variable(torch.randn(self.n), requires_grad = True))
        sigma_s = softplus(pyro.param('log_sigma_s', Variable(torch.randn(self.n), requires_grad = True)))
        sigma = pyro.sample('sigma', LogNormal(mu_s, sigma_s))
        
        mu_l = pyro.param('mu_l', Variable(torch.randn(self.n, self.f), requires_grad = True))
        sigma_l = softplus(pyro.param('log_sigma_l', Variable(torch.randn(self.n, self.f), requires_grad = True)))
        lam =  pyro.sample('lam', LogNormal(mu_l, sigma_l))

        mu_w = pyro.param('mu_w', Variable(torch.randn(self.n, self.f), requires_grad = True))
        sigma_w = softplus(pyro.param('log_sigma_w',  Variable(torch.randn(self.n, self.f), requires_grad = True)))
        weights =  pyro.sample('weights', Normal(mu_w, sigma_w))
        
        return weights, sigma, phi, lam, tau
    
    def fit(self, n_iterations = 1000):
        pyro.clear_param_store()
        condition = pyro.condition(self.model, data = {'obs': self.y_data.squeeze()})
        svi = SVI(model=condition,
                  guide=self.guide,
                  optim=Adam({"lr": 0.01}),
                  loss=Trace_ELBO())
        losses = []
        for step in tqdm(range(n_iterations)):
            loss = svi.step()
            if step % 100 == 0:
                losses.append(loss)
                
        mu_w = pyro.param('mu_w').data.numpy()
        sigma_w = softplus(pyro.param('log_sigma_w')).data.numpy()
        mu_s = pyro.param('mu_s').data.numpy()
        sigma_s = softplus(pyro.param('log_sigma_s')).data.numpy()
        mu_t = pyro.param('mu_t').data.numpy()
        sigma_t = softplus(pyro.param('log_sigma_t')).data.numpy()

        mu_l = pyro.param('mu_l').data.numpy()
        sigma_l = softplus(pyro.param('log_sigma_l')).data.numpy()
        mu_f = pyro.param('mu_f').data.numpy()
        sigma_f = softplus(pyro.param('log_sigma_f')).data.numpy()
        
        return mu_t, sigma_t, mu_f, sigma_f, mu_s, sigma_s, mu_l, sigma_l, mu_w, sigma_w