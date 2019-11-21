#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import time_ns as time
import jax.numpy as np
from jax import random
from jax.scipy.special import logsumexp

import numpyro as npyro

import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood, Predictive


class BayesLinRegress(object):
    def __init__(self, X, y, const=True):
        self.N, self.ns, self.nf = X.shape
        
        self.x_data = X
        
        self.y_data = y
        
        q, r = np.linalg.qr(self.x_data.reshape(-1, self.nf))
        self.Q = q.reshape(self.N, self.ns, self.nf) * self.N
        self.R_T = r.T / self.N

        self.rng_key = random.PRNGKey(time())        
        
    def model(self):
        nf = self.nf
        ns = self.ns
        
        m = npyro.sample('m', dist.Normal(np.zeros(nf), 10.))
                
        sigma = npyro.sample('sigma', dist.InverseGamma(2.*np.ones((ns,)), 1.))
        
        tau = npyro.sample('tau', dist.HalfNormal(.1*np.ones((ns,))))
        
        with npyro.plate('facts', nf):
            with npyro.plate('subs', ns):
                lam = npyro.sample('lam', dist.HalfCauchy(1.))
                var_theta = npyro.sample('var_theta', dist.Normal(0., 1.))
        
        beta = m + lam * tau.reshape(-1, 1)*var_theta
        
        theta = beta @ self.R_T
        
        mu = (theta * self.Q).sum(-1) 
        
        npyro.sample('obs', dist.Normal(mu, sigma), obs=self.y_data)

    def fit(self, num_samples = 2000, warmup_steps=2000, summary=True):
        self.rng_key, rng_key_ = random.split(self.rng_key)
        
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, warmup_steps, num_samples)
        mcmc.run(rng_key_)
        
        if summary:
            mcmc.print_summary()
        
        samples = mcmc.get_samples()
        self.mcmc = mcmc
        self.samples = samples
        
        beta = np.expand_dims(samples['m'], -2) \
            + samples['var_theta'] * samples['lam'] * np.expand_dims(samples['tau'], -1)
        
        return {'beta': beta}
    
    def post_pred_log_likelihood(self):
        
        log_lk = log_likelihood(self.model, self.samples)['obs']
        n = log_lk.shape[0]
        return (logsumexp(log_lk, 0) - np.log(n)).sum()
    
    def predictions(self):
        self.rng_key, rng_key_ = random.split(self.rng_key)
        predictive = Predictive(self.model, self.samples)
        
        return predictive.get_samples(rng_key_)['obs']