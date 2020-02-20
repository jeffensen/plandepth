#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from time import time_ns as time
import jax.numpy as np
from jax import random, vmap
from jax.scipy.special import logsumexp

import numpyro as npyro

import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood, Predictive

vdot = vmap(np.dot)
vinv = vmap(np.linalg.inv)


class BayesLinRegress(object):
    def __init__(self, X, y, const=True):
        self.ns, self.N, self.nf = X.shape

        self.x_data = X

        self.y_data = y

        q, r = vmap(np.linalg.qr)(self.x_data)
        self.Q = q * self.N
        self.R = r / self.N
        self.R_inv = vinv(self.R)

        self.rng_key = random.PRNGKey(time())

    def model(self):
        nf = self.nf
        ns = self.ns

        s = npyro.sample('s', dist.Exponential(np.ones(nf)))
        m = npyro.sample('m', dist.Normal(np.zeros(nf), 1.))

        sigma = npyro.sample('sigma', dist.InverseGamma(2.*np.ones(ns), 1.))

        tau = npyro.sample('tau', dist.Exponential(100. * np.ones(ns)))

        with npyro.plate('facts', nf):
            with npyro.plate('subs', ns):
                lam = npyro.sample('lam', dist.HalfCauchy(1.))
                var_theta = npyro.sample('var_theta', dist.Normal(0., 1.))

        gb = npyro.deterministic('group_beta', m * s)
        theta = npyro.deterministic('theta', self.R.dot(gb) + np.expand_dims(tau, -1) * lam * var_theta)

        npyro.deterministic('beta', vdot(self.R_inv, theta))

        mu = vdot(self.Q, theta)

        npyro.sample('obs', dist.Normal(mu, np.expand_dims(sigma, -1)), obs=self.y_data)

    def fit(self, num_samples=2000, warmup_steps=2000, num_chains=1, summary=True):
        self.rng_key, rng_key_ = random.split(self.rng_key)

        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, warmup_steps, num_samples, num_chains=num_chains)
        mcmc.run(rng_key_)

        if summary:
            mcmc.print_summary()

        samples = mcmc.get_samples(group_by_chain=False)
        self.mcmc = mcmc
        self.samples = samples

        return samples

    def post_pred_log_likelihood(self):

        log_lk = log_likelihood(self.model, self.samples)['obs']
        n = log_lk.shape[0]
        return (logsumexp(log_lk, 0) - np.log(n)).sum()

    def predictions(self):
        self.rng_key, rng_key_ = random.split(self.rng_key)
        predictive = Predictive(self.model, self.samples)

        return predictive.get_samples(rng_key_)['obs']