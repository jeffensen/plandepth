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
from pyro import sample, param, plate, markov, poutine, clear_param_store, get_param_store
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
from pyro.optim import Adam
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoDelta


class Inferrer(object):
    def __init__(self,
                 agent,
                 stimuli,
                 responses,
                 mask):
        
        self.agent = agent
        self.nsub, self.nblk = responses.shape[:2] 
        
        self.responses = responses
        self.mask = mask
        self.N = mask.sum(dim=0)
        
        self.states = stimuli['states']
        self.configs = stimuli['configs']
        self.conditions = stimuli['conditions']
    
    def model(self):
        
        agent = self.agent
        np = agent.np  # number of parameters
        
        nblk =  self.nblk  # number of mini-blocks
        nsub = self.nsub  # number of subjects

        mu = sample("mu", dist.Normal(0., 10.).expand([np]).to_event(1))
        sigma = sample("sigma", dist.HalfCauchy(5.).expand([np]).to_event(1))
        
        probs_prior = torch.tril(ones(3, 3))
#        nuc1 = sample("nu_c1", dist.HalfCauchy(5.))
#        probsc1 = sample("probs_c1", dist.Dirichlet(ones(2)*nuc1).expand([nsub]).to_event(1))
#        
#        nuc2 = sample("nu_c2", dist.HalfCauchy(5.))
#        probsc2 = sample("probs_c2", dist.Dirichlet(ones(3)*nuc2).expand([nsub]).to_event(1))

        locs = sample("locs", dist.Normal(mu, sigma).expand([nsub, np]).to_event(2))
    
        trans_pars = locs
        
        agent.set_parameters(trans_pars)
        
        for b in markov(range(nblk)):
            conditions = self.conditions[..., b]
            states = self.states[:, b]
            responses = self.responses[:, b]
            for t in markov(range(3)):
                if t == 0:
                    res = None
                else:
                    res = responses[:, t-1]
                
                agent.update_beliefs(b, t, states[:, t], conditions, res)
                agent.plan_actions(b, t)
                
                valid = self.mask[:, b, t]
                N = self.N[b, t]

                logits = agent.logits[-1][:, valid]
                res = responses[valid, t]
                
                max_trials = conditions[-1, valid]
                probs = probs_prior[max_trials-t-1]

                with plate('responses_{}_{}'.format(b, t), N) as ind:
                    d = sample('d_{}_{}'.format(b, t),
                               dist.Categorical(probs[ind]),
                               infer={"enumerate": "parallel"})
                    
                    sample('obs_{}_{}'.format(b, t), 
                           dist.Bernoulli(logits=logits[d, ind]),
                           obs=res[ind])
    
    def fit(self, num_iterations = 100, num_particles=10, optim_kwargs={'lr':.1}):
        
        model = self.model
        guide = AutoDiagonalNormal(poutine.block(model, 
                                                 expose=["mu",
                                                         "sigma",
                                                         "locs"]))
        
        clear_param_store()
        
        svi = SVI(model=model,
                  guide=guide,
                  optim=Adam(optim_kwargs),
                  loss=TraceEnum_ELBO(num_particles=num_particles,
                                      max_plate_nesting=1))

        loss = []
        pbar = tqdm(range(num_iterations), position=0)
        for step in pbar:
            loss.append(svi.step())
            pbar.set_description("Mean ELBO %6.2f" % torch.Tensor(loss[-20:]).mean())
        
        self.mean = param('auto_loc')
        self.std = param('auto_scale')
        self.guide = guide        
        self.loss = loss
        
    def sample_from_posterior(self, labels, centered=False, n_samples=10000):
        
        import numpy as np
        nsub = self.nsub
        npars = self.agent.np
        assert npars == len(labels)
        
        keys = ['mu', 'sigma', 'locs']
        
        trans_pars = np.zeros((n_samples, nsub, npars))
        
        mean_group = np.zeros((n_samples, npars))
        sigma_group = np.zeros((n_samples, npars))
        
        for i in range(n_samples):
            sample = self.guide()
            for key in keys:
                sample.setdefault(key, ones(1))
                
            mu = sample['mu']
            sigma = sample['sigma']
            
            if centered:
                pars = sample['locs']
            else:
                pars = sample['locs'] * sigma + mu
            
            trans_pars[i] = pars.detach().numpy()
            
            mean_group[i] = mu.detach().numpy()
            sigma_group[i] = sigma.detach().numpy()
        
        subject_label = np.tile(range(1, nsub+1), (n_samples, 1)).reshape(-1)
        tp_df = pd.DataFrame(data=trans_pars.reshape(-1, npars), columns=labels)
        tp_df['subject'] = subject_label
        
        mg_df = pd.DataFrame(data=mean_group, columns=labels)
        sg_df = pd.DataFrame(data=sigma_group, columns=labels)
        
        return (tp_df, mg_df, sg_df)