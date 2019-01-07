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
from pyro.contrib.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoGuideList, AutoIAFNormal


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
    
    def centered_model(self):
        
        agent = self.agent
        np = agent.np  # number of parameters
        
        nblk =  self.nblk  # number of mini-blocks
        nsub = self.nsub  # number of subjects

        mu = sample("mu", dist.Cauchy(0., 1.).expand([np]).to_event(1))
        sigma = sample("sigma", dist.HalfCauchy(1.).expand([np]).to_event(1))
        
#        probs_prior = torch.tril(ones(3, 3))
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
                depth = max_trials - t - 1
                #probs = probs_prior[max_trials-t-1]

                with plate('responses_{}_{}'.format(b, t), N) as ind:
#                    d = sample('d_{}_{}'.format(b, t),
#                               dist.Categorical(probs[ind]),
#                               infer={"enumerate": "parallel"})
#                   
                    d = depth[ind]
                    sample('obs_{}_{}'.format(b, t), 
                           dist.Bernoulli(logits=logits[d, ind]),
                           obs=res[ind])

    def non_centered_model(self):
        
        agent = self.agent
        np = agent.np  # number of parameters
        
        nblk =  self.nblk  # number of mini-blocks
        nsub = self.nsub  # number of subjects

        mu = sample("mu", dist.Normal(0., 20.).expand([np]).to_event(1))
        
        sigma_g = sample("sigma_g", dist.HalfCauchy(1.).expand([np]).to_event(1))
        sigma_l = sample("sigma_l", dist.Gamma(2., 1.).expand([nsub, np]).to_event(2))
        
#        probs_prior = torch.tril(ones(3, 3))
#        nuc1 = sample("nu_c1", dist.HalfCauchy(5.))
#        probsc1 = sample("probs_c1", dist.Dirichlet(ones(2)*nuc1).expand([nsub]).to_event(1))
#        
#        nuc2 = sample("nu_c2", dist.HalfCauchy(5.))
#        probsc2 = sample("probs_c2", dist.Dirichlet(ones(3)*nuc2).expand([nsub]).to_event(1))

        locs = sample("locs", dist.Normal(0., 1.).expand([nsub, np]).to_event(2))
    
        trans_pars = locs * sigma_g / sigma_l + mu
        
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
                depth = max_trials - t - 1
                #probs = probs_prior[max_trials-t-1]

                with plate('responses_{}_{}'.format(b, t), N) as ind:
#                    d = sample('d_{}_{}'.format(b, t),
#                               dist.Categorical(probs[ind]),
#                               infer={"enumerate": "parallel"})
#                   
                    d = depth[ind]
                    sample('obs_{}_{}'.format(b, t), 
                           dist.Bernoulli(logits=logits[d, ind]),
                           obs=res[ind])    

                
    def guide(self):
        
        np = self.agent.np  # number of parameters
        nsub = self.nsub  # number of subjects
        
        loc_mu = param('loc_mu', zeros(np))
        scale_tril_mu = param('scale_tril_mu', 
                              torch.eye(np), 
                              constraint=constraints.lower_cholesky)
        mu = sample('mu', dist.MultivariateNormal(loc_mu, 
                                                  scale_tril=scale_tril_mu))
        
        loc_sigma_g = param('loc_sigma_g', zeros(np))
        scale_sigma_g = param('scale_sigma_g', 
                                 ones(np),
                                 constraint=constraints.positive)
        
        sigma_g = sample('sigma_g', dist.LogNormal(loc_sigma_g, scale_sigma_g).to_event(1))
        
        loc_sigma_l = param('loc_sigma_l', zeros(nsub, np))
        scale_sigma_l = param('scale_sigma_l', 
                                 ones(nsub, np),
                                 constraint=constraints.positive)
        
        sigma_l = sample('sigma_l', dist.LogNormal(loc_sigma_l, scale_sigma_l).to_event(2))
        
        loc = param('loc', zeros(nsub, np))
        scale = param('scale', ones(nsub, np), constraint=constraints.positive)        
        locs = sample('locs', dist.Normal(loc, scale).to_event(2))
        
        return {'mu': mu, 'sigma_g': sigma_g, 'locs': locs, 'sigma_l': sigma_l}
    
    def fit(self, 
            num_iterations = 100, 
            num_particles=10,
            optim_kwargs={'lr':.1},
            centered=True):
        
        clear_param_store()
        
        if centered:
            model = self.centered_model
        else:
            model = self.non_centered_model
        
        self.centered = centered
        guide = self.guide

        svi = SVI(model=model,
                  guide=guide,
                  optim=Adam(optim_kwargs),
                  loss=Trace_ELBO(num_particles=num_particles))#,
#                                      max_plate_nesting=1))

        loss = []
        pbar = tqdm(range(num_iterations), position=0)
        for step in pbar:
            loss.append(svi.step())
            pbar.set_description("Mean ELBO %6.2f" % torch.Tensor(loss[-20:]).mean())
        
        param_store = get_param_store()
        parameters = {}
        for name in param_store.get_all_param_names():
            parameters[name] = param(name)
            
        self.parameters = parameters
        self.loss = loss
        
    def sample_from_posterior(self, labels, n_samples=10000):
        
        import numpy as np
        nsub = self.nsub
        npars = self.agent.np
        assert npars == len(labels)
        
        keys = ['mu', 'sigma_l', 'sigma_g', 'locs']
        
        trans_pars = np.zeros((n_samples, nsub, npars))
        
        sigma_local = np.zeros((n_samples, nsub, npars))

        mean_global = np.zeros((n_samples, npars))
        sigma_global = np.zeros((n_samples, npars))
        
        for i in range(n_samples):
            sample = self.guide()
            for key in keys:
                sample.setdefault(key, ones(1))
                
            mu = sample['mu']
            sigma_g = sample['sigma_g']
            sigma_l = sample['sigma_l']
            
            if self.centered:
                pars = sample['locs']
            else:
                pars = sample['locs'] * sigma_g / sigma_l + mu
            
            trans_pars[i] = pars.detach().numpy()
            
            sigma_local[i] = 1/sigma_l.detach().numpy()
            
            mean_global[i] = mu.detach().numpy()
            sigma_global[i] = sigma_g.detach().numpy()
        
        subject_label = np.tile(range(1, nsub+1), (n_samples, 1)).reshape(-1)
        tp_df = pd.DataFrame(data=trans_pars.reshape(-1, npars), columns=labels)
        tp_df['subject'] = subject_label
        
        sl_df = pd.DataFrame(data=sigma_local.reshape(-1, npars), columns=labels)
        sl_df['subject'] = subject_label
        
        mg_df = pd.DataFrame(data=mean_global, columns=labels)
        sg_df = pd.DataFrame(data=sigma_global, columns=labels)
        
        return (tp_df, sl_df, mg_df, sg_df)