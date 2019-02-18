#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tqdm import tqdm
import pandas as pd

import torch
from torch import ones, zeros, tensor

from torch.distributions import constraints, biject_to

from pyro.distributions.util import broadcast_shape, sum_rightmost

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
        
        self.depth_transition = zeros(2, 3, 2, 3)
        self.depth_transition[0, :, 0] = tensor([1., 0., 0.])
        self.depth_transition[0, :, 1] = tensor([.5, .5, 0.])
        self.depth_transition[1] = tensor([1., 0., 0.])
        
        self.states = stimuli['states']
        self.configs = stimuli['configs']
        self.conditions = stimuli['conditions']
    
    def model_horseshoe(self):
        
        agent = self.agent
        np = agent.np  # number of parameters
        
        nblk =  self.nblk  # number of mini-blocks
        nsub = self.nsub  # number of subjects

        mu = sample("mu", dist.Normal(0., 20.).expand([np]).to_event(1))
        sigma = sample("sigma", dist.HalfCauchy(1.).expand([np]).to_event(1))
        
        nuc1 = sample("nu_c1", dist.HalfCauchy(5.))
        nuc2 = sample("nu_c2", dist.HalfCauchy(5.))
        
        with plate('subjects', nsub):
            locs = sample("locs", dist.Normal(mu, sigma).expand([np]).to_event(1))
            # define priors over planning depth
            probsc1 = sample("probs_c1", dist.Dirichlet(ones(2)*nuc1))
            probsc2 = sample("probs_c2", dist.Dirichlet(ones(3)*nuc2))
            with plate('depths', 3):
                rho = sample('rho', dist.Dirichlet(ones(3)))
            
        agent.set_parameters(locs)
        
        tmp = zeros(nsub, 3)
        tmp[:, :2] = probsc1
        
        priors = torch.stack([tmp, probsc2])
        
        d = tensor([0]*nsub)
        for b in markov(range(nblk)):
            conditions = self.conditions[..., b]
            states = self.states[:, b]
            responses = self.responses[:, b]
            max_trials = conditions[-1]
            
            if b == 0:
               probs = priors[max_trials - 2, range(nsub)].unsqueeze(0)
            else:
               probs = rho
            
            tm = self.depth_transition[:, :, max_trials - 2]
            with plate('responses_{}'.format(b), nsub) as ind:
                d = sample('d0_{}'.format(b), dist.Categorical(probs[d, ind]))
                d1 = sample('d1_{}'.format(b), dist.Categorical(tm[0, d, ind]))
                d2 = sample('d2_{}'.format(b), dist.Categorical(tm[0, d1, ind]))
                
                depth = [d, d1, d2]
            
                for t in markov(range(3)):
                    if t == 0:
                        res = None
                    else:
                        res = responses[:, t-1]
                    
                    agent.update_beliefs(b, t, states[:, t], conditions, res)
                    agent.plan_actions(b, t)
                    
                    valid = self.mask[:, b, t]
                    res = responses[valid, t]

                    logits = agent.logits[-1][depth[t]]
                    logits = torch.diag(logits)[valid]
                    
                    print(logits.shape)
                        
                    sample('obs_{}_{}'.format(b, t), 
                               dist.Bernoulli(logits=logits),
                               obs=res)
   
    def model_horseshoe_plus(self):
        
        agent = self.agent
        np = agent.np  # number of parameters
        
        nblk =  self.nblk  # number of mini-blocks
        nsub = self.nsub  # number of subjects

        mu = sample("mu", dist.Normal(0., 20.).expand([np]).to_event(1))
        sigma = sample("sigma", dist.HalfCauchy(1.).expand([np]).to_event(1))
        
        probs_prior = torch.tril(ones(3, 3))
#        nuc1 = sample("nu_c1", dist.HalfCauchy(5.))
#        probsc1 = sample("probs_c1", dist.Dirichlet(ones(2)*nuc1).expand([nsub]).to_event(1))
#        
#        nuc2 = sample("nu_c2", dist.HalfCauchy(5.))
#        probsc2 = sample("probs_c2", dist.Dirichlet(ones(3)*nuc2).expand([nsub]).to_event(1))
        
        with plate('subjects', nsub):
            scales = sample("scales", dist.HalfCauchy(sigma).to_event(1))
            locs = sample("locs", dist.Normal(mu, scales).to_event(1))
    
        
        agent.set_parameters(locs)
        
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
                max_depth = max_trials - t - 1
                probs = probs_prior[max_depth]

                with plate('responses_{}_{}'.format(b, t), N) as ind:
                    d = sample('d_{}_{}'.format(b, t),
                               dist.Categorical(probs[ind]),
                               infer={"enumerate": "parallel"})
                   
                    sample('obs_{}_{}'.format(b, t), 
                           dist.Bernoulli(logits=logits[d, ind]),
                           obs=res[ind])   
                
    def guide_horseshoe(self):
        
        npar = self.agent.np  # number of parameters
        nsub = self.nsub  # number of subjects
        
        m_hyp = param('m_hyp', zeros(2*npar))
        st_hyp = param('scale_tril_hyp', 
                              torch.eye(2*npar), 
                              constraint=constraints.lower_cholesky)
        hyp = sample('hyp', dist.MultivariateNormal(m_hyp, 
                                                  scale_tril=st_hyp), 
                            infer={'is_auxiliary': True})
        
        unc_mu = hyp[:npar]
        unc_sigma = hyp[npar:]
    
        trns_sigma = biject_to(constraints.positive)
    
        c_sigma = trns_sigma(unc_sigma)
    
        ld_sigma = trns_sigma.inv.log_abs_det_jacobian(c_sigma, unc_sigma)
        ld_sigma = sum_rightmost(ld_sigma, ld_sigma.dim() - c_sigma.dim() + 1)
    
        mu = sample("mu", dist.Delta(unc_mu, event_dim=1))
        sigma = sample("sigma", dist.Delta(c_sigma, log_density=ld_sigma, event_dim=1))
        
        m_locs = param('m_locs', zeros(nsub, npar))
        st_locs = param('s_locs', torch.eye(npar).repeat(nsub, 1, 1), 
                   constraint=constraints.lower_cholesky)
        
        
        l1 = param('l1', zeros(1))
        s1 = param('s1', ones(1), constraint=constraints.positive)
        nuc1 = sample("nu_c1", dist.LogNormal(loc=l1, scale=s1))
        
        l2 = param('l2', zeros(1))
        s2 = param('s2', ones(1), constraint=constraints.positive)
        nuc2 = sample("nu_c2", dist.LogNormal(loc=l2, scale=s2))
        
        alpha1 = param('alpha1', ones(nsub, 2), constraint=constraints.positive)
        alpha2 = param('alpha2', ones(nsub, 3), constraint=constraints.positive)

        with plate('subjects', nsub):
            locs = sample("locs", dist.MultivariateNormal(m_locs, scale_tril=st_locs))
            probsc1 = sample("probs_c1", dist.Dirichlet(alpha1))
            probsc2 = sample("probs_c2", dist.Dirichlet(alpha2))
        
        return {'mu': mu, 'sigma': sigma, 'locs': locs, 'pc1': probsc1, 'pc2': probsc2, 'nuc1': nuc1, 'nuc2': nuc2}
    

    def guide_horseshoe_plus(self):
        
        npar = self.agent.np  # number of parameters
        nsub = self.nsub  # number of subjects
        trns = biject_to(constraints.positive)

        
        m_hyp = param('m_hyp', zeros(2*npar))
        st_hyp = param('scale_tril_hyp', 
                              torch.eye(2*npar), 
                              constraint=constraints.lower_cholesky)
        hyp = sample('hyp', dist.MultivariateNormal(m_hyp, 
                                                  scale_tril=st_hyp), 
                            infer={'is_auxiliary': True})
        
        unc_mu = hyp[:npar]
        unc_sigma = hyp[npar:]
    
    
        c_sigma = trns(unc_sigma)
    
        ld_sigma = trns.inv.log_abs_det_jacobian(c_sigma, unc_sigma)
        ld_sigma = sum_rightmost(ld_sigma, ld_sigma.dim() - c_sigma.dim() + 1)
    
        mu = sample("mu", dist.Delta(unc_mu, event_dim=1))
        sigma = sample("sigma", dist.Delta(c_sigma, log_density=ld_sigma, event_dim=1))
        
        m_tmp = param('m_tmp', zeros(nsub, 2*npar))
        st_tmp = param('s_tmp', torch.eye(2*npar).repeat(nsub, 1, 1), 
                   constraint=constraints.lower_cholesky)

        with plate('subjects', nsub):
            tmp = sample('tmp', dist.MultivariateNormal(m_tmp, 
                                                  scale_tril=st_tmp), 
                            infer={'is_auxiliary': True})
            
            unc_locs = tmp[..., :npar]
            unc_scale = tmp[..., npar:]
            
            c_scale = trns(unc_scale)
            
            ld_scale = trns.inv.log_abs_det_jacobian(c_scale, unc_scale)
            ld_scale = sum_rightmost(ld_scale, ld_scale.dim() - c_scale.dim() + 1)
            
            locs = sample("locs", dist.Delta(unc_locs, event_dim=1))
            scales = sample("scales", dist.Delta(c_scale, log_density=ld_scale, event_dim=1))
        
        return {'mu': mu, 'sigma': sigma, 'locs': locs, 'scales': scales}
    
    def fit(self, 
            num_iterations = 100, 
            num_particles=10,
            optim_kwargs={'lr':.1},
            parametrisation='horseshoe'):
        
        clear_param_store()
        
        if parametrisation == 'horseshoe':
            self.model = self.model_horseshoe
            self.guide = self.guide_horseshoe
        elif parametrisation == 'horseshoe+':
            self.model = self.model_horseshoe_plus
            self.guide = self.guide_horseshoe_plus
        
        svi = SVI(model=self.model,
                  guide=self.guide,
                  optim=Adam(optim_kwargs),
                  loss=TraceEnum_ELBO(num_particles=num_particles, max_plate_nesting=1))

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
        
        keys = ['mu', 'sigma', 'locs', 'scales']
        
        trans_pars = np.zeros((n_samples, nsub, npars))
        tp_scales = np.zeros((n_samples, nsub, npars))

        mean_global = np.zeros((n_samples, npars))
        sigma_global = np.zeros((n_samples, npars))
        
        for i in range(n_samples):
            sample = self.guide()
            for key in keys:
                sample.setdefault(key, ones(1))
                
            mu = sample['mu']
            sigma = sample['sigma']
            scales = sample['scales']
            pars = sample['locs']

            trans_pars[i] = pars.detach().numpy()
            tp_scales[i] = scales.detach().numpy()
            
            mean_global[i] = mu.detach().numpy()
            sigma_global[i] = sigma.detach().numpy()
        
        subject_label = np.tile(range(1, nsub+1), (n_samples, 1)).reshape(-1)
        tp_df = pd.DataFrame(data=trans_pars.reshape(-1, npars), columns=labels)
        tp_df['subject'] = subject_label
        
        tps_df = pd.DataFrame(data=tp_scales.reshape(-1, npars), columns=labels)
        tps_df['subject'] = subject_label
        
        mg_df = pd.DataFrame(data=mean_global, columns=labels)
        sg_df = pd.DataFrame(data=sigma_global, columns=labels)
        
        return tp_df, tps_df, mg_df, sg_df
    
    def sample_posterior_marginal(self, n_samples=100):
        
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        post_depth_samples = {}
        
        pbar = tqdm(range(n_samples), position=0)

        for n in pbar:
            pbar.set_description("Sample posterior depth")
            # get marginal posterior over planning depths
            post_depth = elbo.compute_marginals(self.model, self.guide)
            for name in post_depth.keys():
                post_depth_samples.setdefault(name, [])
                post_depth_samples[name].append(post_depth[name].probs.detach().clone())
        
        for name in post_depth_samples.keys():
            post_depth_samples[name] = torch.stack(post_depth_samples[name]).numpy()
        
        return post_depth_samples