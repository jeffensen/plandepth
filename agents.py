#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""
from tqdm import tqdm

import numpy as np

import torch
from torch.distributions import Categorical, constraints
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import Adam

import matplotlib.pyplot as plt

zeros = torch.zeros
ones = torch.ones
var = Variable
randn = torch.randn
softplus = torch.nn.Softplus()

ftype = torch.FloatTensor
itype = torch.LongTensor
btype = torch.ByteTensor

class Random(object):
    def __init__(self,
                 responses = None, 
                 states = None, 
                 configs = None,
                 runs = 1, 
                 blocks = 1,
                 trials = 3,
                 na = 2):
        
        self.na = na #number of actions
        self.runs = runs #number of independent runs of the experiment (e.g. number of subjects)
        self.blocks = blocks
        self.trials = trials
        
        self.cat = Categorical(probs = ones(runs, self.na))
        
        #measured responses
        if responses is not None:
            self.responses = var(itype(responses)).view(-1,1)
            self.notnans = ~btype(np.isnan(responses).astype(int)).view(-1)
        
    def update_beliefs(self, trial, outcomes, responses = None):
        pass

    def planning(self, trial):
        pass
            
    def sample_responses(self, trial):
        
        return self.cat.sample()
    
    def model(self):
        #hyperprior across subjects
        tau = pyro.sample('tau', 
                          dist.halfcauchy, 
                          Variable(zeros(1)),
                          Variable(ones(1)))
        
        #hyperprior subject specific
        lam = pyro.sample('lam',
                          dist.halfcauchy,
                          Variable(zeros(self.runs)),
                          tau*Variable(ones(self.runs)))
        
        #subject specific response probability
        p = pyro.sample('p',
                        dist.dirichlet, 
                        Variable(ones(self.runs, self.na))/lam[:,None])
        
        p = p.repeat(self.blocks, self.trials, 1, 1).view(-1)
        p = p[self.notnans.repeat(2)].view(-1,2)
        
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
        
    
class Informed(object):
    def __init__(self, 
                 planet_confs,
                 transition_probability = ones(1),
                 responses = None,
                 states = None, 
                 runs = 1,
                 trials = 1, 
                 na = 2,
                 ns = 6,
                 costs = None,
                 planning_depth = 1):
        
                
        self.runs = runs
        self.trials = trials
        
        self.depth = planning_depth #planning depth
        self.na = na #number of actions
        self.ns = ns #number of states
        
        self.make_transition_matrix(transition_probability)
        #matrix containing planet type in each state
        self.pc = planet_confs
        self.utility = torch.arange(-20,30,10)
        if costs is not None:
            self.costs = costs.view(na, 1, 1)
        else:
            self.costs = ftype([-.5, -1.]).view(na, 1, 1)

        # expected state value
        self.Vs = ftype(runs, planning_depth, ns)
        self.Vs[:,0] = torch.stack([self.utility[self.pc[i]] for i in range(runs)])
        
        # action value difference
        self.D = ftype(runs, planning_depth, ns)
        
        # response probability
        self.prob = ftype(na, trials, runs).zero_()
        
#        self.compute_state_values()
            
    def make_transition_matrix(self, p):
        # p -> transition probability
        na = self.na
        ns = self.ns
        runs = self.runs
        
        if len(p) == 1:
            p = p.repeat(runs)
        
        self.tm = ftype(runs, na, ns, ns).zero_()
        # certain action 
        self.tm[:, 0, :-1, 1:] = torch.eye(ns-1).repeat(runs,1,1)
        self.tm[:, 0,-1,0] = 1
        # uncertain action
        self.tm[:, 1, -2:, 0:3] = (1-p[:, None,None].repeat(1,2,3))/2 
        self.tm[:, 1, -2:, 1] = p[:, None].repeat(1,2)
        self.tm[:, 1, 2, 3:6] = (1-p[:,None].repeat(1,3))/2 
        self.tm[:,1, 2, 4] = p
        self.tm[:,1, 0, 3:6] = (1-p[:,None].repeat(1,3))/2 
        self.tm[:,1, 0, 4] = p
        self.tm[:,1, 3, 0] = (1-p)/2; self.tm[:,1, 3, -2] = (1-p)/2
        self.tm[:,1, 3, -1] = p
        self.tm[:,1, 1, 2:5] = (1-p[:,None].repeat(1,3))/2 
        self.tm[:,1, 1, 3] = p
        
    def compute_state_values(self, tau = None):
        if tau is None:
            tau = ones(self.runs,1)*1e-10
        acosts = self.costs #action costs
        tm = self.tm #transition matrix
        depth = self.depth #planning depth
        
        R = torch.stack([torch.bmm(tm[:,i], self.Vs[:,0][:,:,None]).squeeze()\
                         for i in range(self.na)])
        
        Q = R + acosts        
        for d in range(1,depth):
            #compute Q value differences for different actions
            self.D[:, d-1] = Q[0] - Q[1]
            
            #compute response probability
            p = 1/(1+torch.exp(self.D[:,d-1])/tau)
            
            #set state value
            self.Vs[:, d] = p*Q[1] + (1-p)*Q[0]
            
            if d < depth - 1:
                Q = torch.stack([torch.bmm(tm[:,i], self.Vs[:,d][:,:,None]).squeeze()\
                                 for i in range(self.na)])
                Q += R + acosts
                
        
    def update_beliefs(self, trial, outcomes, responses = None):
        points = outcomes[:,0]
        states = outcomes[:,1].long()
        
        self.planning(states, trial)
        
    def planning(self, states, trial):
        runs = self.prob.shape[0]
        T = self.prob.shape[1]
        d = min(T-trial+1, self.depth)
#        d=self.depth
        
        #get state value
        if d > 1:
            value = self.Rs[0] + self.Rs[d-1]
        else:
            value = self.Rs[0]
        
        R = torch.matmul(self.tm, value)
        R += self.costs
        for i,s in enumerate(states):
            self.prob[:,trial-1, i] = R[:,s,i]
                    
    def sample_responses(self, trial):
        _, choices = self.prob[:,trial-1,:].max(dim = 0)
        
        return choices
    
    def model(self):
        
        depth = pyro.sample('depth', 
                            dist.categorical, 
                            ones(self.runs*self.blocks, 3)/3)
        
        #subject specific response probability
        p = self.get_response_probabilities(depth)
        
        return pyro.sample('responses', dist.categorical, p)
    
    def guide(self):
        
        probs = pyro.param('probs', ones(self.runs*self.blocks, 3)/3, constraint = constraints.simplex)
        depth = pyro.sample('depth', dist.categorical, probs)
        
        return depth
    
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