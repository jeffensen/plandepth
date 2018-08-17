#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.distributions import Categorical

import pyro
import pyro.distributions as dist

zeros = torch.zeros
ones = torch.ones
randn = torch.randn

softplus = torch.nn.functional.softplus
logistic = lambda x: 1/(1+torch.exp(-x))

ftype = torch.FloatTensor
itype = torch.LongTensor
btype = torch.ByteTensor

class Random(object):
    def __init__(self,
                 runs = 1, 
                 blocks = 100,
                 na = 2):
        
        self.npars = 1
                
        self.na = na #number of actions
        self.runs = runs #number of independent runs of the experiment (e.g. number of subjects)
        self.blocks = blocks #number of blocks in each run
    
    def set_params(self, x = None):
        if x is not None:
            self.p = logistic(x[:,0]) #probability of selecting jump command
        else:
            self.p = ones(self.runs)/2
            
    def set_context(self, context, max_T):
        self.probs = ones(self.runs, max_T,2)/2
        self.states = zeros(self.runs, max_T+1)
        self.scores = zeros(self.runs, max_T)
        
        self.trials = context[0]
        self.config = context[1]
        self.condition = context[2]
        self.states[:,0] = context[3]
                
    def update_beliefs(self, t, states, scores, responses):
        t_left = self.trials - t
        self.states[:,t] = states
        self.scores[:,t] = scores
        
    def plan_behavior(self, t, depth):
        self.probs[:,t,0] = 1-self.p
        self.probs[:,t,1] = self.p
        
    def sample_responses(self, t):
        self.cat = Categorical(probs = self.probs[t])
        return self.cat.sample()
    

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
            self.costs = ftype([-2, -5]).view(na, 1, 1)

        # expected state value
        self.Vs = ftype(runs, self.depth+1, ns)
        self.Vs[:,0] = torch.stack([self.utility[self.pc[i]] for i in range(runs)])
        
        # action value difference
        self.D = ftype(runs, planning_depth, ns)
        
        # response probability
        self.prob = ftype(runs, trials, na).zero_()
        
        if responses is not None:
            self.responses = var(responses).view(-1)
            self.notnans = btype((~np.isnan(responses.numpy())).astype(int)).view(-1)
            self.states = var(states)
        
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
        
        R = torch.stack([tm[:,i].bmm(self.Vs[:,0][:,:,None]).squeeze()\
                         for i in range(self.na)])
    
        Q = R + acosts        
        for d in range(1,depth+1):
            #compute Q value differences for different actions
            self.D[:, d-1] = Q[0] - Q[1]
            
            #compute response probability
            p = 1/(1+torch.exp(self.D[:,d-1]/tau))
            
            #set state value
            self.Vs[:, d] = p*Q[1] + (1-p)*Q[0]
            
            Q = torch.stack([tm[:,i].bmm(self.Vs[:,d][:,:,None]).squeeze()\
                             for i in range(self.na)])
            Q += R + acosts
        
        self.D[:, -1] = Q[0] - Q[1]
                
        
    def update_beliefs(self, trial, outcomes, responses = None):
        points = outcomes[:,0]
        states = outcomes[:,1].long()
        
        self.planning(states, points, trial)
        
    def planning(self, states, trial, *args):
        if not args:
            noise = ones(self.runs,1)*1e-10
            depth = self.depth*ones(self.runs, dtype = torch.long)
        else:
            depth = args[0]
            if len(args) > 1:
                noise = args[1]
            else:
                noise = ones(self.runs,1)*1e-10
        
        runs = itype(range(self.runs))
        
        remaining_trials = ones(self.runs, dtype=torch.long)*(self.trials-trial)
        d = torch.min(remaining_trials, depth)
        
        #get the probability of jump action
        p = 1/(1+torch.exp(self.D[runs,d-1]/noise))        
        
        probs = p[runs, states]
        self.prob[:, trial, 1] = probs
        self.prob[:, trial, 0] = 1-probs
                    
    def sample_responses(self, trial):
        _, choices = self.prob[:,trial-1,:].max(dim = 0)
        
        return choices
    
    def compute_response_probabilities(self, depth):
        self.compute_state_values()
        for trial in range(self.trials):
            self.planning(self.states[:,trial], trial, depth)