#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.distributions import Categorical

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
    
    def set_params(self, x = None, max_T=3):
        if x is not None:
            self.p = logistic(x[:,0]) #probability of selecting jump command
        else:
            self.p = ones(self.runs)/2
            
        self.probs = ones(self.blocks, self.runs, max_T,2)/2

            
    def set_context(self, context, max_T):
        self.states = zeros(self.runs, max_T+1)
        self.scores = zeros(self.runs, max_T)
        
        self.trials = context[0]
        self.config = context[1]
        self.condition = context[2]
        self.states[:,0] = context[3]
        pass
                
    def update_beliefs(self, t, states, scores, responses):
        t_left = self.trials - t
        self.states[:,t] = states
        self.scores[:,t] = scores
        pass
        
    def plan_behavior(self,b,t, depth):
        self.probs[b,:,t,0] = 1-self.p
        self.probs[b,:,t,1] = self.p
        
    def sample_responses(self, t):
        self.cat = Categorical(probs = self.probs[t])
        return self.cat.sample()
    

class BackInduction(object):
    def __init__(self, 
                 planet_confs,
                 runs=1,
                 mini_blocks=1,
                 trials=1, 
                 na=2,
                 ns=6,
                 costs=None,
                 planning_depth=1):
        
                
        self.runs = runs
        self.nmb = mini_blocks
        self.trials = trials
        
        self.depth = planning_depth  # maximal planning depth
        self.na = na  # number of actions
        self.ns = ns  # number of states
        
        # matrix containing planet type in each state
        self.pc = planet_confs
        self.utility = torch.arange(-20., 30., 10)
        if costs is not None:
            self.costs = costs.view(na, 1, 1)
        else:
            self.costs = ftype([-2, -5]).view(na, 1, 1)

    def set_parameters(self, trans_par = None):
        
        if trans_par is not None:
            self.trans_prob = logistic(trans_par[:, :2])  # transition probabilty for action jump
            self.beta = trans_par[:, 2]
        else:
            self.trans_prob = torch.tensor([.9, .5])\
                                .view(1,-1).repeat(self.runs, 1)
            
            self.beta = torch.tensor([1e10]).repeat(self.runs)
            
        self.tau = torch.tensor([1e10]).repeat(self.runs)
            
        # expected state value
        self.Vs = zeros(self.runs, self.nmb, self.depth + 1, self.ns)
        
        # action value difference
        self.D = zeros(self.runs, self.nmb, self.depth, self.ns)
        
        # response probability
        self.prob = ones(self.runs, self.nmb, self.trials, self.na)/self.na 

    def make_transition_matrix(self, p):
        na = self.na  # number of actions
        ns = self.ns  # number of states
        runs = self.runs  # number of runs
        
        self.tm = ftype(runs, na, ns, ns).zero_()
        
        # move left action - no tranistion uncertainty
        self.tm[:, 0, :-1, 1:] = torch.eye(ns-1).repeat(runs,1,1)
        self.tm[:, 0,-1,0] = 1
        
        # jump action - with varying levels of transition uncertainty
        self.tm[:, 1, -2:, 0:3] = (1 - p[:, None, None].repeat(1, 2, 3))/2 
        self.tm[:, 1, -2:, 1] = p[:, None].repeat(1, 2)
        self.tm[:, 1, 2, 3:6] = (1 - p[:, None].repeat(1, 3))/2 
        self.tm[:, 1, 2, 4] = p
        self.tm[:, 1, 0, 3:6] = (1 - p[:, None].repeat(1, 3))/2 
        self.tm[:, 1, 0, 4] = p
        self.tm[:, 1, 3, 0] = (1 - p)/2; self.tm[:, 1, 3, -2] = (1 - p)/2
        self.tm[:, 1, 3, -1] = p
        self.tm[:, 1, 1, 2:5] = (1 - p[:,None].repeat(1, 3))/2 
        self.tm[:, 1, 1, 3] = p
        
    def compute_state_values(self, block):
        
        acosts = self.costs  # action costs
        tm = self.tm  # transition matrix
        depth = self.depth  # planning depth
        
        Vs = zeros(self.runs, self.depth+1, self.ns) 
        Vs[:, 0] = (self.utility*self.pc[:, block]).sum(dim=-1)
    
        D = zeros(self.runs, self.depth, self.ns)
        
        R = torch.stack([tm[:,i].bmm(Vs[:, 0][...,None]).squeeze()
                         for i in range(self.na)])
    
        Q = R + acosts        
        for d in range(1,depth+1):
            # compute Q value differences for different actions
            D[:, d-1] = Q[0] - Q[1]
            
            # compute response probability
            p = 1/(1+torch.exp(D[:,d-1]*self.beta[:,None]))
            
            # set state value
            Vs[:, d] = p*Q[1] + (1-p)*Q[0]

            if d < depth:
                Q = torch.stack([tm[:,i].bmm(Vs[:,d][:,:,None]).squeeze()
                             for i in range(self.na)])
                Q += R + acosts
        
        self.Vs[:, block] = Vs
        self.D[:, block] = D        
        
    def update_beliefs(self, block, trial, states, conditions, responses = None):
        self.states = states
        self.noise = conditions[0]
        self.max_trials = conditions[1]
        
        
        if trial == 0:
            # update_transition_probability
            tp = self.trans_prob[range(self.runs), self.noise]
            self.make_transition_matrix(tp)
        
        
    def planning(self, block, trial, *args):
        if trial == 0:
            self.compute_state_values(block)
        
        if not args:
            depth = self.depth
        else:
            depth = args[0]
            
        d = self.max_trials-trial
        d[d > depth] = depth
        valid = d > 0
        
        D = self.D[valid, block]
        states = self.states[valid]
        tau = self.tau[valid]
        value_loc = d[valid]-1
        
        #get the probability of jump action
        p = 1/(1 + torch.exp(D[range(valid.sum()), value_loc, states] * tau))        
        
        self.prob[valid, block, trial, 1] = p
        self.prob[valid, block, trial, 0] = 1-p
                    
    def sample_responses(self, block, trial):
        cat = Categorical(probs=self.prob[:, block, trial])
        
        return cat.sample()