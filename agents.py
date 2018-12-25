#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.distributions import Categorical, Bernoulli
from numpy import nan

zeros = torch.zeros
ones = torch.ones
randn = torch.randn

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
            self.p = x[:,0].sigmoid() #probability of selecting jump command
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
        self.np = 2  # number of free model parameters
        
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
            
        self.transitions = torch.tensor([4, 3, 4, 5, 1, 1])

    def set_parameters(self, trans_par = None):
        
        if trans_par is not None:
            assert trans_par.shape[-1] == self.np
#            self.tp_mean0 = trans_par[:, :2].sigmoid()  # transition probabilty for action jump
#            self.tp_scale0 = trans_par[:, 2:4].exp() # precision of beliefs about transition probability
            self.eps = trans_par[:, 0].sigmoid()
            self.beta = trans_par[:, 1].exp()
        else:
            self.tp_mean0 = torch.tensor([.9, .5])\
                                .view(1,-1).repeat(self.runs, 1)
            self.tp_scale0 = 50*ones(self.runs, 2)
            
            self.beta = torch.tensor([1e10]).repeat(self.runs)
            self.eps = .99 * ones(self.runs)
        
        self.tp_mean0 = torch.tensor([.9, .5])\
                                .view(1,-1).repeat(self.runs, 1)
        self.tp_scale0 = 50*ones(self.runs, 2)
        self.tau = torch.tensor([1e10]).repeat(self.runs)
        
        self.tp_mean = [self.tp_mean0]
        self.tp_scale = [self.tp_scale0]
        
        # state transition matrices
        self.tm = []
            
        # expected state value
        self.Vs = []
        
        # action value difference: Q(a=right) - Q(a=jump)
        self.D = []
        
        # response probability
        self.logits = [] 

    def make_transition_matrix(self, p):
        na = self.na  # number of actions
        ns = self.ns  # number of states
        runs = self.runs  # number of runs
        
        tm = zeros(runs, na, ns, ns)
        
        # move left action - no tranistion uncertainty
        tm[:, 0, :-1, 1:] = torch.eye(ns-1).repeat(runs,1,1)
        tm[:, 0,-1,0] = 1
        
        # jump action - with varying levels of transition uncertainty
        tm[:, 1, -2:, 0:3] = (1 - p.reshape(-1, 1, 1).repeat(1, 2, 3))/2 
        tm[:, 1, -2:, 1] = p.reshape(-1, 1).repeat(1, 2)
        tm[:, 1, 2, 3:6] = (1 - p.reshape(-1, 1).repeat(1, 3))/2 
        tm[:, 1, 2, 4] = p
        tm[:, 1, 0, 3:6] = (1 - p.reshape(-1, 1).repeat(1, 3))/2 
        tm[:, 1, 0, 4] = p
        tm[:, 1, 3, 0] = (1 - p)/2; tm[:, 1, 3, -2] = (1 - p)/2
        tm[:, 1, 3, -1] = p
        tm[:, 1, 1, 2:5] = (1 - p.reshape(-1, 1).repeat(1, 3))/2 
        tm[:, 1, 1, 3] = p
        
        self.tm.append(tm)
    
    def compute_state_values(self, block):
        
        acosts = self.costs  # action costs
        tm = self.tm[-1]  # transition matrix
        depth = self.depth  # planning depth
        
        Vs = [(self.utility*self.pc[:, block]).sum(dim=-1)]
    
        D = [] #zeros(self.runs, self.depth, self.ns)
        
        R = torch.stack([tm[:,i].bmm(Vs[-1][..., None]).squeeze()
                         for i in range(self.na)]) + acosts
    
        Q = R
        for d in range(1,depth+1):
            # compute Q value differences for different actions
            D.append(Q[1] - Q[0])
            
            # compute response probability
            p = (D[-1]*self.tau[:,None]).sigmoid()
            
            # set state value
            Vs.append(p * Q[1] + (1-p) * Q[0])

            if d < depth:
                tmp = torch.stack([tm[:,i].bmm(Vs[-1][:,:,None]).squeeze()
                             for i in range(self.na)])
                Q = tmp + R
        
        self.Vs.append(Vs)
        self.D.append(D)        
        
    def update_beliefs(self, block, trial, states, conditions, responses = None):
        
        self.noise = conditions[0]
        self.max_trials = conditions[1]
        
        subs = torch.arange(self.runs)
        
        if trial == 0:
            # update_transition_probability
            self.make_transition_matrix(self.tp_mean[-1][subs, self.noise])
            
        else:
            # update beliefs update state transitions
            succesful_transitions = self.transitions[self.states] == states
            tp_mean = self.tp_mean[-1].clone()
            
            tp_scale = self.tp_scale[-1].clone()*self.eps[:, None] + 1-self.eps[:, None]
            tp_scale[subs, self.noise] += responses
            
            tp_mean[subs, self.noise] += \
                (succesful_transitions.float() - tp_mean[subs, self.noise])\
                * responses / tp_scale[subs, self.noise]
        
            self.tp_mean.append(tp_mean)
            self.tp_scale.append(tp_scale)
            
            self.make_transition_matrix(tp_mean[subs, self.noise])
            
        # set beliefs about state to new state observation
        self.states = states
        
    def plan_actions(self, block, trial):

        self.compute_state_values(block)
        
        D = torch.stack(self.D[-1])

        self.logits.append(D[:, range(self.runs), self.states] * self.beta)
 
    def sample_responses(self, block, trial):
        depth = self.depth
        
        d = self.max_trials-trial
        d[d > depth] = depth
        valid = d > 0
        d[valid] -= 1
        
        logits = self.logits[-1]
        
        cat = Bernoulli(logits=logits[d, range(self.runs)])
        
        res = cat.sample()
        res[~valid] = nan
        
        return res