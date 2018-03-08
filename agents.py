#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

import torch
from torch.distributions import Categorical

class Random(object):
    def __init__(self, runs = 1, trials = 1, na = 2, device = 'cpu'):
        
        if device == 'gpu':
            self.ftype = torch.cuda.FloatTensor
            self.itype = torch.cuda.LongTensor
        else:
            self.ftype = torch.FloatTensor
            self.itype = torch.LongTensor
        
        self.na = na #number of actions
        
        self.cat = Categorical(probs = torch.ones(runs, self.na))
        
    def update_beliefs(self, trial, outcomes, responses = None):
        pass

    def planning(self, trial):
        pass
            
    def sample_responses(self, trial):
        
        return self.cat.sample()
        
    
class Informed(object):
    def __init__(self, 
                 transition_matrix, 
                 outcome_likelihood, 
                 runs = 1, 
                 trials = 1, 
                 na = 2,
                 ns = 6,
                 costs = None,
                 planning_depth = 1,
                 device = 'cpu'):
        
        if device == 'gpu':
            self.ftype = torch.cuda.FloatTensor
            self.itype = torch.cuda.LongTensor
        else:
            self.ftype = torch.FloatTensor
            self.itype = torch.LongTensor
        
        self.depth = planning_depth #planning depth
        self.na = na #number of actions
        self.ns = ns #number of states
        
        self.tm = transition_matrix
        self.ol = outcome_likelihood
        self.utility = torch.arange(-2,2+1,1)
        if costs is not None:
            self.costs = costs.view(self.na, 1, 1)
        else:
            self.costs = self.ftype([-.5, -1.]).view(self.na, 1, 1)

        #expected value of each state in each run
        self.Rs = self.ftype(self.depth, self.ns, runs)
        self.Rs[0] = torch.matmul(self.ol, self.utility).transpose(dim0=1,dim1=0)
        
        self.prob = self.ftype(na, trials, runs).zero_()
        
        if self.depth > 1:
            self.compute_state_values()
        
    def compute_state_values(self):
        costs = self.costs
        #compute planning depth dependent state values
        R = torch.matmul(self.tm, self.Rs[0])
        value = R + costs
        for d in range(1,self.depth):
            self.Rs[d], _ = value.max(dim = 0)
            if d < self.depth - 1:
                value = torch.matmul(self.tm, self.Rs[d])+R
                value += costs
    
        
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