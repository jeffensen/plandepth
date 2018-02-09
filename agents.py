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
                 device = 'cpu'):
        
        if device == 'gpu':
            self.ftype = torch.cuda.FloatTensor
            self.itype = torch.cuda.LongTensor
        else:
            self.ftype = torch.FloatTensor
            self.itype = torch.LongTensor
        
        self.na = na #number of actions
        
        self.tm = transition_matrix
        self.ol = outcome_likelihood
        
        self.prob = self.ftype(runs, trials, na)
        self.utility = torch.arange(-2,2+1,1)
        
    def update_beliefs(self, trial, outcomes, responses = None):
        points = outcomes[:,0]
        states = outcomes[:,1].long()
        
        self.planning(states, trial)
        
    def planning(self, states, trial):
        T = self.prob.shape[1]
        d = T-trial
        
        print(self.tm[:,states,:].shape)
            
    def sample_responses(self, trial):
        
        return self.prob[:,trial,:].max(dim = -1)