#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:50:01 2018

@author: markovic
"""

import torch
from torch.distributions import Categorical

class MultiStage(object):
    def __init__(self, outcome_likelihood, 
                 transition_matrix, 
                 runs = 1, 
                 trials = 2, 
                 device = 'cpu'):
        
        if device == 'gpu':
            self.ftype = torch.cuda.FloatTensor
            self.itype = torch.cuda.LongTensor
        else:
            self.ftype = torch.FloatTensor
            self.itype = torch.LongTensor
        
        self.nstates = 6 #number of states
        
        self.list = self.itype(range(runs))
        
        self.ol = outcome_likelihood
        self.tm = transition_matrix
        self.states = self.itype(runs, trials+1)
        cat = Categorical(probs = torch.ones(runs, self.nstates))
        self.states[:,0] = cat.sample()
    
    def update_states(self, trial, responses):
        state_prob = self.tm[responses, self.states[:, trial-1]]
        cat = Categorical(probs = state_prob)
        
        self.states[:,trial] = cat.sample()
        
    def sample_outcomes(self, trial):
        outcome_prob = self.ol[self.list, self.states[:,trial]]
        cat = Categorical(probs = outcome_prob)
        
        return torch.stack((cat.sample(), self.states[:,trial]), 1)
        