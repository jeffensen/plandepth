# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 15:58:30 2023

@author: goenner
"""

import numpy as np

def calc_BIC(inferrer, responses, conditions, m_prob, n_actions_considered = 1):
    # Evaluate (log-)likelihood of actual choices, given the inferred model parameters:    
    n_subj = len(responses)
    n_miniblocks = responses.shape[1]
    nll_firstaction_mean = np.nan * np.ones([n_subj, n_miniblocks])    
    nll_firstaction_depths = np.nan * np.ones([n_subj, n_miniblocks, 3])
    nll_hinoise_all = np.nan * np.ones([n_subj, 3])
    nll_lonoise_all = np.nan * np.ones([n_subj, 3])
    nll_all = np.nan * np.ones([n_subj, 3])
    nll_hinoise_120 = np.nan * np.ones([n_subj, 3])
    nll_lonoise_120 = np.nan * np.ones([n_subj, 3])
    nll_120 = np.nan * np.ones([n_subj, 3])
    nll_hinoise_120_mean = np.nan * np.ones([n_subj])
    nll_lonoise_120_mean = np.nan * np.ones([n_subj])
    nll_120_mean = np.nan * np.ones([n_subj])
    pseudo_rsquare_120_mean = np.nan * np.ones([n_subj])
    pseudo_rsquare_hinoise_120_mean = np.nan * np.ones([n_subj])
    pseudo_rsquare_lonoise_120_mean = np.nan * np.ones([n_subj])
    #m_param_count = len(np.unique(pars_df_ya['parameter'])) #  WRONG! Correct solution:  m_param_count = inferrer.agent.np
    m_param_count = inferrer.agent.np
    BIC_120_mean = np.nan * np.ones([n_subj])
    BIC_hinoise_120 = np.nan * np.ones([n_subj])
    BIC_lonoise_120 = np.nan * np.ones([n_subj])


    for i_mblk in range(n_miniblocks):
        for i_action in range(n_actions_considered): # Consider only first action; use range(3) to consider all action steps
            #resp = inferrer.responses[0][i_mblk].numpy()[i_action]      # Index 0 to get rid of "duplicate" data
            #logits_depths = inferrer.agent.logits[3*i_mblk + i_action].detach().numpy()[0] # agent.logits = Value difference between Jump and Move            
            # CAUTION: If there are as many logit values as particles (e.g., 100), we have to extract the mean rather than index 0!
            logits_depths = inferrer.agent.logits[3*i_mblk + i_action].detach().numpy() #.mean(0) # agent.logits = Value difference between Jump and Move             
        
            for i_subj in range(n_subj):
                resp = inferrer.responses[i_subj][i_mblk].numpy()[i_action] 

                p_jump_depths = 1.0 / (1 + np.exp(-logits_depths[i_subj])) # softmax function for choice probabilities
                p_move_depths = 1 - p_jump_depths
                if resp==1:
                    nll = -np.log(p_jump_depths) # Apply -np.log() to obtain the negative log-likelihood (nll), for numerical reasons
                elif resp==0:
                    nll = -np.log(p_move_depths)                              
                nll_firstaction_depths[i_subj, i_mblk, :] = nll              
                #nll_firstaction_mean[i_subj, i_mblk] = np.matmul(nll_firstaction_depths[i_subj, i_mblk, :], m_prob[0][i_mblk,0,:]) # WRONG - m_prob_ya[0] has shape (n_miniblocks, n_subjects, n_steps !!!
                nll_firstaction_mean[i_subj, i_mblk] = np.matmul(nll_firstaction_depths[i_subj, i_mblk, :], m_prob[0][i_mblk, i_subj, :])            

                nll_hinoise_all[i_subj, :] = np.matmul(nll_firstaction_depths[i_subj, :, :].transpose(), conditions[0][i_subj, :].numpy()) # Sum of NLL for high noise (noise==1) 
                nll_lonoise_all[i_subj, :] = np.matmul(nll_firstaction_depths[i_subj, :, :].transpose(), 1 - conditions[0][i_subj, :].numpy()) # Sum of NLL for low noise (noise==0)
                nll_all[i_subj, :] = nll_hinoise_all[i_subj, :] + nll_lonoise_all[i_subj, :]
                nll_hinoise_120[i_subj, :] = np.matmul(nll_firstaction_depths[i_subj, 20:, :].transpose(), conditions[0][i_subj, 20:].numpy()) # Exclude 20 training miniblocks
                nll_lonoise_120[i_subj, :] = np.matmul(nll_firstaction_depths[i_subj, 20:, :].transpose(), 1 - conditions[0][i_subj, 20:].numpy()) 
                nll_120[i_subj, :] = nll_hinoise_120[i_subj, :] + nll_lonoise_120[i_subj, :]

                nll_hinoise_120_mean[i_subj] = np.matmul(nll_firstaction_mean[i_subj, 20:].transpose(), conditions[0][i_subj, 20:].numpy()) # Exclude 20 training miniblocks
                nll_lonoise_120_mean[i_subj] = np.matmul(nll_firstaction_mean[i_subj, 20:].transpose(), 1 - conditions[0][i_subj, 20:].numpy()) 
                nll_120_mean[i_subj] = nll_hinoise_120_mean[i_subj] + nll_lonoise_120_mean[i_subj]

                nll_random_120 = -np.log(0.5)*120 * n_actions_considered
                nll_random_60 = -np.log(0.5)*60 * n_actions_considered    
                pseudo_rsquare_120_mean[i_subj] = 1 - (nll_120_mean[i_subj] / nll_random_120)
                pseudo_rsquare_hinoise_120_mean[i_subj] = 1 - (nll_hinoise_120_mean[i_subj] / nll_random_60)    
                pseudo_rsquare_lonoise_120_mean[i_subj] = 1 - (nll_lonoise_120_mean[i_subj] / nll_random_60)        

                BIC_120_mean[i_subj] = 2*nll_120_mean[i_subj] + m_param_count*np.log(120 * n_actions_considered)
                BIC_hinoise_120[i_subj] = 2*nll_hinoise_120_mean[i_subj] + m_param_count*np.log(60 * n_actions_considered)
                BIC_lonoise_120[i_subj] = 2*nll_lonoise_120_mean[i_subj] + m_param_count*np.log(60 * n_actions_considered)    
                
    return pseudo_rsquare_120_mean, BIC_120_mean, \
           pseudo_rsquare_hinoise_120_mean, BIC_hinoise_120, \
           pseudo_rsquare_lonoise_120_mean, BIC_lonoise_120
           