#!/usr/bin/env python
# coding: utf-8

# Here I will illustrate the steps needed to fit the behavioural model to empirical data. This entails determining the posterior beleifs over model parameters and mini-block specific planning depth.

# In[1]:

import torch

from scipy import io

import pandas as pd

#from torch import zeros, ones

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context = 'talk', style = 'white')
sns.set_palette("colorblind", n_colors=5, color_codes=True)



# In[2]:


#from pathlib import Path
from os.path import expanduser, isdir, join
from os import listdir, walk
#import json

from helper_files import load_and_format_behavioural_data, get_posterior_stats, errorplot#, map_noise_to_values
    



# Probabilistic inference
import sys
sys.path.append('../')

from agents_discount_hyperbolic_theta_alpha_kmax30 import BackInductionDiscountHyperbolicThetaAlphakmax30
from inference import Inferrer

def variational_inference(stimuli, mask, responses):
    max_depth = 3
    runs, mini_blocks, max_trials = responses.shape
    
    confs = stimuli['configs']
    
    # define agent
    agent = BackInductionDiscountHyperbolicThetaAlphakmax30(confs,
                          runs=runs,
                          mini_blocks=mini_blocks,
                          trials=max_trials,
                          costs = torch.tensor([0,0]), # LG: Forgot to set to zero - corrected on 2022-06-27!!!                          
                          planning_depth=max_depth)

    # load inference module and start model fitting
    infer = Inferrer(agent, stimuli, responses, mask)
    infer.fit(num_iterations=2, num_particles=100, optim_kwargs={'lr': .010}) # 1000 
    
    return infer


# In[3]:
import time as time
#print("Sleeping for 1 hour!")
#time.sleep(1*60*60) # time given in seconds

# load and format behavioural data
path1 = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/LOG_Files/full_datasets_OA/'
path2 = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/LOG_Files/full_datasets_YA/'
localpath = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Results_discount_hyperbolic_theta_alpha_kmax30' # LG

#path1 = '/home/h7/goenner/TRR265-B09/LOG_Files/full_datasets_OA/'
#path2 = '/home/h7/goenner/TRR265-B09/LOG_Files/full_datasets_YA/'
#localpath = '/home/h7/goenner/TRR265-B09/Analysis SAT-PD2-Sophia/plandepth/Results_discount_hyperbolic_theta_realprobs/'


filenames = ["space_adventure_pd-results.json",
             "space_adventure_pd_inv-results.json"]    # posible filenames of SAT logfiles

stimuli_oa, mask_oa, responses_oa, conditions_oa, ids_oa = load_and_format_behavioural_data(path1, filenames)
stimuli_ya, mask_ya, responses_ya, conditions_ya, ids_ya = load_and_format_behavioural_data(path2, filenames)


        # In[6]:


# sample from posterior
def format_posterior_samples(infer):
    labels = [r'$\tilde{\beta}$', r'$\theta$', r'$\tilde{\alpha}$', r'$k$']
    pars_df, mg_df, sg_df = infer.sample_from_posterior(labels)
    
    # transform sampled parameter values to the true parameter range
    pars_df[r'$\beta$'] = torch.from_numpy(pars_df[r'$\tilde{\beta}$'].values).exp().numpy()
    pars_df[r'$\tilde{\alpha}$'] = torch.from_numpy(pars_df[r'$\tilde{\alpha}$'].values).sigmoid().numpy()
    pars_df[r'$k$'] = 30*torch.from_numpy(pars_df[r'$k$'].values).sigmoid().numpy()    

    pars_df.drop([r'$\tilde{\beta}$'], axis=1, inplace=True) # , r'$\gamma_{loNoise}$', r'$\gamma_{hiNoise}$'
    return pars_df.melt(id_vars=['subject'], var_name='parameter')


# In[5]:


# Variational inference
infer_oa = variational_inference(stimuli_oa, mask_oa, responses_oa)
pars_df_oa = format_posterior_samples(infer_oa)

n_samples = 100
post_marg_oa = infer_oa.sample_posterior_marginal(n_samples=n_samples)
pars_df_oa['IDs'] = np.array(ids_oa)[pars_df_oa.subject.values - 1]
#pars_df_oa['subjIDs'] = np.array(subj_IDs_oa)[pars_df_oa.subject.values - 1]

# plot convergence of ELBO bound (approximate value of the negative marginal log likelihood)
fig, axes = plt.subplots(1, 1, figsize=(15, 5))
axes.plot(infer_oa.loss[100:])  # 100:
df_loss = pd.DataFrame(infer_oa.loss)    
axes.plot(range(len(df_loss[0].rolling(window=25).mean())-100), df_loss[0].rolling(window=25).mean()[100:], lw=1)    #                 
axes.set_title('ELBO Testdaten')
fig.savefig(localpath + '/ELBO discount_hyperbolic_theta_alpha_kmax30.jpg')


g = sns.FacetGrid(pars_df_oa, col="parameter", height=5, sharey=False);
g = g.map(errorplot, 'subject', 'value').add_legend();
#g.axes[0,0].set_ylim([-1, 2])
#g.axes[0,1].set_ylim([0, 7])
#g.axes[0,2].set_ylim([0, 1])

g.fig.savefig(localpath + '/parameter_participant_oa_discount_hyperbolic_theta_alpha_kmax30.jpg')

# In[6]:


infer_ya = variational_inference(stimuli_ya, mask_ya, responses_ya)
pars_df_ya = format_posterior_samples(infer_ya)

n_samples = 100
post_marg_ya = infer_ya.sample_posterior_marginal(n_samples=n_samples)
pars_df_ya['IDs'] = np.array(ids_ya)[pars_df_ya.subject.values - 1]





# The static parametrisations assumes that the planning depth at the start of every mini block is sampled from the same prior distribution (dependent on the number of choices, 2 or 3), independent on the planning depth in the previous trial. Alternative is to use dynamic parametrisation where planning depth is describes as a hidden markov or hidden semi-markov process. The dynamic representation would be useful to quantify the stability of planning depths within and between groups. For example, certain group of participants might be charachtersied by more varying planning depth between trials than other. 
# 
# Unfortuntaly, I still have problems implementing the dynamic representations in a meaningful way. If you would like to work on it, I can explain the details of the current model and related problems. 

# In[7]:


# plot convergence of ELBO bound (approximate value of the negative marginal log likelihood)

fig, axes = plt.subplots(1, 1, figsize=(15, 5))
axes.plot(infer_ya.loss[100:])  # 100:
df_loss = pd.DataFrame(infer_ya.loss)    
axes.plot(range(len(df_loss[0].rolling(window=25).mean())-100), df_loss[0].rolling(window=25).mean()[100:], lw=1)    #                 
axes.set_title('ELBO Testdaten')
fig.savefig(localpath + '/ELBO Testdaten_ya_discount_hyperbolic_theta_alpha_kmax30.jpg')

# In[8]:


# visualize posterior parameter estimates over subjects


# visualize posterior parameter estimates over subjects

g = sns.FacetGrid(pars_df_ya, col="parameter", height=5, sharey=False);
g = g.map(errorplot, 'subject', 'value').add_legend();
#g.axes[0,0].set_ylim([-1, 2])
#g.axes[0,1].set_ylim([0, 7])
#g.axes[0,2].set_ylim([0, 1])

g.fig.savefig(localpath + '/parameter_participant_ya_discount_hyperbolic_theta_alpha_kmax30.jpg')

# In[9]:


# plot posterior distribution over groups
pars_df_oa['group'] = 'OA'
pars_df_ya['group'] = 'YA'

pars_df = pars_df_oa.append(pars_df_ya, ignore_index=True)

g = sns.FacetGrid(pars_df, col="parameter", hue='group', height=5, sharey=False, sharex=False, palette='Set1');
g = g.map(sns.kdeplot, 'value').add_legend();
g.fig.savefig(localpath + '/post_group_parameters_OA_YA_discount_hyperbolic_theta_alpha_kmax30.pdf', dpi=300)


# In[10]:


pars_df = pars_df_oa.append(pars_df_ya)
pars_df.to_csv(localpath + '/pars_post_samples_discount_hyperbolic_theta_alpha_kmax30.csv')



# In what follows we will compute the posterior marginal over planning depth, compute exceedanace probability and plot the results for individual subjects and for the group level results.

# In[12]:




post_depth_oa, m_prob_oa, exc_count_oa = get_posterior_stats(post_marg_oa)
np.savez(localpath + '/oa_plandepth_stats_B03_discount_hyperbolic_theta_alpha_kmax30', post_depth_oa, m_prob_oa, exc_count_oa)

post_depth_ya, m_prob_ya, exc_count_ya = get_posterior_stats(post_marg_ya)
np.savez(localpath + '/ya_plandepth_stats_B03_discount_hyperbolic_theta_alpha_kmax30', post_depth_ya, m_prob_ya, exc_count_ya)



#file = b = np.load('/home/sass/Dokumente/plandepth/oa_plandepth_stats_B03.npz', allow_pickle=True)
file = b = np.load(localpath + '/oa_plandepth_stats_B03_discount_hyperbolic_theta_alpha_kmax30.npz', allow_pickle=True)
fs = file.files # names of the stored arrays (['post_depth_oa', 'm_prob_oa', 'exc_count_oa'])
post_meanPD_firstAction_oa = np.matmul(file[fs[1]][0,:,:,:], np.arange(1,4))
import pandas as pd
dict_ids_oa={}
dict_ids_oa['ID'] = ids_oa
pd.DataFrame(dict_ids_oa).to_csv(localpath + '/IDs_OA.csv')
df_oa_meanPD = pd.DataFrame(post_meanPD_firstAction_oa) # Without subject IDs
#df_oa_meanPD = np.transpose(pd.DataFrame(dict_ids_oa)).append(pd.DataFrame(post_meanPD_firstAction_oa)) # With subject IDs
pd.DataFrame(df_oa_meanPD).to_csv(localpath + '/meanPD_1st_action_oa_single_discount_hyperbolic_theta_alpha_kmax30.csv')
file.close()



#file = b = np.load('/home/sass/Dokumente/plandepth/ya_plandepth_stats_B03.npz', allow_pickle=True)
file = b = np.load(localpath + '/ya_plandepth_stats_B03_discount_hyperbolic_theta_alpha_kmax30.npz', allow_pickle=True)
fs = file.files # names of the stored arrays (['post_depth_ya', 'm_prob_ya', 'exc_count_ya'])
post_meanPD_firstAction_ya = np.matmul(file[fs[1]][0,:,:,:], np.arange(1,4))

df_ya_meanPD = pd.DataFrame(post_meanPD_firstAction_ya) # Without subject IDs
pd.DataFrame(df_ya_meanPD).to_csv(localpath + '/meanPD_1st_action_ya_single_discount_hyperbolic_theta_alpha_kmax30.csv')
dict_ids_ya={}
dict_ids_ya['ID'] = ids_ya
pd.DataFrame(dict_ids_ya).to_csv(localpath + '/IDs_YA.csv')
#df_ya_meanPD = np.transpose(pd.DataFrame(dict_ids_ya)).append(pd.DataFrame(post_meanPD_firstAction_oa)) # With subject IDs


file.close()





# Evaluate (log-)likelihood of actual choices, given the inferred model parameters:    
n_subj_oa = len(responses_oa)    
nll_firstaction_oa_mean = np.nan * np.ones([n_subj_oa, 140])    
nll_firstaction_oa_depths = np.nan * np.ones([n_subj_oa, 140, 3])
nll_hinoise_all_oa = np.nan * np.ones([n_subj_oa, 3])
nll_lonoise_all_oa = np.nan * np.ones([n_subj_oa, 3])
nll_all_oa = np.nan * np.ones([n_subj_oa, 3])
nll_hinoise_120_oa = np.nan * np.ones([n_subj_oa, 3])
nll_lonoise_120_oa = np.nan * np.ones([n_subj_oa, 3])
nll_120_oa = np.nan * np.ones([n_subj_oa, 3])
nll_hinoise_120_mean_oa = np.nan * np.ones([n_subj_oa])
nll_lonoise_120_mean_oa = np.nan * np.ones([n_subj_oa])
nll_120_mean_oa = np.nan * np.ones([n_subj_oa])
pseudo_rsquare_120_mean_oa = np.nan * np.ones([n_subj_oa])
pseudo_rsquare_hinoise_120_mean_oa = np.nan * np.ones([n_subj_oa])
pseudo_rsquare_lonoise_120_mean_oa = np.nan * np.ones([n_subj_oa])
m_param_count = infer_oa.agent.np
BIC_120_mean_oa = np.nan * np.ones([n_subj_oa])
BIC_hinoise_120_oa = np.nan * np.ones([n_subj_oa])
BIC_lonoise_120_oa = np.nan * np.ones([n_subj_oa])


for i_mblk in range(140):
    for i_action in range(1): # Consider only first action; use range(3) to consider all action steps
        #resp = infer_oa.responses[0][i_mblk].numpy()[i_action]      # Index 0 to get rid of "duplicate" data
        #logits_depths = infer_oa.agent.logits[3*i_mblk + i_action].detach().numpy()[0] # agent.logits = Value difference between Jump and Move            
        # CAUTION: If there are as many logit values as particles (e.g., 100), we have to extract the mean rather than index 0!
        logits_depths = infer_oa.agent.logits[3*i_mblk + i_action].detach().numpy() #.mean(0) # agent.logits = Value difference between Jump and Move             
        
        for i_subj in range(n_subj_oa):
            resp = infer_oa.responses[i_subj][i_mblk].numpy()[i_action] 

            p_jump_depths = 1.0 / (1 + np.exp(-logits_depths[i_subj])) # softmax function for choice probabilities
            p_move_depths = 1 - p_jump_depths
            if resp==1:
                nll = -np.log(p_jump_depths) # Apply -np.log() to obtain the negative log-likelihood (nll), for numerical reasons
            elif resp==0:
                nll = -np.log(p_move_depths)                              
            nll_firstaction_oa_depths[i_subj, i_mblk, :] = nll              
            #nll_firstaction_oa_mean[i_subj, i_mblk] = np.matmul(nll_firstaction_oa_depths[i_subj, i_mblk, :], m_prob_oa[0][i_mblk,0,:]) # WRONG - m_prob_ya[0] has shape (n_miniblocks, n_subjects, n_steps !!!
            nll_firstaction_oa_mean[i_subj, i_mblk] = np.matmul(nll_firstaction_oa_depths[i_subj, i_mblk, :], m_prob_oa[0][i_mblk, i_subj, :])            

            nll_hinoise_all_oa[i_subj, :] = np.matmul(nll_firstaction_oa_depths[i_subj, :, :].transpose(), conditions_oa[0][i_subj, :].numpy()) # Sum of NLL for high noise (noise==1) 
            nll_lonoise_all_oa[i_subj, :] = np.matmul(nll_firstaction_oa_depths[i_subj, :, :].transpose(), 1 - conditions_oa[0][i_subj, :].numpy()) # Sum of NLL for low noise (noise==0)
            nll_all_oa[i_subj, :] = nll_hinoise_all_oa[i_subj, :] + nll_lonoise_all_oa[i_subj, :]
            nll_hinoise_120_oa[i_subj, :] = np.matmul(nll_firstaction_oa_depths[i_subj, 20:, :].transpose(), conditions_oa[0][i_subj, 20:].numpy()) # Exclude 20 training miniblocks
            nll_lonoise_120_oa[i_subj, :] = np.matmul(nll_firstaction_oa_depths[i_subj, 20:, :].transpose(), 1 - conditions_oa[0][i_subj, 20:].numpy()) 
            nll_120_oa[i_subj, :] = nll_hinoise_120_oa[i_subj, :] + nll_lonoise_120_oa[i_subj, :]

            nll_hinoise_120_mean_oa[i_subj] = np.matmul(nll_firstaction_oa_mean[i_subj, 20:].transpose(), conditions_oa[0][i_subj, 20:].numpy()) # Exclude 20 training miniblocks
            nll_lonoise_120_mean_oa[i_subj] = np.matmul(nll_firstaction_oa_mean[i_subj, 20:].transpose(), 1 - conditions_oa[0][i_subj, 20:].numpy()) 
            nll_120_mean_oa[i_subj] = nll_hinoise_120_mean_oa[i_subj] + nll_lonoise_120_mean_oa[i_subj]

            nll_random_120 = -np.log(0.5)*120
            nll_random_60 = -np.log(0.5)*60    
            pseudo_rsquare_120_mean_oa[i_subj] = 1 - (nll_120_mean_oa[i_subj] / nll_random_120)
            pseudo_rsquare_hinoise_120_mean_oa[i_subj] = 1 - (nll_hinoise_120_mean_oa[i_subj] / nll_random_60)    
            pseudo_rsquare_lonoise_120_mean_oa[i_subj] = 1 - (nll_lonoise_120_mean_oa[i_subj] / nll_random_60)        

            BIC_120_mean_oa[i_subj] = 2*nll_120_mean_oa[i_subj] + m_param_count*np.log(120)
            BIC_hinoise_120_oa[i_subj] = 2*nll_hinoise_120_mean_oa[i_subj] + m_param_count*np.log(60)
            BIC_lonoise_120_oa[i_subj] = 2*nll_lonoise_120_mean_oa[i_subj] + m_param_count*np.log(60)            


    # Reference values for random behavior: -np.log(0.5)*140 = 97.04 (all miniblocks), -np.log(0.5)*70 = 48.52 (70 miniblocks)


n_subj_ya = len(responses_ya)    
nll_firstaction_ya_mean = np.nan * np.ones([n_subj_ya, 140])    
nll_firstaction_ya_depths = np.nan * np.ones([n_subj_ya, 140, 3])
nll_hinoise_all_ya = np.nan * np.ones([n_subj_ya, 3])
nll_lonoise_all_ya = np.nan * np.ones([n_subj_ya, 3])
nll_all_ya = np.nan * np.ones([n_subj_ya, 3])
nll_hinoise_120_ya = np.nan * np.ones([n_subj_ya, 3])
nll_lonoise_120_ya = np.nan * np.ones([n_subj_ya, 3])
nll_120_ya = np.nan * np.ones([n_subj_ya, 3])
nll_hinoise_120_mean_ya = np.nan * np.ones([n_subj_ya])
nll_lonoise_120_mean_ya = np.nan * np.ones([n_subj_ya])
nll_120_mean_ya = np.nan * np.ones([n_subj_ya])
pseudo_rsquare_120_mean_ya = np.nan * np.ones([n_subj_ya])
pseudo_rsquare_hinoise_120_mean_ya = np.nan * np.ones([n_subj_ya])
pseudo_rsquare_lonoise_120_mean_ya = np.nan * np.ones([n_subj_ya])
BIC_120_mean_ya = np.nan * np.ones([n_subj_ya])
BIC_hinoise_120_ya = np.nan * np.ones([n_subj_ya])
BIC_lonoise_120_ya = np.nan * np.ones([n_subj_ya])

for i_mblk in range(140):
    for i_action in range(1): # Consider only first action; use range(3) to consider all action steps
        # CAUTION: If there are as many logit values as particles (e.g., 100), we have to extract the mean rather than index 0!
        logits_depths = infer_ya.agent.logits[3*i_mblk + i_action].detach().numpy()#.mean(0) # agent.logits = Value difference between Jump and Move             
        
        for i_subj in range(n_subj_ya):
            resp = infer_ya.responses[i_subj][i_mblk].numpy()[i_action] 

            p_jump_depths = 1.0 / (1 + np.exp(-logits_depths[i_subj])) # softmax function for choice probabilities
            p_move_depths = 1 - p_jump_depths
            if resp==1:
                nll = -np.log(p_jump_depths) # Apply -np.log() to obtain the negative log-likelihood (nll), for numerical reasons
            elif resp==0:
                nll = -np.log(p_move_depths)                              
            nll_firstaction_ya_depths[i_subj, i_mblk, :] = nll              
            #nll_firstaction_ya_mean[i_subj, i_mblk] = np.matmul(nll_firstaction_ya_depths[i_subj, i_mblk, :], m_prob_ya[0][i_mblk,0,:]) # WRONG - m_prob_ya[0] has shape (n_miniblocks, n_subjects, n_steps !!!
            nll_firstaction_ya_mean[i_subj, i_mblk] = np.matmul(nll_firstaction_ya_depths[i_subj, i_mblk, :], m_prob_ya[0][i_mblk, i_subj, :])            

            nll_hinoise_all_ya[i_subj, :] = np.matmul(nll_firstaction_ya_depths[i_subj, :, :].transpose(), conditions_ya[0][i_subj, :].numpy()) # Sum of NLL for high noise (noise==1) 
            nll_lonoise_all_ya[i_subj, :] = np.matmul(nll_firstaction_ya_depths[i_subj, :, :].transpose(), 1 - conditions_ya[0][i_subj, :].numpy()) # Sum of NLL for low noise (noise==0)
            nll_all_ya[i_subj, :] = nll_hinoise_all_ya[i_subj, :] + nll_lonoise_all_ya[i_subj, :]
            nll_hinoise_120_ya[i_subj, :] = np.matmul(nll_firstaction_ya_depths[i_subj, 20:, :].transpose(), conditions_ya[0][i_subj, 20:].numpy()) # Exclude 20 training miniblocks
            nll_lonoise_120_ya[i_subj, :] = np.matmul(nll_firstaction_ya_depths[i_subj, 20:, :].transpose(), 1 - conditions_ya[0][i_subj, 20:].numpy()) 
            nll_120_ya[i_subj, :] = nll_hinoise_120_ya[i_subj, :] + nll_lonoise_120_ya[i_subj, :]

            nll_hinoise_120_mean_ya[i_subj] = np.matmul(nll_firstaction_ya_mean[i_subj, 20:].transpose(), conditions_ya[0][i_subj, 20:].numpy()) # Exclude 20 training miniblocks
            nll_lonoise_120_mean_ya[i_subj] = np.matmul(nll_firstaction_ya_mean[i_subj, 20:].transpose(), 1 - conditions_ya[0][i_subj, 20:].numpy()) 
            nll_120_mean_ya[i_subj] = nll_hinoise_120_mean_ya[i_subj] + nll_lonoise_120_mean_ya[i_subj]

            nll_random_120 = -np.log(0.5)*120
            nll_random_60 = -np.log(0.5)*60    
            pseudo_rsquare_120_mean_ya[i_subj] = 1 - (nll_120_mean_ya[i_subj] / nll_random_120)
            pseudo_rsquare_hinoise_120_mean_ya[i_subj] = 1 - (nll_hinoise_120_mean_ya[i_subj] / nll_random_60)    
            pseudo_rsquare_lonoise_120_mean_ya[i_subj] = 1 - (nll_lonoise_120_mean_ya[i_subj] / nll_random_60)     

            BIC_120_mean_ya[i_subj] = 2*nll_120_mean_ya[i_subj] + m_param_count*np.log(120)
            BIC_hinoise_120_ya[i_subj] = 2*nll_hinoise_120_mean_ya[i_subj] + m_param_count*np.log(60)
            BIC_lonoise_120_ya[i_subj] = 2*nll_lonoise_120_mean_ya[i_subj] + m_param_count*np.log(60)   

# Write ELBO bound (approximate value of the negative marginal log likelihood)
pd.DataFrame(infer_ya.loss).to_csv(localpath + '/ELBO_ya_group_discount_hyperbolic_theta_alpha_kmax30.csv') #     
pd.DataFrame(infer_oa.loss).to_csv(localpath + '/ELBO_oa_group_discount_hyperbolic_theta_alpha_kmax30.csv') # 

# Write neg. log-likelihood / fit values:
dict_nll_oa = {}
dict_nll_oa['nll_1staction_120_mean'] = nll_120_mean_oa
dict_nll_oa['nll_1staction_hinoise_120_mean'] = nll_hinoise_120_mean_oa
dict_nll_oa['nll_1staction_lonoise_120_mean'] = nll_lonoise_120_mean_oa    
dict_nll_oa['BIC_120_mean'] = BIC_120_mean_oa
dict_nll_oa['BIC_hinoise_120_mean'] = BIC_hinoise_120_oa
dict_nll_oa['BIC_lonoise_120_mean'] = BIC_lonoise_120_oa
dict_nll_oa['pseudo_Rsquare_1staction_120_mean'] = pseudo_rsquare_120_mean_oa
dict_nll_oa['pseudo_Rsquare_1staction_hinoise_120_mean'] = pseudo_rsquare_hinoise_120_mean_oa
dict_nll_oa['pseudo_Rsquare_1staction_lonoise_120_mean'] = pseudo_rsquare_lonoise_120_mean_oa   
dict_nll_oa['ID'] = ids_oa
df_nll_oa = pd.DataFrame(data=dict_nll_oa)
df_nll_oa.to_csv(localpath + '/NLL_oa_group_discount_hyperbolic_theta_alpha_kmax30.csv') 


dict_nll_ya = {}
dict_nll_ya['nll_1staction_120_mean'] = nll_120_mean_ya
dict_nll_ya['nll_1staction_hinoise_120_mean'] = nll_hinoise_120_mean_ya
dict_nll_ya['nll_1staction_lonoise_120_mean'] = nll_lonoise_120_mean_ya     
dict_nll_ya['BIC_120_mean'] = BIC_120_mean_ya
dict_nll_ya['BIC_hinoise_120_mean'] = BIC_hinoise_120_ya
dict_nll_ya['BIC_lonoise_120_mean'] = BIC_lonoise_120_ya
dict_nll_ya['pseudo_Rsquare_1staction_120_mean'] = pseudo_rsquare_120_mean_ya
dict_nll_ya['pseudo_Rsquare_1staction_hinoise_120_mean'] = pseudo_rsquare_hinoise_120_mean_ya
dict_nll_ya['pseudo_Rsquare_1staction_lonoise_120_mean'] = pseudo_rsquare_lonoise_120_mean_ya     
dict_nll_ya['ID'] = ids_ya
df_nll_ya = pd.DataFrame(data=dict_nll_ya)
df_nll_ya.to_csv(localpath + '/NLL_ya_group_discount_hyperbolic_theta_alpha_kmax30.csv') 



# get exceedance probabilities of the 3 planning depths (rows = mini block, column = participant)
# PD 1
exc_count_oa_1 = exc_count_oa[0] [:] [:]
exc_count_ya_1 = exc_count_ya[0] [:] [:]


exc_count_oa_PD1 = exc_count_oa_1[0] [:] [:]
import pandas as pd
pd.DataFrame(exc_count_oa_PD1).to_csv(localpath+"/exc_PD1_oa_discount_hyperbolic_theta_alpha_kmax30.csv")
file.close()

exc_count_ya_PD1 = exc_count_ya_1[0] [:] [:]
pd.DataFrame(exc_count_ya_PD1).to_csv(localpath+"/exc_PD1_ya_discount_hyperbolic_theta_alpha_kmax30.csv")
file.close()

# PD 2
exc_count_oa_PD2 = exc_count_oa_1[1] [:] [:]
pd.DataFrame(exc_count_oa_PD2).to_csv(localpath+"/exc_PD2_oa_discount_hyperbolic_theta_alpha_kmax30.csv")
file.close()

exc_count_ya_PD2 = exc_count_ya_1[1] [:] [:]
pd.DataFrame(exc_count_ya_PD2).to_csv(localpath+"/exc_PD2_ya_discount_hyperbolic_theta_alpha_kmax30.csv")
file.close()

# PD 3
exc_count_oa_PD3 = exc_count_oa_1[2] [:] [:]
import pandas as pd
pd.DataFrame(exc_count_oa_PD3).to_csv(localpath+"/exc_PD3_oa_discount_hyperbolic_theta_alpha_kmax30.csv")
file.close()

exc_count_ya_PD3 = exc_count_ya_1[2] [:] [:]
import pandas as pd
pd.DataFrame(exc_count_ya_PD3).to_csv(localpath+"/exc_PD3_ya_discount_hyperbolic_theta_alpha_kmax30.csv")
file.close()



