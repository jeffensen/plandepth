#!/usr/bin/env python
# coding: utf-8
"""
@author: Lorenz Goenner
"""
# In[1]:

import torch
from scipy import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = 'talk', style = 'white')
sns.set_palette("colorblind", n_colors=5, color_codes=True)

from os.path import expanduser, isdir, join
from os import listdir, walk
from helper_files import load_and_format_behavioural_data, get_posterior_stats, errorplot
import sys
sys.path.append('../')


from agents_discount_hyperbolic_theta_lowprob_pruning_kmax30 import BackInductionDiscountHyperbolicThetaLowProbPruningkmax30
from inference import Inferrer
from calc_BIC import calc_BIC

# In[2]:

def variational_inference(stimuli, mask, responses):
    """
    

    Parameters
    ----------
    stimuli : dict
        task conditions (high/low noise, n_subjects, n_miniblocks), 
        planet configs (n_subjects, n_miniblocks, n_actions, n_positions, n_planet_types), 
        states (n_subjects, n_miniblocks, n_states (start, action1, action2, action3)).
    mask : Tensor
        masked invalid trials (missing data; (n_subjects, n_miniblocks, n_actions)).
    responses : Tensor
        action choices in task (n_subjects, n_miniblocks, n_actions).

    Returns
    -------
    infer : inferrer
        DESCRIPTION.

    """
    max_depth = 3
    runs, mini_blocks, max_trials = responses.shape
    
    confs = stimuli['configs']
    
    # define agent
    agent = BackInductionDiscountHyperbolicThetaLowProbPruningkmax30(confs,
                          runs=runs,
                          mini_blocks=mini_blocks,
                          trials=max_trials,
                          costs = torch.tensor([0,0]),                         
                          planning_depth=max_depth)

    # load inference module and start model fitting
    infer = Inferrer(agent, stimuli, responses, mask)
    infer.fit(num_iterations=1000, 
              num_particles=100, 
              optim_kwargs={'lr': .010}) # lr: Learning rate
    
    return infer

# In[3]:

# load and format behavioural data
path1 = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/LOG_Files/full_datasets_OA/' 
path2 = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/LOG_Files/full_datasets_YA/'
localpath = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Results_lowprob_pruning_discount_hyperbolic_theta_kmax30_test1' 

filenames = ["space_adventure_pd-results.json",
             "space_adventure_pd_inv-results.json"]

stimuli_oa, mask_oa, responses_oa, conditions_oa, ids_oa = load_and_format_behavioural_data(path1, filenames)
stimuli_ya, mask_ya, responses_ya, conditions_ya, ids_ya = load_and_format_behavioural_data(path2, filenames)


# In[4]:

# sample from posterior
def format_posterior_samples(infer):
    labels = [r'$\tilde{\beta}$', r'$\theta$', r'$k$']
    pars_df, mg_df, sg_df = infer.sample_from_posterior(labels)
    
    # transform sampled parameter values to the true parameter range
    pars_df[r'$\beta$'] = torch.from_numpy(pars_df[r'$\tilde{\beta}$'].values).exp().numpy()
    pars_df[r'$k$'] = 30*torch.from_numpy(pars_df[r'$k$'].values).sigmoid().numpy()        

    pars_df.drop([r'$\tilde{\beta}$'], axis=1, inplace=True) 
    return pars_df.melt(id_vars=['subject'], var_name='parameter')


# In[5]:

# Variational inference
infer_oa = variational_inference(stimuli_oa, mask_oa, responses_oa)
pars_df_oa = format_posterior_samples(infer_oa)

n_samples = 100
post_marg_oa = infer_oa.sample_posterior_marginal(n_samples=n_samples)
pars_df_oa['IDs'] = np.array(ids_oa)[pars_df_oa.subject.values - 1]

# plot convergence of ELBO bound (approximate value of the negative marginal log likelihood)
fig, axes = plt.subplots(1, 1, figsize=(15, 5))
axes.plot(infer_oa.loss[100:])  # 100:
df_loss = pd.DataFrame(infer_oa.loss)    
axes.plot(range(len(df_loss[0].rolling(window=25).mean())-100), 
          df_loss[0].rolling(window=25).mean()[100:], lw=1)                   
axes.set_title('ELBO Testdaten')
fig.savefig(localpath + '/ELBO Testdaten_oa_lowprob_pruning_discount_hyperbolic_theta_kmax30.jpg')

g = sns.FacetGrid(pars_df_oa, col="parameter", height=5, sharey=False);
g = g.map(errorplot, 'subject', 'value').add_legend();
#g.axes[0,0].set_ylim([-1, 2]) # adapt axes

g.fig.savefig(localpath + '/parameter_participant_oa_lowprob_pruning_discount_hyperbolic_theta_kmax30.jpg')

# In[6]:

infer_ya = variational_inference(stimuli_ya, mask_ya, responses_ya)
pars_df_ya = format_posterior_samples(infer_ya)

n_samples = 100
post_marg_ya = infer_ya.sample_posterior_marginal(n_samples=n_samples)
pars_df_ya['IDs'] = np.array(ids_ya)[pars_df_ya.subject.values - 1]

# plot convergence of ELBO bound (approximate value of the negative marginal log likelihood)

fig, axes = plt.subplots(1, 1, figsize=(15, 5))
axes.plot(infer_ya.loss[100:])  # 100:
df_loss = pd.DataFrame(infer_ya.loss)    
axes.plot(range(len(df_loss[0].rolling(window=25).mean())-100),
          df_loss[0].rolling(window=25).mean()[100:], lw=1)                     
axes.set_title('ELBO Testdaten')
fig.savefig(localpath + '/ELBO Testdaten_ya_lowprob_pruning_discount_hyperbolic_theta_kmax30.jpg')

# visualize posterior parameter estimates over subjects
g = sns.FacetGrid(pars_df_ya, col="parameter", height=5, sharey=False);
g = g.map(errorplot, 'subject', 'value').add_legend();
#g.axes[0,0].set_ylim([-1, 2]) # adapt axes
g.fig.savefig(localpath + '/parameter_participant_ya_lowprob_pruning_discount_hyperbolic_theta_kmax30.jpg')

# plot posterior distribution over groups
pars_df_oa['group'] = 'OA'
pars_df_ya['group'] = 'YA'

pars_df = pd.concat([pars_df_oa, pars_df_ya], ignore_index=True)

g = sns.FacetGrid(pars_df, col="parameter", hue='group', height=5, sharey=False, sharex=False, palette='Set1');
g = g.map(sns.kdeplot, 'value').add_legend();
g.fig.savefig(localpath + '/post_group_parameters_OA_YA_lowprob_pruning_discount_hyperbolic_theta_kmax30.pdf', dpi=300)

pars_df = pd.concat([pars_df_oa, pars_df_ya], ignore_index=True)
pars_df.to_csv(localpath + '/pars_post_samples_lowprob_pruning_discount_hyperbolic_theta_kmax30.csv')


# In[7]: Compute the posterior marginal over planning depth, 
# compute exceedanace probability and plot the results for individual subjects and for the group level results.

post_depth_oa, m_prob_oa, exc_count_oa = get_posterior_stats(post_marg_oa)
np.savez(localpath + '/oa_plandepth_stats_B03_lowprob_pruning_discount_hyperbolic_theta_kmax30', post_depth_oa, m_prob_oa, exc_count_oa)

post_depth_ya, m_prob_ya, exc_count_ya = get_posterior_stats(post_marg_ya)
np.savez(localpath + '/ya_plandepth_stats_B03_lowprob_pruning_discount_hyperbolic_theta_kmax30', post_depth_ya, m_prob_ya, exc_count_ya)



file = b = np.load(localpath + '/oa_plandepth_stats_B03_lowprob_pruning_discount_hyperbolic_theta_kmax30.npz', allow_pickle=True)
fs = file.files 
post_meanPD_firstAction_oa = np.matmul(file[fs[1]][0,:,:,:], np.arange(1,4))

dict_ids_oa={}
dict_ids_oa['ID'] = ids_oa
pd.DataFrame(dict_ids_oa).to_csv(localpath + '/IDs_OA.csv')
df_oa_meanPD = pd.DataFrame(post_meanPD_firstAction_oa) # 
pd.DataFrame(df_oa_meanPD).to_csv(localpath + '/meanPD_1st_action_oa_single_lowprob_pruning_discount_hyperbolic_theta_kmax30.csv')
file.close()

file = b = np.load(localpath + '/ya_plandepth_stats_B03_lowprob_pruning_discount_hyperbolic_theta_kmax30.npz', allow_pickle=True)
fs = file.files # names of the stored arrays (['post_depth_ya', 'm_prob_ya', 'exc_count_ya'])
post_meanPD_firstAction_ya = np.matmul(file[fs[1]][0,:,:,:], np.arange(1,4))

df_ya_meanPD = pd.DataFrame(post_meanPD_firstAction_ya) # 
pd.DataFrame(df_ya_meanPD).to_csv(localpath + '/meanPD_1st_action_ya_single_lowprob_pruning_discount_hyperbolic_theta_kmax30.csv')
dict_ids_ya={}
dict_ids_ya['ID'] = ids_ya
pd.DataFrame(dict_ids_ya).to_csv(localpath + '/IDs_YA.csv')
file.close()
# In[8]: Compute measures for model comparison 

nll_120_mean_oa, nll_hinoise_120_mean_oa, nll_lonoise_120_mean_oa, \
pseudo_rsquare_120_mean_oa, BIC_120_mean_oa, \
pseudo_rsquare_hinoise_120_mean_oa, BIC_hinoise_120_oa, \
pseudo_rsquare_lonoise_120_mean_oa, BIC_lonoise_120_oa = calc_BIC(infer_oa,
                                                                  responses_oa,
                                                                  conditions_oa,
                                                                  m_prob_oa) 
       
pd.DataFrame(infer_oa.loss).to_csv(localpath + '/ELBO_oa_group_lowprob_pruning_discount_hyperbolic_theta_kmax30.csv') # 
#
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
df_nll_oa.to_csv(localpath + '/NLL_oa_group_lowprob_pruning_discount_hyperbolic_theta_kmax30.csv') 

nll_120_mean_ya, nll_hinoise_120_mean_ya, nll_lonoise_120_mean_ya, \
pseudo_rsquare_120_mean_ya, BIC_120_mean_ya, \
pseudo_rsquare_hinoise_120_mean_ya, BIC_hinoise_120_ya, \
pseudo_rsquare_lonoise_120_mean_ya, BIC_lonoise_120_ya = calc_BIC(infer_ya,
                                                                  responses_ya,
                                                                  conditions_ya,
                                                                  m_prob_ya) 
       
pd.DataFrame(infer_ya.loss).to_csv(localpath + '/ELBO_ya_group_lowprob_pruning_discount_hyperbolic_theta_kmax30.csv') # 

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
df_nll_ya.to_csv(localpath + '/NLL_ya_group_lowprob_pruning_discount_hyperbolic_theta_kmax30.csv') 




