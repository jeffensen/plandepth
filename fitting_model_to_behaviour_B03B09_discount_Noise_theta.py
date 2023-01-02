#!/usr/bin/env python
# coding: utf-8

# Here I will illustrate the steps needed to fit the behavioural model to empirical data. This entails determining the posterior beleifs over model parameters and mini-block specific planning depth.

# In[1]:

import torch

from scipy import io

import pandas as pd

from torch import zeros, ones

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context = 'talk', style = 'white')
sns.set_palette("colorblind", n_colors=5, color_codes=True)

# function for plotting asymetric errorbars
def errorplot(*args, **kwargs):
    subjects = args[0]
    values = args[1].values
    
    unique_subjects = np.unique(subjects)
    nsub = len(unique_subjects)
    
    values = values.reshape(-1, nsub)
    
    quantiles = np.percentile(values, [5, 50, 95], axis=0)
    
    low_perc = quantiles[0]
    up_perc = quantiles[-1]
    
    x = unique_subjects
    y = quantiles[1]

    assert np.all(low_perc <= y)
    assert np.all(y <= up_perc)
    
    kwargs['yerr'] = [y-low_perc, up_perc-y]
    kwargs['linestyle'] = ''
    kwargs['marker'] = 'o'
    
    plt.errorbar(x, y, **kwargs)
    
# function for mapping strings ('high', 'low') to numbers 0, 1

def map_noise_to_values(strings):
    for s in strings:
        if s[0] == 'high':
            yield 1
        elif s[0] == 'low':
            yield 0
        else:
            yield np.nan


# In[2]:


from pathlib import Path
from os.path import expanduser, isdir, join
from os import listdir, walk
import json
    
def load_and_format_behavioural_data(local_path, filenames): #kann es nicht als komplette funktion, nur wenn ich alles einzeln abspiele und die paths per Hand ergänze
    
    # search local_path and all subdirectories for files named like filename
    home = str(Path.home())    
    #path = home + local_path # muss man immer händisch machen 
    path = local_path # LG
    fnames = []
    for paths, dirs, files in walk(path): # hier kennt es immer die Filenames nicht, da sie unten erst definiert werden?
        for filename in [f for f in files if f in filenames]:
            fnames.append(join(paths, filename))  # get full paths of alle filename files
   
    # check for exclusion (gameover by negative points or incomplete data)
    for i,f in enumerate(fnames):
        read_file = open(f,"r", encoding='utf-8-sig')
        tmp = json.loads(json.load(read_file)['data']) # assume json file
        read_file.close()
        #print("tmp = ", tmp) # LG
        if all(flag <= 0 for (_, _, flag) in tmp['points']) or len(tmp['points']) != len(tmp['conditions']['noise']): fnames.remove(fnames[i])
        
    runs = len(fnames)  # number of subjects
    
    mini_blocks = 140  # number of mini blocks in each run
    max_trials = 3  # maximal number of trials within a mini block
    max_depth = 3  # maximal planning depth

    na = 2  # number of actions
    ns = 6 # number of states/locations
    no = 5 # number of outcomes/rewards

    responses = zeros(runs, mini_blocks, max_trials)
    states = zeros(runs, mini_blocks, max_trials+1, dtype=torch.long)
    scores = zeros(runs, mini_blocks, max_depth)
    conditions = zeros(2, runs, mini_blocks, dtype=torch.long)
    confs = zeros(runs, mini_blocks, 6, dtype=torch.long)
    balance_cond = zeros(runs)
    ids = []
    subj_IDs = []    
    
    for i,f in enumerate(fnames):
        read_file = open(f,"r", encoding='utf-8-sig')
        # assume json file
        tmp = json.loads(json.load(read_file)['data'])
        read_file.close()
        
        responses[i] = torch.from_numpy(np.array(tmp['responses']['actions']) -1)
        states[i] = torch.from_numpy(np.array(tmp['states']) - 1).long()
        confs[i] = torch.from_numpy(np.array(tmp['planetConfigurations']) - 1).long()
        scores[i] = torch.from_numpy(np.array(tmp['points']))
        strings = tmp['conditions']['noise']
        
        conditions[0, i] = torch.tensor(np.unique(strings, return_inverse=True)[1]*(-1) + 1 , dtype=torch.long)  # "low" -> 0 | "high" -> 1
        conditions[1, i] = torch.from_numpy(np.array(tmp['conditions']['notrials'])).long() 
        
        balance_cond[i] = tmp['balancingCondition'] - 1
        
        # here, ids are just numbered starting from one
        # better would be to include IDs in .json file
        #ids.append(i+1)
        
        #ID = f.split('\\')[0].split('/')[-1]
        #ids.append(ID)        
        if f.split('//')[0] != f:
            ID = f.split('\\')[0].split('/')[-1]
        else:
            ID = f.split('/')[-2]
        ids.append(ID)                

    states[states < 0] = -1
    confs = torch.eye(no)[confs]

    # define dictionary containing information which participants recieved on each trial
    stimuli = {'conditions': conditions,
               'states': states, 
               'configs': confs}

    mask = ~torch.isnan(responses)
    
    return stimuli, mask, responses, conditions, ids


# Probabilistic inference
import sys
sys.path.append('../')

from agents_discount_Noise_theta import BackInductionDiscountNoiseTheta
from inference import Inferrer

def variational_inference(stimuli, mask, responses):
    max_depth = 3
    runs, mini_blocks, max_trials = responses.shape
    
    confs = stimuli['configs']
    
    # define agent
    agent = BackInductionDiscountNoiseTheta(confs,
                          runs=runs,
                          mini_blocks=mini_blocks,
                          trials=max_trials,
                          costs = torch.tensor([0,0]), # LG: Forgot to set to zero - corrected on 2022-06-27!!!                          
                          planning_depth=max_depth)

    # load inference module and start model fitting
    infer = Inferrer(agent, stimuli, responses, mask)
    infer.fit(num_iterations=1000, num_particles=100, optim_kwargs={'lr': .010}) # 1000 # lr=.1
    
    return infer


# In[3]:

# load and format behavioural data
path1 =  "/Dokumente/plandepth/Experimental_Data/OA_xtra"   # change to correct path
path2 =  "/Dokumente/plandepth/Experimental_Data/YA_xtra" 
path1 = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/LOG_Files/full_datasets_OA/'
path2 = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/LOG_Files/full_datasets_YA/'
localpath = 'H:/Sesyn/TRR265-B09/Analysis_SAT-PD2_Sophia/SAT_PD_Inference_Scripts' # LG

filenames = ["space_adventure_pd-results.json",
             "space_adventure_pd_inv-results.json"]    # posible filenames of SAT logfiles

stimuli_oa, mask_oa, responses_oa, conditions_oa, ids_oa = load_and_format_behavioural_data(path1, filenames)
stimuli_ya, mask_ya, responses_ya, conditions_ya, ids_ya = load_and_format_behavioural_data(path2, filenames)


        # In[6]:


# sample from posterior
def format_posterior_samples(infer):
    labels = [r'$\tilde{\beta}$', r'$\theta$', r'$\gamma$']
    pars_df, mg_df, sg_df = infer.sample_from_posterior(labels)
    
    # transform sampled parameter values to the true parameter range
    pars_df[r'$\beta$'] = torch.from_numpy(pars_df[r'$\tilde{\beta}$'].values).exp().numpy()
    pars_df[r'$\gamma$'] = torch.from_numpy(pars_df[r'$\gamma$'].values).sigmoid().numpy()

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
fig.savefig('ELBO Testdaten_oa_lossaversion_discount_Noise_theta_0cost.jpg')
#fig.savefig('ELBO Testdaten_oa_discount_hiLoNoise_500-1000.jpg')


g = sns.FacetGrid(pars_df_oa, col="parameter", height=5, sharey=False);
g = g.map(errorplot, 'subject', 'value').add_legend();
#g.axes[0,0].set_ylim([-1, 2])
#g.axes[0,1].set_ylim([0, 7])
#g.axes[0,2].set_ylim([0, 1])

g.fig.savefig('parameter_participant_oa_lossaversion_discount_Noise_theta_0cost.jpg')

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
fig.savefig('ELBO Testdaten_ya_lossaversion_discount_Noise_theta_0cost.jpg')
#fig.savefig('ELBO Testdaten_ya_discount_hiLoNoise_500-1000.jpg')

# In[8]:


# visualize posterior parameter estimates over subjects


# visualize posterior parameter estimates over subjects

g = sns.FacetGrid(pars_df_ya, col="parameter", height=5, sharey=False);
g = g.map(errorplot, 'subject', 'value').add_legend();
#g.axes[0,0].set_ylim([-1, 2])
#g.axes[0,1].set_ylim([0, 7])
#g.axes[0,2].set_ylim([0, 1])

g.fig.savefig('parameter_participant_ya_lossaversion_discount_Noise_theta_0cost.jpg')

# In[9]:


# plot posterior distribution over groups
pars_df_oa['group'] = 'OA'
pars_df_ya['group'] = 'YA'

pars_df = pars_df_oa.append(pars_df_ya, ignore_index=True)

g = sns.FacetGrid(pars_df, col="parameter", hue='group', height=5, sharey=False, sharex=False, palette='Set1');
g = g.map(sns.kdeplot, 'value').add_legend();
g.fig.savefig('post_group_parameters_OA_YA_lossaversion_discount_Noise_theta_0cost.pdf', dpi=300)


# In[10]:


pars_df = pars_df_oa.append(pars_df_ya)
pars_df.to_csv('pars_post_samples_lossaversion_discount_Noise_theta_0cost.csv')


# In[11]:


# def logit(x):
#     return -np.log(1/x - 1)

# ns_oa = len(np.unique(pars_df_oa.subject.values))
# ns_ya = len(np.unique(pars_df_ya.subject.values))

# pdf_oa = pars_df_oa.copy()

# pdf_ya = pars_df_ya.copy()
# pdf_ya['sample'] = np.broadcast_to(np.arange(1000)[:, None], (1000, 3 * ns_ya)).reshape(-1)

# diff = {}
# for g1, g2 in zip(pdf_oa.groupby('parameter'), pdf_ya.groupby('parameter')):
#     g_oa = g1[1].copy()
#     g_oa['sample'] = np.broadcast_to(np.arange(1000)[:, None], (1000, ns_oa)).reshape(-1)
    
#     g_ya = g2[1].copy()
#     g_ya['sample'] = np.broadcast_to(np.arange(1000)[:, None], (1000, ns_ya)).reshape(-1)

#     smpl_oa = g_oa.pivot(index='sample', columns='subject', values='value')
#     smpl_ya = g_ya.pivot(index='sample', columns='subject', values='value')
    
#     lbl = g1[0]
#     if lbl == r'$\beta$':
#         lbl = r'$\ln \beta$'
#         smpl_oa = smpl_oa.transform(np.log)
#         smpl_ya = smpl_ya.transform(np.log)
#     elif lbl == r'$\alpha$':
#         lbl = r'$logit(\alpha)$'
#         smpl_oa = smpl_oa.transform(logit)
#         smpl_ya = smpl_ya.transform(logit)
        
#     t_score = []
#     p_value = []
#     for s1, s2 in zip(smpl_oa.values, smpl_ya.values):
#         res = ttest_ind(s1, s2)

#         t = res[0]
#         p = res[1]
#         t_score.append(t)
#         p_value.append(p)
        
#     print(lbl, 't-test stats =', np.mean(t_score), 'p-value = ', np.mean(p_value))
#     diff[lbl] = smpl_oa.values.mean(-1) - smpl_ya.values.mean(-1)
    
# df = pd.DataFrame(diff)
# df.to_csv('params_diffs.csv')


# In what follows we will compute the posterior marginal over planning depth, compute exceedanace probability and plot the results for individual subjects and for the group level results.

# In[12]:


def get_posterior_stats(post_marg, mini_blocks=140):
    n_samples, runs, max_trials = post_marg['d_0_0'].shape
    post_depth = {0: np.zeros((n_samples, mini_blocks, runs, max_trials)),
              1: np.zeros((n_samples, mini_blocks, runs, max_trials))}
    for pm in post_marg:
        b, t = np.array(pm.split('_')[1:]).astype(int)
        if t in post_depth:
            post_depth[t][:, b] = post_marg[pm]

    # get sample mean over planning depth for the first and second choice
    m_prob = [post_depth[c].mean(0) for c in range(2)]

    # get sample plannign depth exceedance count of the first and second choice
    # exceedance count => number of times planning depth d had highest posterior probability
    exc_count = [np.array([np.sum(post_depth[t].argmax(-1) == i, 0) for i in range(3)]) for t in range(2)]
    
    return post_depth, m_prob, exc_count


post_depth_oa, m_prob_oa, exc_count_oa = get_posterior_stats(post_marg_oa)
np.savez('oa_plandepth_stats_B03_lossaversion_discount_Noise_theta_0cost', post_depth_oa, m_prob_oa, exc_count_oa)

post_depth_ya, m_prob_ya, exc_count_ya = get_posterior_stats(post_marg_ya)
np.savez('ya_plandepth_stats_B03_lossaversion_discount_Noise_theta_0cost', post_depth_ya, m_prob_ya, exc_count_ya)



#file = b = np.load('/home/sass/Dokumente/plandepth/oa_plandepth_stats_B03.npz', allow_pickle=True)
file = b = np.load(localpath + '/oa_plandepth_stats_B03_lossaversion_discount_Noise_theta_0cost.npz', allow_pickle=True)
fs = file.files # names of the stored arrays (['post_depth_oa', 'm_prob_oa', 'exc_count_oa'])
post_meanPD_firstAction_oa = np.matmul(file[fs[1]][0,:,:,:], np.arange(1,4))
import pandas as pd
#pd.DataFrame(post_meanPD_firstAction_oa).to_csv("/home/sass/Dokumente/plandepth/meanPD_1st_action_oa.csv")
pd.DataFrame(post_meanPD_firstAction_oa).to_csv(localpath + '/meanPD_1st_action_oa_single_lossaversion_discount_Noise_theta_0cost.csv')
file.close()



#file = b = np.load('/home/sass/Dokumente/plandepth/ya_plandepth_stats_B03.npz', allow_pickle=True)
file = b = np.load(localpath + '/ya_plandepth_stats_B03_lossaversion_discount_Noise_theta_0cost.npz', allow_pickle=True)
fs = file.files # names of the stored arrays (['post_depth_ya', 'm_prob_ya', 'exc_count_ya'])
post_meanPD_firstAction_ya = np.matmul(file[fs[1]][0,:,:,:], np.arange(1,4))
import pandas as pd
#pd.DataFrame(post_meanPD_firstAction_ya).to_csv("/home/sass/Dokumente/plandepth/meanPD_1st_action_ya.csv")
pd.DataFrame(post_meanPD_firstAction_ya).to_csv(localpath + '/meanPD_1st_action_ya_single_lossaversion_discount_Noise_theta_0cost.csv')
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
m_param_count = len(np.unique(pars_df_ya['parameter'])) # 
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
pd.DataFrame(infer_ya.loss).to_csv(localpath + '/ELBO_ya_group_lossaversion_discount_Noise_theta_0cost.csv') #     
pd.DataFrame(infer_oa.loss).to_csv(localpath + '/ELBO_oa_group_lossaversion_discount_Noise_theta_0cost.csv') # 

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
df_nll_oa = pd.DataFrame(data=dict_nll_oa)
df_nll_oa.to_csv('NLL_oa_group_lossaversion_discount_Noise_theta_0cost.csv') 


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
df_nll_ya = pd.DataFrame(data=dict_nll_ya)
df_nll_ya.to_csv('NLL_ya_group_lossaversion_discount_Noise_theta_0cost.csv') 



# get exceedance probabilities of the 3 planning depths (rows = mini block, column = participant)
# PD 1
exc_count_oa_1 = exc_count_oa[0] [:] [:]
exc_count_ya_1 = exc_count_ya[0] [:] [:]


exc_count_oa_PD1 = exc_count_oa_1[0] [:] [:]
import pandas as pd
pd.DataFrame(exc_count_oa_PD1).to_csv(localpath+"/exc_PD1_oa_lossaversion_discount_Noise_theta_0cost.csv")
file.close()

exc_count_ya_PD1 = exc_count_ya_1[0] [:] [:]
pd.DataFrame(exc_count_ya_PD1).to_csv(localpath+"/exc_PD1_ya_lossaversion_discount_Noise_theta_0cost.csv")
file.close()

# PD 2
exc_count_oa_PD2 = exc_count_oa_1[1] [:] [:]
pd.DataFrame(exc_count_oa_PD2).to_csv(localpath+"/exc_PD2_oa_lossaversion_discount_Noise_theta_0cost.csv")
file.close()

exc_count_ya_PD2 = exc_count_ya_1[1] [:] [:]
pd.DataFrame(exc_count_ya_PD2).to_csv(localpath+"/exc_PD2_ya_lossaversion_discount_Noise_theta_0cost.csv")
file.close()

# PD 3
exc_count_oa_PD3 = exc_count_oa_1[2] [:] [:]
import pandas as pd
pd.DataFrame(exc_count_oa_PD3).to_csv(localpath+"/exc_PD3_oa_lossaversion_discount_Noise_theta_0cost.csv")
file.close()

exc_count_ya_PD3 = exc_count_ya_1[2] [:] [:]
import pandas as pd
pd.DataFrame(exc_count_ya_PD3).to_csv(localpath+"/exc_PD3_ya_lossaversion_discount_Noise_theta_0cost.csv")
file.close()


# In[13]:


# plot true planning depth and estimated mean posterior depth of the first action for up to five subjects of group

nplots = min(len(ids_oa),5)
fig, axes = plt.subplots(1, nplots, figsize=(15, 5), sharex=True, sharey=True)

for ns, ax in zip(range(nplots), axes):
    sns.heatmap(m_prob_oa[0][:, ns], cmap='viridis', ax=ax, cbar=True)
    ax.set_xticklabels([1, 2, 3])
    ax.set_xlabel('depth')
    
axes[0].set_ylabel('mini-block index')
fig.suptitle('mean probability', fontsize=24)

# plot true planning depth and depth exceedance probability of the first action for up to five subjects of group
# plot true planning depth and estimated mean posterior depth of the first action for up to five subjects of group
fig, axes = plt.subplots(1, nplots, figsize=(15, 5), sharex=True, sharey=True)

for ns, ax in zip(range(nplots), axes):
    sns.heatmap(exc_count_oa[0][..., ns].T/n_samples, cmap='viridis', ax=ax, cbar=True)
    ax.set_xticklabels([1, 2, 3])
    ax.set_xlabel('depth')

axes[0].set_ylabel('mini-block index')
fig.suptitle('exceedance probability', fontsize=24);


# In[14]:


# plot true planning depth and estimated mean posterior depth of the second action for up to five subjects of group

nplots = min(len(ids_ya),5)
fig, axes = plt.subplots(1, nplots, figsize=(15, 5), sharex=True, sharey=True)

for ns, ax in zip(range(nplots), axes):
    sns.heatmap(m_prob_oa[1][:, ns], cmap='viridis', ax=ax, cbar=True)
    ax.set_xticklabels([1, 2, 3])
    ax.set_xlabel('depth')
    
axes[0].set_ylabel('mini-block index')
fig.suptitle('mean probability', fontsize=24)

# plot true planning depth and depth exceedance probability of the second action for up to five subjects of group
# plot true planning depth and estimated mean posterior depth of the second action for up to five subjects of group
fig, axes = plt.subplots(1, nplots, figsize=(15, 5), sharex=True, sharey=True)

for ns, ax in zip(range(nplots), axes):
    sns.heatmap(exc_count_oa[1][..., ns].T/n_samples, cmap='viridis', ax=ax, cbar=True)
    ax.set_xticklabels([1, 2, 3])
    ax.set_xlabel('depth')

axes[0].set_ylabel('mini-block index')
fig.suptitle('exceedance probability', fontsize=24);


# In[ ]:



