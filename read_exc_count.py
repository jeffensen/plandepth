# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:38:31 2023

@author: goenner
"""

import pylab as plt
import numpy as np
import pandas as pd
import torch
import sys
import scipy.io as io
import json

sys.path.append('../')
from tasks import SpaceAdventure
from agents import BackInduction
from agents_discount_Noise_theta_anchor_pruning import BackInductionDiscountNoiseThetaAnchorPruning
from simulate import Simulator

plt.plot(np.random.rand(10))


def inverse_sigmoid(x):
    # Inverse function of torch.sigmoid
    if x == 1:
        return np.inf
    elif x == 0:
        return - np.inf
    else:
        return -np.log(1.0 / x - 1)




    
torch.manual_seed(16324)
#pyro.enable_validation(True)

#sns.set(context='talk', style='white', color_codes=True)

runs0 = 100 # 100 # 40            #number of simulations 
mini_blocks0 = 120     #+20 for training which will be removed in the following
max_trials0 = 3        #maximum number of actions per mini-block
max_depth0 = 3         #maximum planning depth
na0 = 2                #number of actions
ns0 = 6                #number of states
no0 = 5                #number of outcomes
starting_points = 350  #number of points at the beginning of task

# load task configuration file 
#read_file = open('/home/sass/Dokumente/plandepth/config_file/space_adventure_pd_config_task_new_orig.json',"r")
read_file = open('config_file/space_adventure_pd_config_task_new_orig.json',"r")
exp1 = json.load(read_file)
read_file.close()

# load starting positions of each mini-block 
starts0 = exp1['startsExp']
import numpy
starts0 = numpy.asarray(starts0)
starts0 = starts0[20:140] # [19:139]
starts0 = starts0 -1

# load planet configurations for each mini-block
planets0 = exp1['planetsExp']
planets0 = numpy.asarray(planets0)
planets0 = planets0[20:140] # [19:139,:]
planets0 = planets0 -1


vect0 = np.eye(5)[planets0]

ol0 = torch.from_numpy(vect0)

starts0 = torch.from_numpy(starts0)


# load noise condition (low -> 0, high -> 1)
noise0 = exp1['conditionsExp']['noise']
for i in range(len(noise0)):
  if noise0[i] == 'high':
    noise0[i] = 1

for i in range(len(noise0)):
  if noise0[i] == 'low':
    noise0[i] = 0
    
noise0 = numpy.asarray(noise0)
noise0 = noise0[20:140] # [19:139]

# number of actions for each mini-block 
trials0 = exp1['conditionsExp']['notrials']
trials0 = numpy.asarray(trials0)
trials0 = trials0[20:140] # [19:139]

# load action costs (all zero)
costs0 = numpy.asarray(exp1['actionCost'])
costs0 = torch.FloatTensor(costs0)

# load fuel rewards/punishment for each planet type [-20,-10,0,10,20]
fuel0 = numpy.asarray(exp1['planetRewards'])
fuel0 = torch.FloatTensor(fuel0)  


confs0 = ol0.repeat(runs0,1,1,1).float()

starts0 = starts0.repeat(runs0,1).float()

# build tensors for conditions described by number of actions and noise condition
conditions0 = torch.zeros(2, runs0, mini_blocks0, dtype=torch.long)
conditions0[0] = torch.tensor(noise0, dtype=torch.long)[None, :]
conditions0[1] = torch.tensor(trials0, dtype=torch.long)




'''#
datapath = 'H:\Sesyn\TRR265-B09\Analysis_SAT-PD2_Sophia\SAT_PD_Inference_Scripts\Results_inference_group_NLL_0cost_1000it_lr0-01_20220706'
modelname = 'rational'
df_exc_pd1_oa = pd.read_csv(datapath + '\exc_PD1_oa_0cost.csv')
df_exc_pd2_oa = pd.read_csv(datapath + '\exc_PD2_oa_0cost.csv')
df_exc_pd3_oa = pd.read_csv(datapath + '\exc_PD3_oa_0cost.csv')
df_params_all = pd.read_csv(datapath + '/pars_post_samples_0cost.csv')
file_oa = np.load(datapath + '/oa_plandepth_stats_B03.npz', allow_pickle=True)
file_ya = np.load(datapath + '/ya_plandepth_stats_B03.npz', allow_pickle=True)
'''

'''#
datapath = 'H:\Sesyn\TRR265-B09\Analysis_SAT-PD2_Sophia\SAT_PD_Inference_Scripts\Results_inference_group_discount_Noise_theta_0cost_20220706'
modelname = 'discount_Noise_theta'
df_exc_pd1_oa = pd.read_csv(datapath + '\exc_PD1_oa_0cost.csv')
df_exc_pd2_oa = pd.read_csv(datapath + '\exc_PD2_oa_0cost.csv')
df_exc_pd3_oa = pd.read_csv(datapath + '\exc_PD3_oa_0cost.csv')
df_params_all = pd.read_csv(datapath + '/pars_post_samples_discount_Noise_theta_0cost.csv')
file_oa = np.load(datapath + '/oa_plandepth_stats_B03_discount_Noise_theta_0cost.npz', allow_pickle=True)  
file_ya = np.load(datapath + '/ya_plandepth_stats_B03_discount_Noise_theta_0cost.npz', allow_pickle=True)  
'''

datapath = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Results_anchor_pruning_discount_Noise_theta'
modelname = 'discount_Noise_theta_anchor_pruning'
df_params_all = pd.read_csv(datapath + '/pars_post_samples_anchor_pruning_discount_Noise_theta.csv')
file_oa = np.load(datapath + '/oa_plandepth_stats_B03_anchor_pruning_discount_Noise_theta.npz', allow_pickle=True)  
file_ya = np.load(datapath + '/ya_plandepth_stats_B03_anchor_pruning_discount_Noise_theta.npz', allow_pickle=True)  

fs = file_oa.files # names of the stored arrays (['post_depth_oa', 'm_prob_oa', 'exc_count_oa'])
post_meanPD_firstAction_oa = np.matmul(file_oa[fs[1]][0,:,:,:], np.arange(1,4))
exc_count_firstAction_oa = file_oa[fs[2]][0,:,:,:]
pd_map_estimate_oa = np.argmax(file_oa[fs[2]][0,:,:,:], 0)
pd_posterior_probs_oa = file_oa[fs[1]][0,:,:,:]
file_oa.close()


fs = file_ya.files # names of the stored arrays (['post_depth_ya', 'm_prob_ya', 'exc_count_ya'])
post_meanPD_firstAction_ya = np.matmul(file_ya[fs[1]][0,:,:,:], np.arange(1,4))
exc_count_firstAction_ya = file_ya[fs[2]][0,:,:,:]
pd_map_estimate_ya = np.argmax(file_ya[fs[2]][0,:,:,:], 0)
pd_posterior_probs_ya = file_ya[fs[1]][0,:,:,:]
file_ya.close()




id_list_oa = np.unique(df_params_all['IDs'][df_params_all['group']=='OA'])
id_list_ya = np.unique(df_params_all['IDs'][df_params_all['group']=='YA'])

max_group_size = max(len(id_list_oa), len(id_list_ya))
n_groups = 2
i_group_oa = 0
i_group_ya = 1

mean_beta = np.nan * np.ones([n_groups, max_group_size])
mean_theta = np.nan * np.ones([n_groups, max_group_size])
mean_gamma = np.nan * np.ones([n_groups, max_group_size])
mean_alpha = np.nan * np.ones([n_groups, max_group_size])

mean_beta_oa = np.nan * np.ones(len(id_list_oa))
mean_theta_oa = np.nan * np.ones(len(id_list_oa))
mean_gamma_oa = np.nan * np.ones(len(id_list_oa))
mean_alpha_oa = np.nan * np.ones(len(id_list_oa))


agents = []
states = []
simulations = []
performance = []

trans_pars_depth = [] # LG
points_depth = [] # LG
responses_depth = [] # LG

mean_gain_per_miniblock = np.nan * np.ones([n_groups, max_group_size, 3, mini_blocks0])
mean_gain_per_miniblock_oa = np.nan * np.ones([len(id_list_oa), 3, mini_blocks0])

mixed_agent_gain_per_miniblock_map = np.nan * np.ones([n_groups, max_group_size, mini_blocks0])
mixed_agent_gain_per_miniblock_map_oa = np.nan * np.ones([len(id_list_oa), mini_blocks0])

mixed_agent_gain_per_miniblock_meanPD = np.nan * np.ones([n_groups, max_group_size, mini_blocks0])

parameters = np.nan * np.ones([n_groups, max_group_size, 3])
parameters_oa = np.nan * np.ones([len(id_list_oa), 3])



# For each subject, add a simulation of 100 (?) agents based on the inferred parameters, for 3 planning depths!
# Store gain per plannning depth for each subject.   
# Based on the MAP estimate of planning depth (alternatively, based on a mixture model),
# "mix" the gain per miniblock according to the estimated planning depth per miniblock.

# TODO:
# Repeat for YA subjects    

for i_group in range(n_groups):
    if i_group == i_group_oa:
        subj_id_list = id_list_oa
    elif i_group == i_group_ya:
        subj_id_list = id_list_ya        
    
    for i_subj in range(len(subj_id_list)):
        # 1. set simulation parameters    
        # 2. simulate
        # 3. calculate "mixed" gain
    
        print("Subject ", subj_id_list[i_subj])  

        df_params_all['parameter'][df_params_all['IDs']==subj_id_list[i_subj]]    
        filtered_ind_beta = np.where( (df_params_all['IDs']  == subj_id_list[i_subj]) & (df_params_all['parameter'] == "$\\beta$" ))[0]
        filtered_ind_theta = np.where( (df_params_all['IDs']  == subj_id_list[i_subj]) & (df_params_all['parameter'] == "$\\theta$" ))[0]    
        filtered_ind_gamma = np.where( (df_params_all['IDs']  == subj_id_list[i_subj]) & (df_params_all['parameter'] == "$\\gamma$" ))[0]    
        filtered_ind_alpha = np.where( (df_params_all['IDs']  == subj_id_list[i_subj]) & (df_params_all['parameter'] == "$\\alpha$" ))[0]        

        mean_beta[i_group, i_subj] = df_params_all['value'][filtered_ind_beta].mean()
        mean_theta[i_group, i_subj] = df_params_all['value'][filtered_ind_theta].mean()    
        mean_gamma[i_group, i_subj] = df_params_all['value'][filtered_ind_gamma].mean()    
        mean_alpha[i_group, i_subj] = df_params_all['value'][filtered_ind_alpha].mean()        


        # Back-transformed parameters:

        #m0 = [np.log(torch.tensor(float(str(mean_beta[i_group, i_subj])))), \
        #  torch.tensor(float(str(mean_theta[i_group, i_subj]))), \
        #  inverse_sigmoid(torch.tensor(float(str(mean_alpha[i_group, i_subj]))))] # for default agent (rational)

        m0 = [np.log(torch.tensor(float(str(mean_beta[i_group, i_subj])))), \
          torch.tensor(float(str(mean_theta[i_group, i_subj]))), \
          inverse_sigmoid(torch.tensor(float(str(mean_gamma[i_group, i_subj]))))] # for discounting


        parameters[i_group, i_subj, :] = m0
        #parameters_oa[i_subj, :] = m0

        for i_pd in range(3):
        # define space adventure task with aquired configurations
        # set number of trials to the max number of actions per mini-block
            space_advent0 = SpaceAdventure(conditions0,
                                  outcome_likelihoods=confs0,
                                  init_states=starts0,
                                  runs=runs0,
                                  mini_blocks=mini_blocks0,
                                  trials=max_trials0)

            #agent0 = BackInduction(confs0,
            #                  runs=runs0,
            #                  mini_blocks=mini_blocks0,
            #                  trials=3,
            #                  costs = torch.tensor([0., 0.]), # Neu (LG)                              
            #                  planning_depth=i_pd+1) # Default agent, rational planning
            
            agent0 = BackInductionDiscountNoiseThetaAnchorPruning(confs0,
                              runs=runs0,
                              mini_blocks=mini_blocks0,
                              trials=3,
                              costs = torch.tensor([0., 0.]), # Neu (LG)                              
                              planning_depth=i_pd+1)
            
            #fixed values for parameters
            trans_pars0 = torch.tensor(m0).repeat(runs0,1) # this line sets beta, theta and alpha / gammma
            agent0.set_parameters(trans_pars0)


            # simulate behavior
            sim0 = Simulator(space_advent0,
                    agent0,
                    runs=runs0,
                    mini_blocks=mini_blocks0,
                    trials=3)
            sim0.simulate_experiment()

            simulations.append(sim0)
            agents.append(agent0)
            states.append(space_advent0.states.clone())

            responses0 = simulations[-1].responses.clone() #response actions in simulation for every mini-block 
            responses0[torch.isnan(responses0)] = -1.
            responses0 = responses0.long()
            points0 = (costs0[responses0] + fuel0[simulations[-1].outcomes])  #reward for landing on a certain planet in simulation

            points0[simulations[-1].outcomes < 0] = 0 #set MB in which points go below 0 on 0 ?
            performance.append(points0.sum(dim=-1))   #sum up the gains 
        
            mean_gain_per_miniblock[i_group, i_subj, i_pd, :] = points0.sum(dim=-1).mean(0)
            #mean_gain_per_miniblock_oa[i_subj, i_pd, :] = points0.sum(dim=-1).mean(0)
    
            #trans_pars_depth.append(trans_pars0)
            #points_depth.append(points0)
            #responses_depth.append(responses0) 
        
            if i_pd==2: 
                #print("beta = %f, Mean gain PD%i: %s" %(round(mean_beta_oa[i_subj], 2), i_pd+1, points0.sum(dim=-1).mean(0).sum()))
                print("beta = %f, Mean gain PD%i: %s" %(round(mean_beta[i_group, i_subj], 2), i_pd+1, points0.sum(dim=-1).mean(0).sum()))                
        
            # TODO: "Mix" gain of simulated agents to obtainn a reference
            for i_mb in range(0, 120):
                #print(i_mb, (mean_gain_per_miniblock_oa[i_subj, pd_map_estimate_oa[i_mb+20, i_subj], i_mb] ))
                if i_group == i_group_oa:
                    mixed_agent_gain_per_miniblock_map[i_group, i_subj, i_mb] = mean_gain_per_miniblock[i_group, i_subj, pd_map_estimate_oa[i_mb+20, i_subj], i_mb]
                    mixed_agent_gain_per_miniblock_meanPD[i_group, i_subj, i_mb] = np.average(mean_gain_per_miniblock[i_group, i_subj, :, i_mb], weights = pd_posterior_probs_oa[i_mb, i_subj, :])
                    # post_meanPD_firstAction_oa
                elif i_group == i_group_ya:                
                    mixed_agent_gain_per_miniblock_map[i_group, i_subj, i_mb] = mean_gain_per_miniblock[i_group, i_subj, pd_map_estimate_ya[i_mb+20, i_subj], i_mb]                                
                    mixed_agent_gain_per_miniblock_meanPD[i_group, i_subj, i_mb] = np.average(mean_gain_per_miniblock[i_group, i_subj, :, i_mb], weights = pd_posterior_probs_ya[i_mb, i_subj, :])                    
                #mixed_agent_gain_per_miniblock_map_oa[i_subj, i_mb] = mean_gain_per_miniblock_oa[i_subj, pd_map_estimate_oa[i_mb+20, i_subj], i_mb]
    

dict_gain_PDagents_oa = {}
dict_gain_PDagents_oa['Gain_PD1_agent'] = mean_gain_per_miniblock[i_group_oa, :, 0, :]
dict_gain_PDagents_oa['Gain_PD2_agent'] = mean_gain_per_miniblock[i_group_oa, :, 1, :]
dict_gain_PDagents_oa['Gain_PD3_agent'] = mean_gain_per_miniblock[i_group_oa, :, 2, :]
dict_gain_PDagents_oa['Parameters'] = parameters[i_group_oa, :, :]

df_gain_per_mb_pd1_oa = pd.DataFrame(data=dict_gain_PDagents_oa['Gain_PD1_agent'])
df_gain_per_mb_pd2_oa = pd.DataFrame(data=dict_gain_PDagents_oa['Gain_PD2_agent'])
df_gain_per_mb_pd3_oa = pd.DataFrame(data=dict_gain_PDagents_oa['Gain_PD3_agent'])
df_parameters_oa = pd.DataFrame(data=dict_gain_PDagents_oa['Parameters'])

df_gain_per_mb_pd1_oa.to_csv(datapath + '/simulated_gain_pd1_oa.csv')     
df_gain_per_mb_pd2_oa.to_csv(datapath + '/simulated_gain_pd2_oa.csv') 
df_gain_per_mb_pd3_oa.to_csv(datapath + '/simulated_gain_pd3_oa.csv') 
df_parameters_oa.to_csv(datapath + '/simulation_params_oa.csv')     

dict_mixed_gain_map_oa = {}
#dict_mixed_gain_map_oa['Gain_Mixed_agents_MAP'] = mixed_agent_gain_per_miniblock_map_oa.transpose()
dict_mixed_gain_map_oa['Gain_Mixed_agents_MAP'] = mixed_agent_gain_per_miniblock_map[i_group_oa, :, :].transpose()
df_mixed_gain_oa = pd.DataFrame(dict_mixed_gain_map_oa['Gain_Mixed_agents_MAP'])
# With IDs:
#df_mixed_gain = np.transpose(pd.DataFrame(id_list_oa)).append(pd.DataFrame(dict_mixed_gain_map_oa['Gain_Mixed_agents_MAP'])) # With subject IDs
df_mixed_gain_oa.to_csv(datapath + '/Gain_Mixed_agents_MAP_oa.csv')     



dict_gain_PDagents_ya = {}
dict_gain_PDagents_ya['Gain_PD1_agent'] = mean_gain_per_miniblock[i_group_ya, :, 0, :]
dict_gain_PDagents_ya['Gain_PD2_agent'] = mean_gain_per_miniblock[i_group_ya, :, 1, :]
dict_gain_PDagents_ya['Gain_PD3_agent'] = mean_gain_per_miniblock[i_group_ya, :, 2, :]
dict_gain_PDagents_ya['Parameters'] = parameters[i_group_ya, :, :]

df_gain_per_mb_pd1_ya = pd.DataFrame(data=dict_gain_PDagents_ya['Gain_PD1_agent'])
df_gain_per_mb_pd2_ya = pd.DataFrame(data=dict_gain_PDagents_ya['Gain_PD2_agent'])
df_gain_per_mb_pd3_ya = pd.DataFrame(data=dict_gain_PDagents_ya['Gain_PD3_agent'])
df_parameters_ya = pd.DataFrame(data=dict_gain_PDagents_ya['Parameters'])

df_gain_per_mb_pd1_ya.to_csv(datapath + '/simulated_gain_pd1_ya.csv')     
df_gain_per_mb_pd2_ya.to_csv(datapath + '/simulated_gain_pd2_ya.csv') 
df_gain_per_mb_pd3_ya.to_csv(datapath + '/simulated_gain_pd3_ya.csv') 
df_parameters_ya.to_csv(datapath + '/simulation_params_ya.csv')     

dict_mixed_gain_map_ya = {}
#dict_mixed_gain_map_ya['Gain_Mixed_agents_MAP'] = mixed_agent_gain_per_miniblock_map_ya.transpose()
dict_mixed_gain_map_ya['Gain_Mixed_agents_MAP'] = mixed_agent_gain_per_miniblock_map[i_group_ya, :, :].transpose()
df_mixed_gain_ya = pd.DataFrame(dict_mixed_gain_map_ya['Gain_Mixed_agents_MAP'])
# With IDs:
#df_mixed_gain = np.transpose(pd.DataFrame(id_list_ya)).append(pd.DataFrame(dict_mixed_gain_map_ya['Gain_Mixed_agents_MAP'])) # With subject IDs
df_mixed_gain_ya.to_csv(datapath + '/Gain_Mixed_agents_MAP_ya.csv')     



# TODO:
# Comparison with subjects' data (gain)

date_inference = '20220412' # '20211207'
data_behav_ya = pd.read_csv('P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Results_Inference_behavioral_'+date_inference+'/data_ya.csv')
data_behav_oa = pd.read_csv('P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Results_Inference_behavioral_'+date_inference+'/data_oa.csv')
#data_PD_1staction_ya = pd.read_csv('P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Results_Inference_behavioral_'+date_inference+'/meanPD_1st_action_ya.csv')
#data_PD_1staction_oa = pd.read_csv('P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Results_Inference_behavioral_'+date_inference+'/meanPD_1st_action_oa.csv')
#data_inferredparams_ya = pd.read_csv('P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Results_Inference_behavioral_'+date_inference+'/pars_post_samples.csv')
n_subjects_ya = int(data_behav_ya['subject'].shape[0] / 140)
n_subjects_oa = int(data_behav_oa['subject'].shape[0] / 140)
#number_per_mb_ya = np.reshape(np.array(data_behav_ya['block_number']), (n_subjects, 140)) # test - 1st dim (0) is subjects, 2nd dim (1) is miniblocks
gain_per_mb_ya = np.reshape(np.array(data_behav_ya['gain']), (n_subjects_ya, 140))
gain_per_mb_oa = np.reshape(np.array(data_behav_oa['gain']), (n_subjects_oa, 140))

index_hiNoise_120 = np.where(noise0 == 1)
index_loNoise_120 = np.where(noise0 == 0)

index_hiNoise_120_140 = (index_hiNoise_120 + np.array(20))[0]
index_loNoise_120_140 = (index_loNoise_120 + np.array(20))[0]

gain_per_mb_ya.mean(0)[index_hiNoise_120].sum()
gain_per_mb_ya.mean(0)[index_loNoise_120].sum()

#mixed_gain_per_miniblock_map_ya.mean(0)[index_hiNoise_120].sum()
#mixed_gain_per_miniblock_map_ya.mean(0)[index_loNoise_120].sum()

gain_per_mb_oa.mean(0)[index_hiNoise_120_140].sum()
gain_per_mb_oa.mean(0)[index_loNoise_120_140].sum()

np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_hiNoise_120].sum()
np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_loNoise_120].sum()

np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_hiNoise_120].sum()
np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_loNoise_120].sum()


plt.figure()
plt.bar([0,3], [gain_per_mb_ya.mean(0)[index_hiNoise_120_140].sum(), gain_per_mb_ya.mean(0)[index_loNoise_120_140].sum()])
plt.bar([1,4], [np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_hiNoise_120].sum(),
np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_loNoise_120].sum()])
ax=plt.gca()
ax.set_xticks([0,1,3,4])
ax.set_xticklabels(['YA subjects, \n high noise', \
                     'matched \n agents, \n high noise', \
                         'YA subjects, \n low noise', \
                             'matched \n agents, \n low noise'])
ax.set_ylim([0,850])    
plt.savefig('gain_subjects_and_mixed_individual_agents_'+modelname+'_ya.png', dpi=300)

plt.figure()
plt.bar([0,3], [gain_per_mb_oa.mean(0)[index_hiNoise_120_140].sum(), gain_per_mb_oa.mean(0)[index_loNoise_120_140].sum()])
plt.bar([1,4], [np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_hiNoise_120].sum(),
np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_loNoise_120].sum()])
ax=plt.gca()
ax.set_xticks([0,1,3,4])
ax.set_xticklabels(['OA subjects, \n high noise', \
                     'matched \n agents, \n high noise', \
                         'OA subjects, \n low noise', \
                             'matched \n agents, \n low noise'])
ax.set_ylim([0,850])        
plt.savefig('gain_subjects_and_mixed_individual_agents_'+modelname+'_oa.png', dpi=300)    

'''#
plt.figure()
plt.bar([0,3], [gain_per_mb_ya.mean(0)[index_hiNoise_120_140].sum(), gain_per_mb_ya.mean(0)[index_loNoise_120_140].sum()])
plt.bar([1,4], [np.nanmean(mixed_agent_gain_per_miniblock_meanPD[i_group_ya,:,:], 0)[index_hiNoise_120].sum(),
np.nanmean(mixed_agent_gain_per_miniblock_meanPD[i_group_ya,:,:], 0)[index_loNoise_120].sum()])
ax=plt.gca()
ax.set_xticks([0,1,3,4])
ax.set_xticklabels(['YA subjects, \n high noise', \
                     'matched \n agents, \n high noise', \
                         'YA subjects, \n low noise', \
                             'matched \n agents, \n low noise'])
ax.set_ylim([0,850])    
plt.savefig('gain_subjects_and_mixed_individual_agents_'+modelname+'_ya_meanPD.png', dpi=300)
    
plt.figure()
plt.bar([0,3], [gain_per_mb_oa.mean(0)[index_hiNoise_120_140].sum(), gain_per_mb_oa.mean(0)[index_loNoise_120_140].sum()])
plt.bar([1,4], [np.nanmean(mixed_agent_gain_per_miniblock_meanPD[i_group_oa,:,:], 0)[index_hiNoise_120].sum(),
np.nanmean(mixed_agent_gain_per_miniblock_meanPD[i_group_oa,:,:], 0)[index_loNoise_120].sum()])
ax=plt.gca()
ax.set_xticks([0,1,3,4])
ax.set_xticklabels(['OA subjects, \n high noise', \
                     'matched \n agents, \n high noise', \
                         'OA subjects, \n low noise', \
                             'matched \n agents, \n low noise'])
ax.set_ylim([0,850])        
plt.savefig('gain_subjects_and_mixed_individual_agents_'+modelname+'_oa_meanPD.png', dpi=300)    
'''
np.nanmean(mean_gain_per_miniblock[i_group_oa, :, 1, :], 0)[index_hiNoise_120].sum()
np.nanmean(mean_gain_per_miniblock[i_group_oa, :, 1, :], 0)[index_loNoise_120].sum()


base_path = 'H:/Sesyn/TRR265-B09/Analysis_SAT-PD2_Sophia/SAT_PD_Inference_Scripts/'
index_loNoise_sorted = pd.read_csv(base_path+'index_loNoise_sorted_by_PD3_agentgain.csv')['loNoise'].values  
index_hiNoise_sorted = pd.read_csv(base_path+'index_hiNoise_sorted_by_PD3_agentgain.csv')['hiNoise'].values    

import scipy.stats as st
plt.figure()
plt.plot(np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_hiNoise_120], gain_per_mb_oa.mean(0)[index_hiNoise_120_140], '.', label='high noise'); 
plt.plot([-30,40],[-30,40], 'k--', lw=1)
plt.plot(np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_loNoise_120], gain_per_mb_oa.mean(0)[index_loNoise_120_140], '.', color='C1', label='low noise'); #plt.plot([-30,40],[-30,40], 'k--')
plt.legend()
linreg_oa_hinoise = st.linregress( np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_hiNoise_120], gain_per_mb_oa.mean(0)[(index_hiNoise_120 + np.array(20))[0]] )
plt.plot(np.arange(-30, 40), linreg_oa_hinoise.slope * np.arange(-30, 40) + linreg_oa_hinoise.intercept, '--', color='C0', lw=2)
plt.text(-20,20, 'r= '+str(np.round(linreg_oa_hinoise.rvalue, 2))+', p= '+str(np.round(linreg_oa_hinoise.pvalue, 2)), color='C0', fontsize=12)
linreg_oa_lonoise = st.linregress( np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_loNoise_120], gain_per_mb_oa.mean(0)[(index_loNoise_120 + np.array(20))[0]] )
plt.plot(np.arange(-30, 40), linreg_oa_lonoise.slope * np.arange(-30, 40) + linreg_oa_lonoise.intercept, '--', color='C1', lw=2)
plt.text(20, -10, 'r= '+str(np.round(linreg_oa_lonoise.rvalue, 2))+', p= '+str(np.round(linreg_oa_lonoise.pvalue, 2)), color='C1', fontsize=12)
plt.xlabel('Gain per miniblock (mixed individual rational agents)')
plt.ylabel('Mean gain per miniblock (subjects)')
plt.title('OA subjects')
plt.savefig('gain_subjects_vs_mixed_individual_agents_'+modelname+'_oa.png', dpi=300)

plt.figure()
plt.plot(np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_hiNoise_120], gain_per_mb_ya.mean(0)[index_hiNoise_120_140], '.', label='high noise'); 
plt.plot([-30,40],[-30,40], 'k--', lw=1)
plt.plot(np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_loNoise_120], gain_per_mb_ya.mean(0)[index_loNoise_120_140], '.', color='C1', label='low noise'); #plt.plot([-30,40],[-30,40], 'k--')
plt.legend()
linreg_ya_hinoise = st.linregress( np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_hiNoise_120], gain_per_mb_ya.mean(0)[(index_hiNoise_120 + np.array(20))[0]] )
plt.text(-20,20, 'r= '+str(np.round(linreg_ya_hinoise.rvalue, 2))+', p= '+str(np.round(linreg_ya_hinoise.pvalue, 2)), color='C0', fontsize=12)
plt.plot(np.arange(-30, 40), linreg_ya_hinoise.slope * np.arange(-30, 40) + linreg_ya_hinoise.intercept, '--', color='C0', lw=2)
linreg_ya_lonoise = st.linregress( np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_loNoise_120], gain_per_mb_ya.mean(0)[(index_loNoise_120 + np.array(20))[0]] )
plt.plot(np.arange(-30, 40), linreg_ya_lonoise.slope * np.arange(-30, 40) + linreg_ya_lonoise.intercept, '--', color='C1', lw=2)
plt.text(20, -10, 'r= '+str(np.round(linreg_ya_lonoise.rvalue, 2))+', p= '+str(np.round(linreg_ya_lonoise.pvalue, 2)), color='C1', fontsize=12)
plt.xlabel('Gain per miniblock (mixed individual rational agents)')
plt.ylabel('Mean gain per miniblock (subjects)')
plt.title('YA subjects')
plt.savefig('gain_subjects_vs_mixed_individual_agents_'+modelname+'_ya.png', dpi=300)

plt.figure()
plt.plot(np.nansum(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_hiNoise_120], gain_per_mb_ya.sum(0)[index_hiNoise_120_140], '.', label='high noise'); 
plt.plot([-1500,2500],[-1500,2500], 'k--', lw=1)
Delta=200
plt.plot([-1500,2500],[-1500+Delta,2500+Delta], 'r--', lw=1)
plt.plot([-1500,2500],[-1500-Delta,2500-Delta], 'b--', lw=1)
plt.plot(np.nansum(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_loNoise_120], gain_per_mb_ya.sum(0)[index_loNoise_120_140], '.', color='C1', label='low noise'); #plt.plot([-30,40],[-30,40], 'k--')
plt.legend()
linreg_ya_hinoise = st.linregress( np.nansum(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_hiNoise_120], gain_per_mb_ya.sum(0)[(index_hiNoise_120 + np.array(20))[0]] )
plt.text(-20,20, 'r= '+str(np.round(linreg_ya_hinoise.rvalue, 2))+', p= '+str(np.round(linreg_ya_hinoise.pvalue, 2)), color='C0', fontsize=12)
plt.plot(np.arange(-30, 40), linreg_ya_hinoise.slope * np.arange(-30, 40) + linreg_ya_hinoise.intercept, '--', color='C0', lw=2)
linreg_ya_lonoise = st.linregress( np.nansum(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_loNoise_120], gain_per_mb_ya.sum(0)[(index_loNoise_120 + np.array(20))[0]] )
plt.plot(np.arange(-30, 40), linreg_ya_lonoise.slope * np.arange(-30, 40) + linreg_ya_lonoise.intercept, '--', color='C1', lw=2)
plt.text(20, -10, 'r= '+str(np.round(linreg_ya_lonoise.rvalue, 2))+', p= '+str(np.round(linreg_ya_lonoise.pvalue, 2)), color='C1', fontsize=12)
plt.xlabel('Total gain (mixed individual rational agents)')
plt.ylabel('Mean total gain (subjects)')
plt.title('YA subjects')


plt.figure()
plt.boxplot([(abs(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:] - gain_per_mb_ya[:,20:])[:, index_hiNoise_120])[:,0,:].mean(1), \
             (abs(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:] - gain_per_mb_ya[:,20:])[:, index_loNoise_120])[:,0,:].mean(1)])
ax=plt.gca()
ax.set_xticklabels(['YA High noise','YA low noise'])
plt.ylabel('Mean absolute difference (points)')

plt.figure()
plt.boxplot([(abs(mixed_agent_gain_per_miniblock_map[i_group_oa,:-4,:] - gain_per_mb_oa[:,20:])[:, index_hiNoise_120])[:,0,:].mean(1), \
             (abs(mixed_agent_gain_per_miniblock_map[i_group_oa,:-4,:] - gain_per_mb_oa[:,20:])[:, index_loNoise_120])[:,0,:].mean(1)])
ax=plt.gca()
ax.set_xticklabels(['OA High noise','OA low noise'])
plt.ylabel('Mean absolute difference (points)')

plt.subplot(221)
plt.plot(gain_per_mb_ya[:,:][:, index_loNoise_120 + np.array(20)][:,0,:], mixed_agent_gain_per_miniblock_map[i_group_ya,:,:][:, index_loNoise_120][:,0,:],  'b.', alpha=0.01); plt.plot([-40,40],[-40,40],'k--')
plt.xlabel('Subject gain (single miniblocks)')
plt.ylabel('Mean agent gain')
plt.title('YA low noise')
plt.subplot(222)
plt.plot(gain_per_mb_ya[:,20:][:, index_hiNoise_120][:,0,:], mixed_agent_gain_per_miniblock_map[i_group_ya,:,:][:, index_loNoise_120][:,0,:],  'r.', alpha=0.01); plt.plot([-40,40],[-40,40],'k--')
plt.xlabel('Subject gain (single miniblocks)')
plt.title('YA high noise')
plt.subplot(223)
plt.plot(gain_per_mb_oa[:,20:][:, index_loNoise_120][:,0,:], mixed_agent_gain_per_miniblock_map[i_group_oa,:-4,:][:, index_loNoise_120][:,0,:],  'b.', alpha=0.01); plt.plot([-40,40],[-40,40],'k--')
plt.xlabel('Subject gain (single miniblocks)')
plt.ylabel('Mean agent gain')
plt.title('OA low noise')
plt.subplot(224)
plt.plot(gain_per_mb_oa[:,20:][:, index_hiNoise_120][:,0,:], mixed_agent_gain_per_miniblock_map[i_group_oa,:-4,:][:, index_loNoise_120][:,0,:],  'r.', alpha=0.01); plt.plot([-40,40],[-40,40],'k--')
plt.xlabel('Subject gain (single miniblocks)')
plt.title('OA high noise')
plt.tight_layout()

linreg_ya_lonoise_all = st.linregress( np.reshape( mixed_agent_gain_per_miniblock_map[i_group_ya,:,:][:, index_loNoise_120][:,0,:], 60*60), np.reshape(gain_per_mb_ya[:, index_loNoise_120 + np.array(20)][:,0,:], 60*60) )
linreg_ya_hinoise_all = st.linregress( np.reshape( mixed_agent_gain_per_miniblock_map[i_group_ya,:,:][:, index_hiNoise_120][:,0,:], 60*60), np.reshape(gain_per_mb_ya[:, index_hiNoise_120 + np.array(20)][:,0,:], 60*60) )

pval_linreg_ya_lonoise_persubj= np.nan * np.ones(60)
rval_linreg_ya_lonoise_persubj= np.nan * np.ones(60)
for i_subj in range(60):
    linreg_ya_lonoise_persubj = st.linregress(  mixed_agent_gain_per_miniblock_map[i_group_ya, i_subj,:][index_loNoise_120], \
                                              gain_per_mb_ya[i_subj, index_loNoise_120 + np.array(20)] )
    pval_linreg_ya_lonoise_persubj[i_subj] = linreg_ya_lonoise_persubj.pvalue
    rval_linreg_ya_lonoise_persubj[i_subj] = linreg_ya_lonoise_persubj.rvalue    

    
'''#
plt.figure()
plt.plot(gain_per_mb_oa.mean(0)[index_hiNoise_120_140], post_meanPD_firstAction_oa.mean(1)[index_hiNoise_120_140], '.', label='high noise'); 
plt.plot(gain_per_mb_oa.mean(0)[index_loNoise_120_140], post_meanPD_firstAction_oa.mean(1)[index_loNoise_120_140],  '.', label='low noise'); 
plt.legend()
plt.ylabel('Mean planning depth')
plt.xlabel('Mean gain per miniblock (subjects)')
plt.title('OA subjects')
'''

plt.figure()
plt.plot(np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_hiNoise_120], post_meanPD_firstAction_oa.mean(1)[index_hiNoise_120_140], '.', label='high noise'); 
linreg_oa_pd_hinoise = st.linregress( np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_hiNoise_120], post_meanPD_firstAction_oa.mean(1)[index_hiNoise_120_140] )
plt.plot(np.arange(-30, 40), linreg_oa_pd_hinoise.slope * np.arange(-30, 40) + linreg_oa_pd_hinoise.intercept, '--', color='C0', lw=2)
plt.text(-20, 2.1, 'r= '+str(np.round(linreg_oa_pd_hinoise.rvalue, 2))+', p= '+str(np.round(linreg_oa_pd_hinoise.pvalue, 2)), color='C0', fontsize=12)
plt.plot(np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_loNoise_120], post_meanPD_firstAction_oa.mean(1)[index_loNoise_120_140],  '.', label='low noise'); 
linreg_oa_pd_lonoise = st.linregress( np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_oa,:,:], 0)[index_loNoise_120], post_meanPD_firstAction_oa.mean(1)[index_hiNoise_120_140] )
plt.plot(np.arange(-30, 40), linreg_oa_pd_lonoise.slope * np.arange(-30, 40) + linreg_oa_pd_lonoise.intercept, '--', color='C1', lw=2)
plt.text(20, 2.1, 'r= '+str(np.round(linreg_oa_pd_lonoise.rvalue, 2))+', p= '+str(np.round(linreg_oa_pd_lonoise.pvalue, 2)), color='C1', fontsize=12)
plt.legend()
plt.ylabel('Mean planning depth')
plt.xlabel('Gain per miniblock (mixed individual agents '+modelname+')')
plt.title('OA subjects')
plt.ylim([1.6, 2.8])


plt.figure()
plt.plot(np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_hiNoise_120], post_meanPD_firstAction_ya.mean(1)[index_hiNoise_120_140], '.', label='high noise'); 
linreg_ya_pd_hinoise = st.linregress( np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_hiNoise_120], post_meanPD_firstAction_ya.mean(1)[index_hiNoise_120_140] )
plt.plot(np.arange(-30, 40), linreg_ya_pd_hinoise.slope * np.arange(-30, 40) + linreg_ya_pd_hinoise.intercept, '--', color='C0', lw=2)
plt.text(-20, 2.2, 'r= '+str(np.round(linreg_ya_pd_hinoise.rvalue, 2))+', p= '+str(np.round(linreg_ya_pd_hinoise.pvalue, 2)), color='C0', fontsize=12)
plt.plot(np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_loNoise_120], post_meanPD_firstAction_ya.mean(1)[index_loNoise_120_140],  '.', label='low noise'); 
linreg_ya_pd_lonoise = st.linregress( np.nanmean(mixed_agent_gain_per_miniblock_map[i_group_ya,:,:], 0)[index_loNoise_120], post_meanPD_firstAction_ya.mean(1)[index_hiNoise_120_140] )
plt.plot(np.arange(-30, 40), linreg_ya_pd_lonoise.slope * np.arange(-30, 40) + linreg_ya_pd_lonoise.intercept, '--', color='C1', lw=2)
plt.text(20, 2.1, 'r= '+str(np.round(linreg_ya_pd_lonoise.rvalue, 2))+', p= '+str(np.round(linreg_ya_pd_lonoise.pvalue, 2)), color='C1', fontsize=12)
plt.legend()
plt.ylabel('Mean planning depth')
plt.xlabel('Gain per miniblock (mixed individual agents '+modelname+')')
plt.title('YA subjects')
plt.ylim([1.6, 2.8])
#plt.savefig('gain_mixed_individual_agents_ya.png', dpi=300)