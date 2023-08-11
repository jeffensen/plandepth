# In[0]:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:01:19 2021

@author: Sophia-Helen Sass for adaptaions
"""
"""
Created on Fri Dec 21 17:23:10 2018
Here we will test the validity of the inference procedure for estimating free parameters of the behavioural model.
In a frist step we will simulate behaviour from the agents with a fixed planning depth and try to recover model
parameters as mini-block dependent planning depth. In the second step, we will simulate behaviour from agents
with varying planning depth and try to determine the estimation accuracy of the free model paramters and
mini-block dependent planning depth.
@author: Dimitrije Markovic
"""

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import scipy.io as io
import json
import scipy.stats as st
import pandas as pd

sys.path.append('../')
from tasks import SpaceAdventure
from agents import BackInduction
from agents_nonlearning_hilonoise import BackInductionNoLearning
from agents_discount_Noise_theta import BackInductionDiscountNoiseTheta
from agents_anchor_pruning import BackInductionAnchorPruning
from agents_discount_Noise_theta_learnprobs import BackInductionDiscountNoiseThetaLearnprobs
from agents_discount_Noise_theta_anchor_pruning import BackInductionDiscountNoiseThetaAnchorPruning
from agents_discount_Noise_theta_fitprobs import BackInductionDiscountNoiseThetaFitprobs
from agents_discount_Noise_theta_realprobs import BackInductionDiscountNoiseThetaRealprobs
from simulate import Simulator
from inference import Inferrer
from helper_files import get_posterior_stats # load_and_format_behavioural_data

from calc_BIC import calc_BIC

# In[1]:  #### simulation and recovery for SAT PD version 2.0 ###########################################
# changes in task: new planet configs, no action costs, 140 trials in total, first 20 are training trials
#                  noise conditions are pseudo-randomized (no blocks), only mini-blocks of 3 actions 

import pylab as plt
import numpy as np
plt.plot(np.random.rand(10))

# set global variables
torch.manual_seed(16324)
pyro.enable_validation(True)

sns.set(context='talk', style='white', color_codes=True)

runs0 = 20 # 1000 # 40            #number of simulations 
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
#starts0 = starts0[19:139]
starts0 = starts0[20:140]
starts0 = starts0 -1

# load planet configurations for each mini-block
planets0 = exp1['planetsExp']
planets0 = numpy.asarray(planets0)
planets11 = planets0
#planets0 = planets0[19:139,:]
planets0 = planets0[20:140,:]
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
#noise0 = noise0[19:139]
noise0 = noise0[20:140]


# number of actions for each mini-block 
trials0 = exp1['conditionsExp']['notrials']
trials0 = numpy.asarray(trials0)
#trials0 = trials0[19:139]
trials0 = trials0[20:140]

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
agent_keys = ['rational', 'nolearning_hilonoise', 'discount_noise_theta_gamma0.7', 'discount_noise_theta_gamma0.3', \
                  'anchor_pruning', 'discount_noise_theta_learnprobs', 'discount_noise_theta_anchor_pruning', \
                  'discount_noise_theta_fitprobs', 'random',  \
                  'discount_noise_theta_realprobs_gamma0.7', 'discount_noise_theta_realprobs_gamma0.3']
'''

   
#agent_keys = ['discount_noise_theta_fitprobs']:    
    
#agent_keys = ['rational', 'nolearning_hilonoise']   
agent_keys = ['rational']

simulations = {}
performance = {}
trans_pars_depth = {}
points_depth = {}
responses_depth = {}
final_points = {}
states = {}

for agent_key in agent_keys:    
    simulations[agent_key] = []
    performance[agent_key] = []    
    trans_pars_depth[agent_key] = []    
    points_depth[agent_key] = []    
    responses_depth[agent_key] = []    
    final_points[agent_key] = []    
    states[agent_key] = []        



datapath = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Model fitting'

agents = {}

m0 = {} # mean parameter values
trans_pars0 = {}


 

#sim_number0 = 0                                             # here we set the planning depth for the 
                                                            # inference (where 0 referd to PD = 1)

sim_number0 = 2
# This is where inference starts

n_iter = 10 # 100


fitting_agents = {}
elbo = {}
modelfit = {}

datapath = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Results_crossfitting'

for i_fitted in range(len(agent_keys)):
    fitting_agents['fit-'+agent_keys[i_fitted]] = {}
    modelfit['fit-'+agent_keys[i_fitted]] = {}     
   
    for i_simulated in range(len(agent_keys)):    
        print('fitted: '+agent_keys[i_fitted] + ', simulated: '+agent_keys[i_simulated]) 
        
        fitting_agents['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ] = 'fit-'+str(i_fitted)+'-sim-'+str(i_simulated)
        modelfit['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ] = {}

        # load simulated responses:
        file_responses = np.load(datapath + '/responses_depth-default-vs_nolearning_20runs_1000iter_100samp.npz', allow_pickle=True)    
        fresp = file_responses.files # names of the stored arrays (['post_depth_oa', 'm_prob_oa', 'exc_count_oa'])
        responses = file_responses[fresp[0]]
        responses0 = responses.tolist()[agent_keys[i_fitted]]

        mask0 = ~torch.isnan(responses0)

        # load simulated conditions / states:
        stimuli0 = {'conditions': conditions0,
           'states': states[agent_keys[i_simulated]][sim_number0],
           'configs': confs0}            

        if agent_keys[i_fitted] == 'rational':    
        
            fitting_agents['fit-rational']['sim-'+agent_keys[i_simulated]] = BackInduction(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

        elif agent_keys[i_fitted] == 'nolearning_hilonoise':    
            fitting_agents['fit-nolearning_hilonoise']['sim-'+agent_keys[i_simulated]] = BackInductionNoLearning(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)    

        
        infer = Inferrer( fitting_agents['fit-'+agent_keys[i_fitted]]['sim-'+agent_keys[i_simulated]], stimuli0, responses0, mask0)
        #infer.fit(num_iterations=2000, num_particles=100)#, optim_kwargs={'lr': 0.5}) # 500  # change the number of iterations here if you like
        infer.fit(num_iterations = n_iter, num_particles=100, optim_kwargs={'lr': 0.1})#) # 500  # change the number of iterations here if you like
    
        elbo['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ] = infer.loss[:]
        n_samples = 100 # 100
        post_marg = infer.sample_posterior_marginal(n_samples=n_samples)        
        post_depth, m_prob, exc_count = get_posterior_stats(post_marg)
        np.savez(datapath + '/plandepth_stats_fit-'+agent_keys[i_fitted]+'_sim-'+agent_keys[i_simulated]+'_'+str(runs0)+'runs_'+str(n_iter)+'iter', post_depth, m_prob, exc_count)
        
        pseudo_rsquare_120_mean, BIC_120_mean, \
           pseudo_rsquare_hinoise_120_mean, BIC_hinoise_120, \
           pseudo_rsquare_lonoise_120_mean, BIC_lonoise_120 = calc_BIC(infer, responses0, conditions0, m_prob)
        modelfit['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ]['pseudo_rsquare_120_mean'] = pseudo_rsquare_120_mean
        modelfit['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ]['pseudo_rsquare_hinoise_120_mean'] = pseudo_rsquare_hinoise_120_mean        
        modelfit['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ]['pseudo_rsquare_lonoise_120_mean'] = pseudo_rsquare_lonoise_120_mean
        modelfit['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ]['BIC_120_mean'] = BIC_120_mean        
        modelfit['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ]['BIC_hinoise_120'] = BIC_hinoise_120                
        modelfit['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ]['BIC_lonoise_120'] = BIC_lonoise_120    
        
        
        
#np.savez(datapath + '/modelfit_fit-'+agent_keys[i_fitted]+'_sim-'+agent_keys[i_simulated], modelfit)
np.savez(datapath + '/modelfit-default-vs_nolearning_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', modelfit)
np.savez(datapath + '/elbo-default-vs_nolearning_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', elbo)
np.savez(datapath + '/responses_depth-default-vs_nolearning_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', responses_depth)
np.savez(datapath + '/states-default-vs_nolearning_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', states)
np.savez(datapath + '/m0_params-default-vs_nolearning_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', m0)

# file = np.load(datapath + '/modelfit_fit-'+agent_keys[i_fitted]+'_sim-'+agent_keys[i_simulated], allow_pickle=True)

df_modelfit = pd.DataFrame(modelfit)
#pd.DataFrame(modelfit).to_csv(datapath + '/modelfit.csv')           

# Confusion matrix: See Wilson & Collins 2019, eLife
confusion_mat_BIC = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(fit | sim) - Probability that a given fitted model fits a given simulated model best, based on BIC
inversion_mat_BIC = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(sim | fit) - Confidence that data resulted from a given simulated model, given a certain best-fitting model based on BIC

confusion_mat_rhosq = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(fit | sim) - Probability that a given fitted model fits a given simulated model best, based on rhosquare
inversion_mat_rhosq = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(sim | fit) - Confidence that data resulted from a given simulated model, given a certain best-fitting model based on rhosquare

for i_simulated in range(len(agent_keys)):
    
    if i_simulated == 0:
        sim_model = 'sim-rational'
    elif i_simulated == 1:
        sim_model = 'sim-nolearning_hilonoise'        
        
    for i_fitted in range(len(agent_keys)):
        
        if i_fitted == 0:
            fit_model = 'fit-rational'
            other_model = 'fit-nolearning_hilonoise'
        elif i_fitted == 1:
            fit_model = 'fit-nolearning_hilonoise'            
            other_model = 'fit-rational'            
            
            confusion_mat_BIC[i_simulated, i_fitted] = \
                sum(modelfit[fit_model][sim_model]['BIC_120_mean'] < \
                    modelfit[other_model][sim_model]['BIC_120_mean']) / \
                    float( runs0 - np.sum(np.isnan(modelfit[other_model][sim_model]['BIC_120_mean'] + modelfit[fit_model][sim_model]['BIC_120_mean']))) # Compare "current" fitted model with other fitted models (per simulated agent)

            confusion_mat_rhosq[i_simulated, i_fitted] = \
                sum(modelfit[fit_model][sim_model]['pseudo_rsquare_120_mean'] > \
                    modelfit[other_model][sim_model]['pseudo_rsquare_120_mean']) / \
                    float( runs0 - np.sum(np.isnan(modelfit[other_model][sim_model]['BIC_120_mean'] + modelfit[fit_model][sim_model]['BIC_120_mean']))) # Compare "current" fitted model with other fitted models (per simulated agent)


for i_fitted in range(len(agent_keys)):    
    for i_simulated in range(len(agent_keys)):    
        inversion_mat_BIC[i_simulated, i_fitted] = confusion_mat_BIC[i_simulated, i_fitted] / np.sum(confusion_mat_BIC[:, i_fitted])
        inversion_mat_rhosq[i_simulated, i_fitted] = confusion_mat_rhosq[i_simulated, i_fitted] / np.sum(confusion_mat_rhosq[:, i_fitted])        


plt.matshow(confusion_mat_BIC, cmap='viridis', fignum=None) # 'YlOrRd') #
for i in range(2):
    for j in range(2):
        if confusion_mat_BIC[i,j] >= 0.5:
            plt.text(j, i, str(np.round(confusion_mat_BIC[i,j], 2)), fontsize=20)
        else:
            plt.text(j, i, str(np.round(confusion_mat_BIC[i,j], 2)), fontsize=20, color='white')        
plt.xlabel('best-fitting model')            
plt.ylabel('simulated model')   
plt.title('Confusion matrix based on BIC: \n  p(best-fitting model | simulated model)')       
ax=plt.gca()
ax.set_xticklabels(['', 'Full planning \n with alpha', 'Full planning \n no alpha'])
ax.set_yticklabels(['', 'Full planning \n with alpha', 'Full planning \n no alpha'])
#plt.tight_layout()
plt.gcf().set_size_inches(10, 7)
plt.savefig(datapath + '/confusion_mat_BIC_default-vs_nolearning_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png', dpi=150)

plt.matshow(inversion_mat_BIC, cmap='viridis') # 'YlOrRd') #
#plt.colorbar()
for i in range(2):
    for j in range(2):
        if inversion_mat_BIC[i,j] >= 0.5:
            plt.text(j, i, str(np.round(inversion_mat_BIC[i,j], 2)), fontsize=20)
        else:
            plt.text(j, i, str(np.round(inversion_mat_BIC[i,j], 2)), fontsize=20, color='white')        
plt.xlabel('best-fitting model')            
plt.ylabel('simulated model')   
plt.title('Inversion matrix based on BIC: \n  p(simulated model | best-fitting model)')       
ax=plt.gca()
ax.set_xticklabels(['', 'Full planning \n with alpha', 'Full planning \n no alpha'])
ax.set_yticklabels(['', 'Full planning \n with alpha', 'Full planning \n no alpha'])
plt.gcf().set_size_inches(10, 7)
plt.savefig(datapath + '/inversion_mat_BIC_default-vs_nolearning_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png', dpi=150)

        
'''#
for simulated_agent_key in agent_keys: # , 
    
    responses0 = simulations[simulated_agent_key][sim_number0].responses.clone()
    mask0 = ~torch.isnan(responses0)

    stimuli0 = {'conditions': conditions0,
           'states': states[simulated_agent_key][sim_number0],
           'configs': confs0}

    if simulated_agent_key == 'rational':    
        agent2_rational['rational'] = BackInduction(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
        
        agent2_anchor_pruning['rational'] = BackInductionAnchorPruning(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)        

        agent2_discount_noise_theta_gamma07['rational'] = BackInductionDiscountNoiseTheta(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)     

    elif simulated_agent_key == 'anchor_pruning':    
        agent2_rational['anchor_pruning'] = BackInduction(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)    
        
        agent2_anchor_pruning['anchor_pruning'] = BackInductionAnchorPruning(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)             

        agent2_discount_noise_theta_gamma07['anchor_pruning'] = BackInductionDiscountNoiseTheta(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)     
        
    elif simulated_agent_key == 'discount_noise_theta_gamma0.7':    
        agent2_rational['discount_noise_theta_gamma0.7'] = BackInduction(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
        
        agent2_anchor_pruning['discount_noise_theta_gamma0.7'] = BackInductionAnchorPruning(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)        

        agent2_discount_noise_theta_gamma07['discount_noise_theta_gamma0.7'] = BackInductionDiscountNoiseTheta(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)           

    for fitting_agent_key in agent_keys:
        if fitting_agent_key == 'rational':
            infer = Inferrer(agent2_rational[simulated_agent_key], stimuli0, responses0, mask0)
            #infer.fit(num_iterations=2000, num_particles=100)#, optim_kwargs={'lr': 0.5}) # 500  # change the number of iterations here if you like
            infer.fit(num_iterations = n_iter, num_particles=100, optim_kwargs={'lr': 0.1})#) # 500  # change the number of iterations here if you like

        elif fitting_agent_key == 'anchor_pruning':
            infer = Inferrer(agent2_anchor_pruning[simulated_agent_key], stimuli0, responses0, mask0)
            #infer.fit(num_iterations=2000, num_particles=100)#, optim_kwargs={'lr': 0.5}) # 500  # change the number of iterations here if you like
            infer.fit(num_iterations = n_iter, num_particles=100, optim_kwargs={'lr': 0.1})#) # 500  # change the number of iterations here if you like

        elif fitting_agent_key == 'discount_noise_theta_gamma0.7':
            infer = Inferrer(agent2_discount_noise_theta_gamma07[simulated_agent_key], stimuli0, responses0, mask0)
            #infer.fit(num_iterations=2000, num_particles=100)#, optim_kwargs={'lr': 0.5}) # 500  # change the number of iterations here if you like
            infer.fit(num_iterations = n_iter, num_particles=100, optim_kwargs={'lr': 0.1})#) # 500  # change the number of iterations here if you like
'''
            
        # plotting the ELBO for the given inference 
        plt.figure()
        plt.plot(infer.loss[:]) #change section you want to see here 
        #plt.plot(np.log(infer.loss[100:])) #change section you want to see here 
        plt.ylabel('ELBO')  
        plt.xlabel('x')
        plt.savefig('ELBO_PD3_sim_' + simulated_agent_key + '_fit_' + fitting_agent_key + '.pdf', dpi=600, bbox_inches='tight')
        plt.savefig('ELBO_PD3_sim_' + simulated_agent_key + '_fit_' + fitting_agent_key + '.png', dpi=600, bbox_inches='tight')

        labels = [r'$\tilde{\beta}$', r'$\theta$',   r'$\tilde{\alpha}$'] #these refer to the next plots and calculations
        