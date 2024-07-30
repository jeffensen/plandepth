# In[0]:
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Lorenz Gönner & Sophia-Helen Sass
"""
"""
Here we simulate task behavior with the alternative models for planning strategies:
    full-breadth planning, low-probability pruning 
    And the two models including the probability discounting bias:
    discounted full-breadth planning and discounted low-probability pruning
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
from agents_discount_hyperbolic_theta_alpha_kmax30 import BackInductionDiscountHyperbolicThetaAlphakmax30
from agents_lowprob_pruning import BackInductionLowProbPruning
from agents_discount_hyperbolic_theta_lowprob_pruning_kmax30 import BackInductionDiscountHyperbolicThetaLowProbPruningkmax30

from simulate import Simulator
from inference import Inferrer
from helper_files import get_posterior_stats 

from calc_BIC import calc_BIC
from scipy.stats import mode

# In[1]:  

# set global variables
torch.manual_seed(16324)
pyro.enable_validation(True)

sns.set(context='talk', style='white', color_codes=True)

runs0 = 30             #number of simulations ("participants") 
mini_blocks0 = 120     #+20 for training which will be removed in the following
max_trials0 = 3        #maximum number of actions per mini-block
max_depth0 = 3         #maximum planning depth
na0 = 2                #number of actions
ns0 = 6                #number of states
no0 = 5                #number of outcomes
starting_points = 350  #number of points at the beginning of task

# load task configuration file 
read_file = open('config_file/space_adventure_pd_config_task_new_orig.json',"r")
exp1 = json.load(read_file)
read_file.close()

# load starting positions of each mini-block 
starts0 = exp1['startsExp']
import numpy
starts0 = numpy.asarray(starts0)
starts0 = starts0[20:140]
starts0 = starts0 -1

# load planet configurations for each mini-block
planets0 = exp1['planetsExp']
planets0 = numpy.asarray(planets0)
planets11 = planets0
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

 

agent_keys = ['rational', 'lowprob_pruning', 'discount_hyperbolic_theta_alpha_k3.0', \
              'discount_hyperbolic_theta_lowprob_pruning_k3.0']   
    
    
    
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

datapath = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Model_Fitting/Simulation_Results'

agents = {}

m0 = {} # mean parameter values
trans_pars0 = {}
dict_mb_gain_all = {}

 
for agent_key in agent_keys:    

    for i in range(3):
    # define space adventure task with aquired configurations
    # set number of trials to the max number of actions per mini-block
        space_advent0 = SpaceAdventure(conditions0,
                                  outcome_likelihoods=confs0,
                                  init_states=starts0,
                                  runs=runs0,
                                  mini_blocks=mini_blocks0,
                                  trials=max_trials0)

    # define the optimal agent, each with a different maximal planning depth
        if agent_key == 'rational':    
            agents['rational'] = BackInduction(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and alpha parameters as a normal distribution around a certain value
            #m0['rational'] = torch.tensor([1.099, 0., 0.0])# beta= 3, because 1.099=np.log(3); alpha = 0.5 (too high!!!)
            m0['rational'] = torch.tensor([1.099, 0., -2.])#  beta= 3, because 1.099=np.log(3); alpha = 0.1

        elif agent_key == 'lowprob_pruning': 
            agents['lowprob_pruning'] = BackInductionLowProbPruning(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta and theta parameters 
            m0['lowprob_pruning'] = torch.tensor([1.099, 0.])# beta= 3 --> tranformation in agent script 1.099=np.log(3) 

        
        elif agent_key == 'discount_hyperbolic_theta_alpha_k3.0':
             agents['discount_hyperbolic_theta_alpha_k3.0'] = BackInductionDiscountHyperbolicThetaAlphakmax30(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and kappa (discounting) parameters 
             m0['discount_hyperbolic_theta_alpha_k3.0'] = torch.tensor([1.099, 0., -2, -2.2])# beta= 3, because 1.099=np.log(3) // theta=0, alpha=0.1, k=3 = 30*sigmoid(-2.2)

        elif agent_key == 'discount_hyperbolic_theta_lowprob_pruning_k3.0':
             agents['discount_hyperbolic_theta_lowprob_pruning_k3.0'] = BackInductionDiscountHyperbolicThetaLowProbPruningkmax30(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and kappa (discounting) parameters 
             m0['discount_hyperbolic_theta_lowprob_pruning_k3.0'] = torch.tensor([1.099, 0., -2.2])# beta= 3, because 1.099=np.log(3) //  k=1 = 30*sigmoid(-2.2)

      
        trans_pars0[agent_key] = torch.distributions.Normal(m0[agent_key], 0.5).sample((runs0,)) # lower variability in parameters!
        agents[agent_key].set_parameters(trans_pars0[agent_key])
     

    # simulate behavior
        sim0 = Simulator(space_advent0,
                    agents[agent_key],
                    runs=runs0,
                    mini_blocks=mini_blocks0,
                    trials=3)
        sim0.simulate_experiment()

        simulations[agent_key].append(sim0)
        states[agent_key].append(space_advent0.states.clone())        

        responses0 = simulations[agent_key][-1].responses.clone() #response actions in simulation for every mini-block 
        responses0[torch.isnan(responses0)] = -1.
        responses0 = responses0.long()
        responses0_orig = simulations[agent_key][-1].responses.clone()
        responses0_orig[torch.isnan(responses0)] = -1.                
        points0 = (costs0[responses0] + fuel0[simulations[agent_key][-1].outcomes])  #reward for landing on a certain planet in simulation

        points0[simulations[agent_key][-1].outcomes < 0] = 0 #set MB in which points go below 0 on 0 
        performance[agent_key].append(points0.sum(dim=-1))   #sum up the gain of fuel points 
    
        trans_pars_depth[agent_key].append(trans_pars0[agent_key])
        points_depth[agent_key].append(points0)
        responses_depth[agent_key].append(responses0_orig)
    

        final_points[agent_key].append(points_depth[agent_key][i][:,:,:].numpy().sum(2).sum(1).mean())
 

    #'''#
    dict_mb_gain = {}
    dict_mb_gain['Mean_gain_PD3'] = points_depth[agent_key][2][:,:,:].numpy().sum(2).mean(0)
    dict_mb_gain['Std_gain_PD3'] = points_depth[agent_key][2][:,:,:].numpy().sum(2).std(0)
    dict_mb_gain['Mean_gain_PD2'] = points_depth[agent_key][1][:,:,:].numpy().sum(2).mean(0)
    dict_mb_gain['Std_gain_PD2'] = points_depth[agent_key][1][:,:,:].numpy().sum(2).std(0)
    dict_mb_gain['Mean_gain_PD1'] = points_depth[agent_key][0][:,:,:].numpy().sum(2).mean(0)
    dict_mb_gain['Std_gain_PD1'] = points_depth[agent_key][0][:,:,:].numpy().sum(2).std(0)

    df_mean_std_permb = pd.DataFrame(data=dict_mb_gain)    
  #  df_mean_std_permb.to_csv(datapath + '/miniblock_gain_mean_std_'+agent_key+'_'+str(runs0)+'.csv')    
    #'''
    
    dict_mb_gain_all[agent_key] = {}
    dict_mb_gain_all[agent_key]['Mean_gain_PD3'] = points_depth[agent_key][2][:,:,:].numpy().sum(2).mean(0)
    dict_mb_gain_all[agent_key]['Std_gain_PD3'] = points_depth[agent_key][2][:,:,:].numpy().sum(2).std(0)
    dict_mb_gain_all[agent_key]['Mean_gain_PD2'] = points_depth[agent_key][1][:,:,:].numpy().sum(2).mean(0)
    dict_mb_gain_all[agent_key]['Std_gain_PD2'] = points_depth[agent_key][1][:,:,:].numpy().sum(2).std(0)
    dict_mb_gain_all[agent_key]['Mean_gain_PD1'] = points_depth[agent_key][0][:,:,:].numpy().sum(2).mean(0)
    dict_mb_gain_all[agent_key]['Std_gain_PD1'] = points_depth[agent_key][0][:,:,:].numpy().sum(2).std(0)    
    
    df_mean_std_permb_all = pd.DataFrame(data=dict_mb_gain_all)
    df_mean_std_permb_all.to_csv(datapath + '/miniblock_gain_mean_std_allagents_'+str(runs0)+'.csv')    


# In[3]:
# RE-FITTING:

sim_number0 = 100
n_iter = 500

fitting_agents = {}
elbo = {}
modelfit = {}

datapath = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Model_Fitting/Crossfitting_Results'
datapath_modelfit = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Model_Fitting/Simulation_Results'

for i_fitted in range(len(agent_keys)):
    fitting_agents['fit-'+agent_keys[i_fitted]] = {}
    elbo['fit-'+agent_keys[i_fitted]] = {}        
    modelfit['fit-'+agent_keys[i_fitted]] = {}     
   
    for i_simulated in range(len(agent_keys)):    
        print()            
        print('fitted: '+agent_keys[i_fitted] + ', simulated: '+agent_keys[i_simulated]) 
        
        fitting_agents['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ] = 'fit-'+str(i_fitted)+'-sim-'+str(i_simulated)
        modelfit['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ] = {}


        
        responses0 = responses_depth[agent_keys[i_fitted]][2]

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
                          planning_depth=3)

      
        elif agent_keys[i_fitted] == 'lowprob_pruning':    
            fitting_agents['fit-lowprob_pruning']['sim-'+agent_keys[i_simulated]] = BackInductionLowProbPruning(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=3)    
            
       
        elif agent_keys[i_fitted] == 'discount_hyperbolic_theta_alpha_k3.0':
             fitting_agents['fit-discount_hyperbolic_theta_alpha_k3.0']['sim-'+agent_keys[i_simulated]] = BackInductionDiscountHyperbolicThetaAlphakmax30(confs0,
                          runs=runs0, 
                          mini_blocks=mini_blocks0, 
                          trials=3, 
                          costs = torch.tensor([0., 0.]),                           
                          planning_depth=3) # it was i +1 and i was 2 but i think its okay?
            

        elif agent_keys[i_fitted] == 'discount_hyperbolic_theta_lowprob_pruning_k3.0':    
            fitting_agents['fit-discount_hyperbolic_theta_lowprob_pruning_k3.0']['sim-'+agent_keys[i_simulated]] = BackInductionDiscountHyperbolicThetaLowProbPruningkmax30(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=3)
            
        
        infer = Inferrer( fitting_agents['fit-'+agent_keys[i_fitted]]['sim-'+agent_keys[i_simulated]], stimuli0, responses0, mask0)
        infer.fit(num_iterations = n_iter, num_particles=100, optim_kwargs={'lr': 0.1})  
        elbo['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ] = infer.loss[:]
        n_samples = 100
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
        

comparison_label = 'publication' 
np.savez(datapath + '/modelfit-'+comparison_label+'_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', modelfit)
np.savez(datapath + '/elbo-'+comparison_label+'_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', elbo)
np.savez(datapath + '/responses_depth-'+comparison_label+'_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', responses_depth)
np.savez(datapath + '/states-'+comparison_label+'_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', states)
np.savez(datapath + '/m0_params-'+comparison_label+'_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', m0)


df_modelfit = pd.DataFrame(modelfit)
pd.DataFrame(modelfit).to_csv(datapath + '/modelfit.csv')           

# Confusion matrix: See Wilson & Collins 2019, eLife
confusion_mat_BIC = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(fit | sim) - Probability that a given fitted model fits a given simulated model best, based on BIC
confusion_mat_BIC_hinoise = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(fit | sim) - Probability that a given fitted model fits a given simulated model best, based on BIC
confusion_mat_BIC_lonoise = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(fit | sim) - Probability that a given fitted model fits a given simulated model best, based on BIC

inversion_mat_BIC = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(sim | fit) - Confidence that data resulted from a given simulated model, given a certain best-fitting model based on BIC
inversion_mat_BIC_hinoise = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(sim | fit) - Confidence that data resulted from a given simulated model, given a certain best-fitting model based on BIC
inversion_mat_BIC_lonoise = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(sim | fit) - Confidence that data resulted from a given simulated model, given a certain best-fitting model based on BIC

confusion_mat_rhosq = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(fit | sim) - Probability that a given fitted model fits a given simulated model best, based on rhosquare
confusion_mat_rhosq_hinoise = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(fit | sim) - Probability that a given fitted model fits a given simulated model best, based on rhosquare
confusion_mat_rhosq_lonoise = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(fit | sim) - Probability that a given fitted model fits a given simulated model best, based on rhosquare
inversion_mat_rhosq = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(sim | fit) - Confidence that data resulted from a given simulated model, given a certain best-fitting model based on rhosquare
inversion_mat_rhosq_hinoise = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(sim | fit) - Confidence that data resulted from a given simulated model, given a certain best-fitting model based on rhosquare
inversion_mat_rhosq_lonoise = np.nan * np.ones([len(agent_keys), len(agent_keys)]) # p(sim | fit) - Confidence that data resulted from a given simulated model, given a certain best-fitting model based on rhosquare



for i_simulated in range(len(agent_keys)):
    sim_model = 'sim-' + agent_keys[i_simulated]
        
    for i_fitted in range(len(agent_keys)):
        fit_model = 'fit-' + agent_keys[i_fitted]        
            
        nanmin_BIC_120_mean = np.nanmin(np.array([
                             modelfit['fit-'+agent_keys[0]][sim_model]['BIC_120_mean'], \
                             modelfit['fit-'+agent_keys[1]][sim_model]['BIC_120_mean'], \
                             modelfit['fit-'+agent_keys[2]][sim_model]['BIC_120_mean'], \
                             modelfit['fit-'+agent_keys[3]][sim_model]['BIC_120_mean']]), 0)            

        nanmin_BIC_hinoise = np.nanmin(np.array([
                             modelfit['fit-'+agent_keys[0]][sim_model]['BIC_hinoise_120'], \
                             modelfit['fit-'+agent_keys[1]][sim_model]['BIC_hinoise_120'], \
                             modelfit['fit-'+agent_keys[2]][sim_model]['BIC_hinoise_120'], \
                             modelfit['fit-'+agent_keys[3]][sim_model]['BIC_hinoise_120']]), 0)            

        nanmin_BIC_lonoise = np.nanmin(np.array([
                             modelfit['fit-'+agent_keys[0]][sim_model]['BIC_lonoise_120'], \
                             modelfit['fit-'+agent_keys[1]][sim_model]['BIC_lonoise_120'], \
                             modelfit['fit-'+agent_keys[2]][sim_model]['BIC_lonoise_120'], \
                             modelfit['fit-'+agent_keys[3]][sim_model]['BIC_lonoise_120']]), 0) 
            
        nanmax_rhosquare_120_mean = np.nanmax(np.array([
                             modelfit['fit-'+agent_keys[0]][sim_model]['pseudo_rsquare_120_mean'], \
                             modelfit['fit-'+agent_keys[1]][sim_model]['pseudo_rsquare_120_mean'], \
                             modelfit['fit-'+agent_keys[2]][sim_model]['pseudo_rsquare_120_mean'], \
                             modelfit['fit-'+agent_keys[3]][sim_model]['pseudo_rsquare_120_mean']]), 0)   

        nanmax_rhosquare_hinoise = np.nanmax(np.array([
                             modelfit['fit-'+agent_keys[0]][sim_model]['pseudo_rsquare_hinoise_120_mean'], \
                             modelfit['fit-'+agent_keys[1]][sim_model]['pseudo_rsquare_hinoise_120_mean'], \
                             modelfit['fit-'+agent_keys[2]][sim_model]['pseudo_rsquare_hinoise_120_mean'], \
                             modelfit['fit-'+agent_keys[3]][sim_model]['pseudo_rsquare_hinoise_120_mean']]), 0)   

        nanmax_rhosquare_lonoise = np.nanmax(np.array([
                             modelfit['fit-'+agent_keys[0]][sim_model]['pseudo_rsquare_lonoise_120_mean'], \
                             modelfit['fit-'+agent_keys[1]][sim_model]['pseudo_rsquare_lonoise_120_mean'], \
                             modelfit['fit-'+agent_keys[2]][sim_model]['pseudo_rsquare_lonoise_120_mean'], \
                             modelfit['fit-'+agent_keys[3]][sim_model]['pseudo_rsquare_lonoise_120_mean']]), 0)   
            
        n_valid_model_fits_BIC_120_mean = float( runs0 - np.sum(np.isnan(nanmin_BIC_120_mean)))        
        n_valid_model_fits_BIC_hinoise = float( runs0 - np.sum(np.isnan(nanmin_BIC_hinoise)))         
        n_valid_model_fits_BIC_lonoise = float( runs0 - np.sum(np.isnan(nanmin_BIC_lonoise)))                 
        
        confusion_mat_BIC[i_simulated, i_fitted] = sum(modelfit[fit_model][sim_model]['BIC_120_mean'] <= \
                                                       nanmin_BIC_120_mean) / n_valid_model_fits_BIC_120_mean
        confusion_mat_BIC_hinoise[i_simulated, i_fitted] = sum(modelfit[fit_model][sim_model]['BIC_hinoise_120'] <= \
                                                               nanmin_BIC_hinoise) / n_valid_model_fits_BIC_hinoise
        confusion_mat_BIC_lonoise[i_simulated, i_fitted] = sum(modelfit[fit_model][sim_model]['BIC_lonoise_120'] <= \
                                                               nanmin_BIC_lonoise) / n_valid_model_fits_BIC_lonoise

        n_valid_model_fits_rhosquare_120_mean = float( runs0 - np.sum(np.isnan(nanmax_rhosquare_120_mean)))        
        n_valid_model_fits_rhosquare_hinoise = float( runs0 - np.sum(np.isnan(nanmax_rhosquare_hinoise)))                
        n_valid_model_fits_rhosquare_lonoise = float( runs0 - np.sum(np.isnan(nanmax_rhosquare_lonoise)))                        
        
        confusion_mat_rhosq[i_simulated, i_fitted] = sum(modelfit[fit_model][sim_model]['pseudo_rsquare_120_mean'] >= \
                                                         nanmax_rhosquare_120_mean) / n_valid_model_fits_rhosquare_120_mean
        confusion_mat_rhosq_hinoise[i_simulated, i_fitted] = sum(modelfit[fit_model][sim_model]['pseudo_rsquare_hinoise_120_mean'] >= \
                                                         nanmax_rhosquare_hinoise) / n_valid_model_fits_rhosquare_hinoise        
        confusion_mat_rhosq_lonoise[i_simulated, i_fitted] = sum(modelfit[fit_model][sim_model]['pseudo_rsquare_lonoise_120_mean'] >= \
                                                         nanmax_rhosquare_lonoise) / n_valid_model_fits_rhosquare_lonoise        

for i_fitted in range(len(agent_keys)):    
    for i_simulated in range(len(agent_keys)):    
        inversion_mat_BIC[i_simulated, i_fitted] = confusion_mat_BIC[i_simulated, i_fitted] / np.sum(confusion_mat_BIC[:, i_fitted])
        inversion_mat_BIC_hinoise[i_simulated, i_fitted] = confusion_mat_BIC_hinoise[i_simulated, i_fitted] / np.sum(confusion_mat_BIC_hinoise[:, i_fitted])        
        inversion_mat_BIC_lonoise[i_simulated, i_fitted] = confusion_mat_BIC_lonoise[i_simulated, i_fitted] / np.sum(confusion_mat_BIC_lonoise[:, i_fitted])        
        
        inversion_mat_rhosq[i_simulated, i_fitted] = confusion_mat_rhosq[i_simulated, i_fitted] / np.sum(confusion_mat_rhosq[:, i_fitted])        
        inversion_mat_rhosq_hinoise[i_simulated, i_fitted] = confusion_mat_rhosq_hinoise[i_simulated, i_fitted] / np.sum(confusion_mat_rhosq_hinoise[:, i_fitted])        
        inversion_mat_rhosq_lonoise[i_simulated, i_fitted] = confusion_mat_rhosq_lonoise[i_simulated, i_fitted] / np.sum(confusion_mat_rhosq_lonoise[:, i_fitted])                


def plot_confusion_inversion_matrix(matrix, \
                                    title, \
                                     x_labels, \
                                     y_labels, \
                                     filename):    
    plt.figure(dpi=150)
    plt.matshow(matrix, cmap='viridis', fignum=False) # 
    fsize = 20
    #plt.colorbar()
    for i in range(len(agent_keys)):
        for j in range(len(agent_keys)):
            if matrix[i,j] >= 0.5:
                plt.text(j-0.2, i, str(np.round(matrix[i,j], 2)), fontsize=fsize)
            else:
                plt.text(j-0.2, i, str(np.round(matrix[i,j], 2)), fontsize=fsize, color='white')        
    plt.xlabel('best-fitting model')            
    plt.ylabel('simulated model')   
    plt.title(title)       
    ax=plt.gca()
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    #for tick in ax.yaxis.get_major_ticks():
    #    tick.label.set_fontsize(fsize)
    #plt.gcf().set_size_inches(10, 7)
    plt.gcf().set_size_inches(11, 8)    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    return


str_model1 = 'Full-breadth planning'
str_model2 = 'Low-probability \n pruning'
str_model3 = 'Hyperbolic \n discounting \n k=3' 
str_model4 = 'Hyperbolic \n discounting \n low-probability \n pruning'

plot_confusion_inversion_matrix(confusion_mat_BIC, \
                                'Confusion matrix based on BIC: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/confusion_mat_BIC_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(confusion_mat_BIC_hinoise, \
                                'High noise \n Confusion matrix based on BIC: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/confusion_mat_BIC_highnoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(confusion_mat_BIC_lonoise, \
                                'Low noise \n Confusion matrix based on BIC: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/confusion_mat_BIC_lownoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')
    
plot_confusion_inversion_matrix(inversion_mat_BIC, \
                                'Inversion matrix based on BIC: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/inversion_mat_BIC_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')
  
plot_confusion_inversion_matrix(inversion_mat_BIC_hinoise, \
                                'High noise \n Inversion matrix based on BIC: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/inversion_mat_BIC_highnoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(inversion_mat_BIC_lonoise, \
                                'Low noise \n Inversion matrix based on BIC: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/inversion_mat_BIC_lownoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')
     
    
# Same plots for matrices based on rhosquare:
    
plot_confusion_inversion_matrix(confusion_mat_rhosq, \
                                'Confusion matrix based on rhosquare: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/confusion_mat_rhosquare_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(confusion_mat_rhosq_hinoise, \
                                'High noise \n Confusion matrix based on rhosquare: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/confusion_mat_rhosquare_highnoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(confusion_mat_rhosq_lonoise, \
                                'Low noise \n Confusion matrix based on rhosquare: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/confusion_mat_rhosquare_lownoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')
    
plot_confusion_inversion_matrix(inversion_mat_rhosq, \
                                'Inversion matrix based on rhosquare: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/inversion_mat_rhosquare_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(inversion_mat_rhosq_hinoise, \
                                'High noise \n Inversion matrix based on rhosquare: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/inversion_mat_rhosquare_highnoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(inversion_mat_rhosq_lonoise, \
                                'Low noise \n Inversion matrix based on rhosquare: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                datapath + '/inversion_mat_rhosquare_lownoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')