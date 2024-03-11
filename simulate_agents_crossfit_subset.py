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
from agents_discount_hyperbolic_theta_realprobs_kmax30 import BackInductionDiscountHyperbolicThetaRealProbskmax30
from agents_discount_hyperbolic_theta_anchor_pruning_kmax30 import BackInductionDiscountHyperbolicThetaAnchorPruningkmax30
from agents_discount_hyperbolic_theta_alpha_kmax30 import BackInductionDiscountHyperbolicThetaAlphakmax30

from simulate import Simulator
from inference import Inferrer
from helper_files import get_posterior_stats # load_and_format_behavioural_data

from calc_BIC import calc_BIC
from scipy.stats import mode

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

runs0 = 1000 # 20 # 40            #number of simulations 
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
                  'discount_noise_theta_realprobs_gamma0.7', 'discount_noise_theta_realprobs_gamma0.3', \
                  'discount_hyperbolic_theta_realprobs_k1.0', 'discount_hyperbolic_theta_realprobs_k20.0']
'''

   
#agent_keys = ['discount_noise_theta_fitprobs']:    
    
#agent_keys = ['rational', 'nolearning_hilonoise']   

#agent_keys = ['rational', \
#              'discount_hyperbolic_theta_realprobs_k1.0', 'discount_hyperbolic_theta_realprobs_k20.0', \
#              'discount_hyperbolic_theta_anchorpruning_k1.0', 'discount_hyperbolic_theta_anchorpruning_k20.0']
    

#agent_keys = ['rational', 'anchor_pruning', 'discount_hyperbolic_theta_realprobs_k10.0', \
#              'discount_hyperbolic_theta_anchorpruning_k10.0']   #  Discounting mit Real probs

agent_keys = ['rational', 'anchor_pruning', 'discount_hyperbolic_theta_alpha_k10.0', \
              'discount_hyperbolic_theta_anchorpruning_k10.0']   #  Discounting mit alpha
    
    
    
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

        elif agent_key == 'nolearning_hilonoise':    
            agents['nolearning_hilonoise'] = BackInductionNoLearning(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and alpha parameters as a normal distribution around a certain value
            m0['nolearning_hilonoise'] = torch.tensor([1.099, 0.])#     
            
        elif agent_key == 'discount_noise_theta_gamma0.7': 
            agents['discount_noise_theta_gamma0.7'] = BackInductionDiscountNoiseTheta(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
            m0['discount_noise_theta_gamma0.7'] = torch.tensor([1.099, 0., 0.85])# beta= 3, because 1.099=np.log(3) // gamma=0.7=sigmoid(0.85)

        elif agent_key == 'discount_noise_theta_gamma0.3': 
            agents['discount_noise_theta_gamma0.3'] = BackInductionDiscountNoiseTheta(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
            m0['discount_noise_theta_gamma0.3'] = torch.tensor([1.099, 0., -0.85])# beta= 3, because 1.099=np.log(3) // gamma=0.3=sigmoid(-0.85)  

        elif agent_key == 'anchor_pruning': 
            agents['anchor_pruning'] = BackInductionAnchorPruning(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
            m0['anchor_pruning'] = torch.tensor([1.099, 0.])# beta= 3, because 1.099=np.log(3) 

        elif agent_key == 'discount_noise_theta_learnprobs':    
            agents['discount_noise_theta_learnprobs'] = BackInductionDiscountNoiseThetaLearnprobs(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and alpha parameters as a normal distribution around a certain value
            #m0['discount_noise_theta_learnprobs'] = torch.tensor([1.099, 0., -2, 10.0])# beta= 3, because 1.099=np.log(3), theta=0, alpha=0.1, gamma=0.99 - same performance as a learning rational agent with alpha=0.1
            m0['discount_noise_theta_learnprobs'] = torch.tensor([1.099, 0., -2, 0.85])# beta= 3, because 1.099=np.log(3), theta=0, alpha=0.1, gamma=0.7
   
        elif agent_key == 'discount_noise_theta_anchor_pruning':    
            agents['discount_noise_theta_anchor_pruning'] = BackInductionDiscountNoiseThetaAnchorPruning(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and alpha parameters as a normal distribution around a certain value
            m0['discount_noise_theta_anchor_pruning'] = torch.tensor([1.099, 0., 0.85])# beta= 3, because 1.099=np.log(3), theta=0, gamma=0.7
 
        elif agent_key == 'discount_noise_theta_fitprobs':    
            agents['discount_noise_theta_fitprobs'] = BackInductionDiscountNoiseThetaFitprobs(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and alpha parameters as a normal distribution around a certain value
            #m0['discount_noise_theta_fitprobs'] = torch.tensor([1.099, 0., 10, 0, 0.85])# beta= 3, because 1.099=np.log(3), theta=0, prob_lonoise=0.99, prob_hinoise=0.5, gamma=0.7: 1716 points on average - very good!
            #m0['discount_noise_theta_fitprobs'] = torch.tensor([1.099, 0., 2.2, 0, 0.85])# beta= 3, because 1.099=np.log(3), theta=0, prob_lonoise=0.9, prob_hinoise=0.5, gamma=0.7: 1711 points on average - very good!
            #m0['discount_noise_theta_fitprobs'] = torch.tensor([1.099, 0., 2.2, 2.2, 0.85])# beta= 3, because 1.099=np.log(3), theta=0, prob_lonoise=0.9, prob_hinoise=0.9, gamma=0.7: 1673 points on average - still pretty good!           
            m0['discount_noise_theta_fitprobs'] = torch.tensor([1.099, 0., 10, 10, 0.85]) # beta= 3, because 1.099=np.log(3), theta=0, prob_lonoise=0.99, prob_hinoise=0.99, gamma=0.7: 1658 points on average - still pretty good!                      
  
        elif agent_key == 'random':    
            agents['random'] = BackInduction(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)        
            m0['random'] = torch.tensor([-10, 0., 0.]) # beta close to zero        
   
        elif agent_key == 'discount_noise_theta_realprobs_gamma0.7':
             agents['discount_noise_theta_realprobs_gamma0.7'] = BackInductionDiscountNoiseThetaRealprobs(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
             m0['discount_noise_theta_realprobs_gamma0.7'] = torch.tensor([1.099, 0., 0.85])# beta= 3, because 1.099=np.log(3) // gamma=0.7=sigmoid(0.85)

        elif agent_key == 'discount_noise_theta_realprobs_gamma0.3':
             agents['discount_noise_theta_realprobs_gamma0.3'] = BackInductionDiscountNoiseThetaRealprobs(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
             m0['discount_noise_theta_realprobs_gamma0.3'] = torch.tensor([1.099, 0., -0.85])# beta= 3, because 1.099=np.log(3) // gamma=0.3=sigmoid(-0.85)
             
        elif agent_key == 'discount_hyperbolic_theta_realprobs_k1.0':
             agents['discount_hyperbolic_theta_realprobs_k1.0'] = BackInductionDiscountHyperbolicThetaRealProbskmax30(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_realprobs_k1.0'] = torch.tensor([1.099, 0., -3.35])# beta= 3, because 1.099=np.log(3) // k=1 = 30*sigmoid(-3.35)

        elif agent_key == 'discount_hyperbolic_theta_realprobs_k10.0':
             agents['discount_hyperbolic_theta_realprobs_k10.0'] = BackInductionDiscountHyperbolicThetaRealProbskmax30(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_realprobs_k10.0'] = torch.tensor([1.099, 0., -0.7])# beta= 3, because 1.099=np.log(3) // k=10 = 30*sigmoid(-0.7)

             
        elif agent_key == 'discount_hyperbolic_theta_realprobs_k20.0':
             agents['discount_hyperbolic_theta_realprobs_k20.0'] = BackInductionDiscountHyperbolicThetaRealProbskmax30(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_realprobs_k20.0'] = torch.tensor([1.099, 0., 0.7])# beta= 3, because 1.099=np.log(3) // k=20 = 30*sigmoid(0.7)

        elif agent_key == 'discount_hyperbolic_theta_alpha_k10.0':
             agents['discount_hyperbolic_theta_alpha_k10.0'] = BackInductionDiscountHyperbolicThetaAlphakmax30(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_alpha_k10.0'] = torch.tensor([1.099, 0., -2, -0.7])# beta= 3, because 1.099=np.log(3) // theta=0, alpha=0.1, k=10 = 30*sigmoid(-0.7)

        elif agent_key == 'discount_hyperbolic_theta_anchorpruning_k1.0':
             agents['discount_hyperbolic_theta_anchorpruning_k1.0'] = BackInductionDiscountHyperbolicThetaAnchorPruningkmax30(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_anchorpruning_k1.0'] = torch.tensor([1.099, 0., -3.35])# beta= 3, because 1.099=np.log(3) //  k=1 = 30*sigmoid(-3.35)

        elif agent_key == 'discount_hyperbolic_theta_anchorpruning_k10.0':
             agents['discount_hyperbolic_theta_anchorpruning_k10.0'] = BackInductionDiscountHyperbolicThetaAnchorPruningkmax30(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_anchorpruning_k10.0'] = torch.tensor([1.099, 0., -0.7])# beta= 3, because 1.099=np.log(3) // k=10 = 30*sigmoid(-0.7)

        elif agent_key == 'discount_hyperbolic_theta_anchorpruning_k20.0':
             agents['discount_hyperbolic_theta_anchorpruning_k20.0'] = BackInductionDiscountHyperbolicThetaAnchorPruningkmax30(confs0,
                          runs=runs0, mini_blocks=mini_blocks0, trials=3, costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)
            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_anchorpruning_k20.0'] = torch.tensor([1.099, 0., 0.7])# beta= 3, because 1.099=np.log(3) // k=20 = 30*sigmoid(0.7)
    

        #trans_pars0[agent_key] = torch.distributions.Normal(m0[agent_key], 1.).sample((runs0,))
        trans_pars0[agent_key] = torch.distributions.Normal(m0[agent_key], 0.5).sample((runs0,)) # lower variability in parameters!
        agents[agent_key].set_parameters(trans_pars0[agent_key])
     
    #fixed values for parameters
       #trans_pars0[agent_key] = torch.tensor([2.,0.,0.5]).repeat(runs0,1) # this line sets beta, theta and alpha
       #agents[agent_key].set_parameters(trans_pars0[agent_key])


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

        points0[simulations[agent_key][-1].outcomes < 0] = 0 #set MB in which points go below 0 on 0 ?
        performance[agent_key].append(points0.sum(dim=-1))   #sum up the gains 
    
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
    df_mean_std_permb.to_csv(datapath + '/miniblock_gain_mean_std_'+agent_key+'_'+str(runs0)+'.csv')    
    #'''
    
    dict_mb_gain_all[agent_key] = {}
    dict_mb_gain_all[agent_key]['Mean_gain_PD3'] = points_depth[agent_key][2][:,:,:].numpy().sum(2).mean(0)
    dict_mb_gain_all[agent_key]['Std_gain_PD3'] = points_depth[agent_key][2][:,:,:].numpy().sum(2).std(0)
    dict_mb_gain_all[agent_key]['Mean_gain_PD2'] = points_depth[agent_key][1][:,:,:].numpy().sum(2).mean(0)
    dict_mb_gain_all[agent_key]['Std_gain_PD2'] = points_depth[agent_key][1][:,:,:].numpy().sum(2).std(0)
    dict_mb_gain_all[agent_key]['Mean_gain_PD1'] = points_depth[agent_key][0][:,:,:].numpy().sum(2).mean(0)
    dict_mb_gain_all[agent_key]['Std_gain_PD1'] = points_depth[agent_key][0][:,:,:].numpy().sum(2).std(0)    
    
    df_mean_std_permb_all = pd.DataFrame(data=dict_mb_gain_all)
    df_mean_std_permb_all.to_csv(datapath + '/miniblock_gain_mean_std_allagents_'+str(runs0)+'_Jan2024.csv')    




# Diagnostic plot:
plt.figure(figsize=(12,6), dpi=300)    
plt.plot(df_mean_std_permb_all['rational']['Mean_gain_PD3'] - df_mean_std_permb_all['anchor_pruning']['Mean_gain_PD3'], '.-') 
plt.xlabel('Miniblock')   
plt.ylabel('$\Delta$ Points PD3 rational - PD3 anchor pruning \n (Mean of 1000 agents)')
plotpath = 'P:/037/B3_MAIN_STUDY/01_Preparation/Miniblock design/Notizen/Test Anchor Pruning SAT-PD2/'
plt.savefig(plotpath + 'Points_rational_vs_anchor_pruning_SAT-PD2.png')
plt.boxplot(df_mean_std_permb_all['rational']['Mean_gain_PD3'] - df_mean_std_permb_all['anchor_pruning']['Mean_gain_PD3'])  


df_agentdiff_PD3 = pd.DataFrame(data = df_mean_std_permb_all['rational']['Mean_gain_PD3'] - df_mean_std_permb_all['anchor_pruning']['Mean_gain_PD3'])
df_agentdiff_PD3.to_csv(plotpath + 'Gain_difference_PD3_rational_vs_anchorpruning.csv')
  
# Miniblock 117 has max. difference:
# planets0[117] = array([4, 1, 1, 3, 0, 4])    
# starts0[0,117] = 0 ("ganz links"!!!)
# np.median(simulations['rational'][-1].responses[:,117], 0) # Jump-Move-Move, mean: 26.66 points
# np.median(states['rational'][-1][:,117], 0)
# points_depth['rational'][-1].mean(0)[117]
# High variability of choices in anchor pruning?! No clear action preference?!
# simulations['anchor_pruning'][-1].responses[:,117].numpy().mean(0)

# RE-FITTING:



sim_number0 = 2
# This is where inference starts

n_iter = 500 # 100


fitting_agents = {}
elbo = {}
modelfit = {}

datapath = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Results_crossfitting'
datapath_modelfit = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Model fitting'

for i_fitted in range(len(agent_keys)):
    fitting_agents['fit-'+agent_keys[i_fitted]] = {}
    elbo['fit-'+agent_keys[i_fitted]] = {}        
    modelfit['fit-'+agent_keys[i_fitted]] = {}     
   
    for i_simulated in range(len(agent_keys)):    
        print()            
        print('fitted: '+agent_keys[i_fitted] + ', simulated: '+agent_keys[i_simulated]) 
        
        fitting_agents['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ] = 'fit-'+str(i_fitted)+'-sim-'+str(i_simulated)
        modelfit['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ] = {}

        # load simulated responses:
        #file_responses = np.load(datapath + '/responses_depth-default-vs_nolearning_20runs_1000iter_100samp.npz', allow_pickle=True)    
        #file_responses = np.load(datapath_modelfit + '/responses_depth-default-vs_nolearning_1000runs_10iter_100samp.npz', allow_pickle=True)            
        
        
        #fresp = file_responses.files # names of the stored arrays (['post_depth_oa', 'm_prob_oa', 'exc_count_oa'])
        #responses = file_responses[fresp[0]]
        #responses0 = responses.tolist()[agent_keys[i_fitted]]        
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

        elif agent_keys[i_fitted] == 'nolearning_hilonoise':    
            fitting_agents['fit-nolearning_hilonoise']['sim-'+agent_keys[i_simulated]] = BackInductionNoLearning(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=3)    

        elif agent_keys[i_fitted] == 'anchor_pruning':    
            fitting_agents['fit-anchor_pruning']['sim-'+agent_keys[i_simulated]] = BackInductionAnchorPruning(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=3)    
            
        elif agent_keys[i_fitted] == 'discount_hyperbolic_theta_realprobs_k10.0':    
            fitting_agents['fit-discount_hyperbolic_theta_realprobs_k10.0']['sim-'+agent_keys[i_simulated]] = BackInductionDiscountHyperbolicThetaRealProbskmax30(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=3)        
            
        elif agent_keys[i_fitted] == 'discount_hyperbolic_theta_alpha_k10.0':
             fitting_agents['fit-discount_hyperbolic_theta_alpha_k10.0']['sim-'+agent_keys[i_simulated]] = BackInductionDiscountHyperbolicThetaAlphakmax30(confs0,
                          runs=runs0, 
                          mini_blocks=mini_blocks0, 
                          trials=3, 
                          costs = torch.tensor([0., 0.]),                           planning_depth=i+1)
            

        elif agent_keys[i_fitted] == 'discount_hyperbolic_theta_anchorpruning_k10.0':    
            fitting_agents['fit-discount_hyperbolic_theta_anchorpruning_k10.0']['sim-'+agent_keys[i_simulated]] = BackInductionDiscountHyperbolicThetaAnchorPruningkmax30(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=3)
        
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
comparison_label = '4models_alpha' # 'default-vs_nolearning'
np.savez(datapath + '/modelfit-'+comparison_label+'_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', modelfit)
np.savez(datapath + '/elbo-'+comparison_label+'_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', elbo)
np.savez(datapath + '/responses_depth-'+comparison_label+'_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', responses_depth)
np.savez(datapath + '/states-'+comparison_label+'_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', states)
np.savez(datapath + '/m0_params-'+comparison_label+'_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp', m0)


df_modelfit = pd.DataFrame(modelfit)
#pd.DataFrame(modelfit).to_csv(datapath + '/modelfit.csv')           

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



str_model1 = 'Full planning \n with alpha'
str_model2 = 'Anchor \n pruning'
str_model3 = 'Hyperbolic \n discounting \n alpha' # 'Hyperbolic \n discounting \n real prob.'
str_model3_param = 'Hyperbolic \n discounting \n alpha \n k=10' # 'Hyperbolic \n discounting \n real prob. \n k=10'
str_model4 = 'Hyperbolic \n discounting \n anchor \n pruning'

plot_confusion_inversion_matrix(confusion_mat_BIC, \
                                'Confusion matrix based on BIC: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/confusion_mat_BIC_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(confusion_mat_BIC_hinoise, \
                                'High noise \n Confusion matrix based on BIC: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/confusion_mat_BIC_highnoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(confusion_mat_BIC_lonoise, \
                                'Low noise \n Confusion matrix based on BIC: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/confusion_mat_BIC_lownoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')
    
plot_confusion_inversion_matrix(inversion_mat_BIC, \
                                'Inversion matrix based on BIC: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/inversion_mat_BIC_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')
  
plot_confusion_inversion_matrix(inversion_mat_BIC_hinoise, \
                                'High noise \n Inversion matrix based on BIC: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/inversion_mat_BIC_highnoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(inversion_mat_BIC_lonoise, \
                                'Low noise \n Inversion matrix based on BIC: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/inversion_mat_BIC_lownoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')
     
    
# Same plots for matrices based on rhosquare:
    
plot_confusion_inversion_matrix(confusion_mat_rhosq, \
                                'Confusion matrix based on rhosquare: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/confusion_mat_rhosquare_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(confusion_mat_rhosq_hinoise, \
                                'High noise \n Confusion matrix based on rhosquare: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/confusion_mat_rhosquare_highnoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(confusion_mat_rhosq_lonoise, \
                                'Low noise \n Confusion matrix based on rhosquare: \n  p(best-fitting model | simulated model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/confusion_mat_rhosquare_lownoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')
    
plot_confusion_inversion_matrix(inversion_mat_rhosq, \
                                'Inversion matrix based on rhosquare: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/inversion_mat_rhosquare_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(inversion_mat_rhosq_hinoise, \
                                'High noise \n Inversion matrix based on rhosquare: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/inversion_mat_rhosquare_highnoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')

plot_confusion_inversion_matrix(inversion_mat_rhosq_lonoise, \
                                'Low noise \n Inversion matrix based on rhosquare: \n  p(simulated model | best-fitting model)', \
                                ['', str_model1, str_model2, str_model3, str_model4], \
                                ['', str_model1, str_model2, str_model3_param, str_model4], \
                                datapath + '/inversion_mat_rhosquare_lownoise_' + comparison_label + '_'+str(runs0)+'runs_'+str(n_iter)+'iter_'+str(n_samples)+'samp.png')