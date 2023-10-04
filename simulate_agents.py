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

runs0 = 30 # 1000 # 20 # 40            #number of simulations 
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
    
agent_keys = ['rational', 'anchor_pruning', 'discount_hyperbolic_theta_realprobs_k10.0', \
              'discount_hyperbolic_theta_anchorpruning_k10.0']   #  Discounting mit Real probs oder mit alpha?    

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
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and alpha parameters as a normal distribution around a certain value
            #m0['rational'] = torch.tensor([1.099, 0., 0.0])# beta= 3, because 1.099=np.log(3); alpha = 0.5 (too high!!!)
            m0['rational'] = torch.tensor([1.099, 0., -2.])#            

        elif agent_key == 'nolearning_hilonoise':    
            agents['nolearning_hilonoise'] = BackInductionNoLearning(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and alpha parameters as a normal distribution around a certain value
            #m0['rational'] = torch.tensor([1.099, 0., 0.0])# beta= 3, because 1.099=np.log(3);
            m0['nolearning_hilonoise'] = torch.tensor([1.099, 0.])#     
            
        elif agent_key == 'discount_noise_theta_gamma0.7': 
            agents['discount_noise_theta_gamma0.7'] = BackInductionDiscountNoiseTheta(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and gamma (discounting) parameters 
            m0['discount_noise_theta_gamma0.7'] = torch.tensor([1.099, 0., 0.85])# beta= 3, because 1.099=np.log(3) // gamma=0.7=sigmoid(0.85)

        elif agent_key == 'discount_noise_theta_gamma0.3': 
            agents['discount_noise_theta_gamma0.3'] = BackInductionDiscountNoiseTheta(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and gamma (discounting) parameters 
            m0['discount_noise_theta_gamma0.3'] = torch.tensor([1.099, 0., -0.85])# beta= 3, because 1.099=np.log(3) // gamma=0.3=sigmoid(-0.85)  

        elif agent_key == 'anchor_pruning': 
            agents['anchor_pruning'] = BackInductionAnchorPruning(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and gamma (discounting) parameters 
            m0['anchor_pruning'] = torch.tensor([1.099, 0.])# beta= 3, because 1.099=np.log(3) 

        elif agent_key == 'discount_noise_theta_learnprobs':    
            agents['discount_noise_theta_learnprobs'] = BackInductionDiscountNoiseThetaLearnprobs(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and alpha parameters as a normal distribution around a certain value
            #m0['discount_noise_theta_learnprobs'] = torch.tensor([1.099, 0., -2, 10.0])# beta= 3, because 1.099=np.log(3), theta=0, alpha=0.1, gamma=0.99 - same performance as a learning rational agent with alpha=0.1
            m0['discount_noise_theta_learnprobs'] = torch.tensor([1.099, 0., -2, 0.85])# beta= 3, because 1.099=np.log(3), theta=0, alpha=0.1, gamma=0.7
   
        elif agent_key == 'discount_noise_theta_anchor_pruning':    
            agents['discount_noise_theta_anchor_pruning'] = BackInductionDiscountNoiseThetaAnchorPruning(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and alpha parameters as a normal distribution around a certain value
            m0['discount_noise_theta_anchor_pruning'] = torch.tensor([1.099, 0., 0.85])# beta= 3, because 1.099=np.log(3), theta=0, gamma=0.7
 
        elif agent_key == 'discount_noise_theta_fitprobs':    
            agents['discount_noise_theta_fitprobs'] = BackInductionDiscountNoiseThetaFitprobs(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and alpha parameters as a normal distribution around a certain value
            #m0['discount_noise_theta_fitprobs'] = torch.tensor([1.099, 0., 10, 0, 0.85])# beta= 3, because 1.099=np.log(3), theta=0, prob_lonoise=0.99, prob_hinoise=0.5, gamma=0.7: 1716 points on average - very good!
            #m0['discount_noise_theta_fitprobs'] = torch.tensor([1.099, 0., 2.2, 0, 0.85])# beta= 3, because 1.099=np.log(3), theta=0, prob_lonoise=0.9, prob_hinoise=0.5, gamma=0.7: 1711 points on average - very good!
            #m0['discount_noise_theta_fitprobs'] = torch.tensor([1.099, 0., 2.2, 2.2, 0.85])# beta= 3, because 1.099=np.log(3), theta=0, prob_lonoise=0.9, prob_hinoise=0.9, gamma=0.7: 1673 points on average - still pretty good!           
            m0['discount_noise_theta_fitprobs'] = torch.tensor([1.099, 0., 10, 10, 0.85]) # beta= 3, because 1.099=np.log(3), theta=0, prob_lonoise=0.99, prob_hinoise=0.99, gamma=0.7: 1658 points on average - still pretty good!                      
  
        elif agent_key == 'random':    
            agents['random'] = BackInduction(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)        
            m0['random'] = torch.tensor([-10, 0., 0.]) # beta close to zero
        
   
        elif agent_key == 'discount_noise_theta_realprobs_gamma0.7':
             agents['discount_noise_theta_realprobs_gamma0.7'] = BackInductionDiscountNoiseThetaRealprobs(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and gamma (discounting) parameters 
             m0['discount_noise_theta_realprobs_gamma0.7'] = torch.tensor([1.099, 0., 0.85])# beta= 3, because 1.099=np.log(3) // gamma=0.7=sigmoid(0.85)

        elif agent_key == 'discount_noise_theta_realprobs_gamma0.3':
             agents['discount_noise_theta_realprobs_gamma0.3'] = BackInductionDiscountNoiseThetaRealprobs(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and gamma (discounting) parameters 
             m0['discount_noise_theta_realprobs_gamma0.3'] = torch.tensor([1.099, 0., -0.85])# beta= 3, because 1.099=np.log(3) // gamma=0.3=sigmoid(-0.85)
             
        elif agent_key == 'discount_hyperbolic_theta_realprobs_k1.0':
             agents['discount_hyperbolic_theta_realprobs_k1.0'] = BackInductionDiscountHyperbolicThetaRealProbskmax30(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_realprobs_k1.0'] = torch.tensor([1.099, 0., -3.35])# beta= 3, because 1.099=np.log(3) // k=1 = 30*sigmoid(-3.35)

        elif agent_key == 'discount_hyperbolic_theta_realprobs_k10.0':
             agents['discount_hyperbolic_theta_realprobs_k10.0'] = BackInductionDiscountHyperbolicThetaRealProbskmax30(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_realprobs_k10.0'] = torch.tensor([1.099, 0., -0.7])# beta= 3, because 1.099=np.log(3) // k=10 = 30*sigmoid(-0.7)

             
        elif agent_key == 'discount_hyperbolic_theta_realprobs_k20.0':
             agents['discount_hyperbolic_theta_realprobs_k20.0'] = BackInductionDiscountHyperbolicThetaRealProbskmax30(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_realprobs_k20.0'] = torch.tensor([1.099, 0., 0.7])# beta= 3, because 1.099=np.log(3) // k=20 = 30*sigmoid(0.7)

        elif agent_key == 'discount_hyperbolic_theta_anchorpruning_k1.0':
             agents['discount_hyperbolic_theta_anchorpruning_k1.0'] = BackInductionDiscountHyperbolicThetaAnchorPruningkmax30(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_anchorpruning_k1.0'] = torch.tensor([1.099, 0., -3.35])# beta= 3, because 1.099=np.log(3) //  k=1 = 30*sigmoid(-3.35)

        elif agent_key == 'discount_hyperbolic_theta_anchorpruning_k10.0':
             agents['discount_hyperbolic_theta_anchorpruning_k10.0'] = BackInductionDiscountHyperbolicThetaAnchorPruningkmax30(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
                          planning_depth=i+1)

            # set beta, theta and gamma (discounting) parameters 
             m0['discount_hyperbolic_theta_anchorpruning_k10.0'] = torch.tensor([1.099, 0., -0.7])# beta= 3, because 1.099=np.log(3) // k=10 = 30*sigmoid(-0.7)


        elif agent_key == 'discount_hyperbolic_theta_anchorpruning_k20.0':
             agents['discount_hyperbolic_theta_anchorpruning_k20.0'] = BackInductionDiscountHyperbolicThetaAnchorPruningkmax30(confs0,
                          runs=runs0,
                          mini_blocks=mini_blocks0,
                          trials=3,
                          costs = torch.tensor([0., 0.]), 
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
        points0 = (costs0[responses0] + fuel0[simulations[agent_key][-1].outcomes])  #reward for landing on a certain planet in simulation

        points0[simulations[agent_key][-1].outcomes < 0] = 0 #set MB in which points go below 0 on 0 ?
        performance[agent_key].append(points0.sum(dim=-1))   #sum up the gains 
    
        trans_pars_depth[agent_key].append(trans_pars0[agent_key])
        points_depth[agent_key].append(points0)
        responses_depth[agent_key].append(responses0)

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


# Question:
    # How similar are the actions predicted by the different agent models?

corr_pvals = np.nan * np.ones(len(agent_keys))
corr_rhovals = np.nan * np.ones(len(agent_keys))

corr_pvals_agents = np.nan * np.ones([len(agent_keys), runs0])
corr_rhovals_agents = np.nan * np.ones([len(agent_keys), runs0])

action_overlap_first = np.nan * np.ones([len(agent_keys), runs0])
action_overlap_all = np.nan * np.ones([len(agent_keys), runs0])


for i_agent in range(len(agent_keys)):
    agent_key = agent_keys[i_agent]
    corr_rhovals[i_agent], corr_pvals[i_agent] = st.pearsonr(responses_depth['rational'][2][0,:,0], responses_depth[agent_key][2][0,:,0])
    for i_run in range(runs0):
        corr_rhovals_agents[i_agent, i_run], corr_pvals_agents[i_agent, i_run] = st.pearsonr(responses_depth['rational'][2][i_run, :, 0], responses_depth[agent_key][2][i_run, :, 0])    # First action only
        #corr_rhovals_agents[i_agent, i_run], corr_pvals_agents[i_agent, i_run] = st.pearsonr(responses_depth['rational'][2][i_run, :, :].flatten(), responses_depth[agent_key][2][i_run, :, :].flatten())  # All actions          
        
        action_overlap_first[i_agent, i_run] = (responses_depth['rational'][2][i_run, :, 0] == responses_depth[agent_key][2][i_run, :, 0]).numpy().mean()
        action_overlap_all[i_agent, i_run] = (responses_depth['rational'][2][i_run, :, :].flatten() == responses_depth[agent_key][2][i_run, :, :].flatten()).numpy().mean()
        
    # take mean across all agents ?! 
    # corr_rhovals_agents.mean(1)

#np.transpose(pd.DataFrame(final_points)).to_csv(datapath + '/mean_points_agents.csv')

# In[3]
# plotting agent's behavior for planning depth 1:3
'''#
for i in range(3):
    plt.figure()
    #plt.plot(performance['rational'][i].numpy().cumsum(axis=-1).T + starting_points, 'b')
    plt.plot(performance['rational'][i].numpy().cumsum(axis=-1).T + starting_points, 'C'+str(i))    
    plt.ylabel('points')
    plt.xlabel('nb of mini-blocks')
    plt.ylim([0,2700])
    #plt.savefig('score_PD'+str(i+1)+'.pdf', bbox_inches='tight', transparent=True, dpi=600)    
    #plt.savefig('score_PD'+str(i+1)+'.png', bbox_inches='tight', transparent=True, dpi=600)        


plt.figure(figsize=(10, 5))
labels = [r'd=1', r'd=2', r'd=3']
plt.hist(torch.stack(performance['rational']).numpy().cumsum(axis=-1)[..., -1].T+ starting_points, bins=30, stacked=True)
plt.legend(labels)
plt.ylabel('count')
plt.xlabel('score')
#plt.savefig('finalscore_exp.pdf', bbox_inches='tight', transparent=True, dpi=600)
#plt.savefig('finalscore_exp.png', bbox_inches='tight', transparent=True, dpi=600)


for i in range(3):
    plt.figure()
    plt.plot(performance['random'][i].numpy().cumsum(axis=-1).T + starting_points, 'C'+str(i))    
    plt.ylabel('points')
    plt.xlabel('nb of mini-blocks')
    plt.ylim([-500,2700])
'''



plt.figure(figsize=(12,8))
fsize=9
bp1 = plt.boxplot([points_depth['rational'][2][:,:,:].numpy().sum(2).sum(1), #.mean(0),
                   points_depth['discount_noise_theta_gamma0.7'][2][:,:,:].numpy().sum(2).sum(1),
                   points_depth['discount_noise_theta_gamma0.3'][2][:,:,:].numpy().sum(2).sum(1),
                   points_depth['anchor_pruning'][2][:,:,:].numpy().sum(2).sum(1),
                   points_depth['discount_noise_theta_learnprobs'][2][:,:,:].numpy().sum(2).sum(1),
                   points_depth['discount_noise_theta_anchor_pruning'][2][:,:,:].numpy().sum(2).sum(1),
                   points_depth['discount_noise_theta_gamma0.7'][2][:,:,:].numpy().sum(2).sum(1),
                   points_depth['discount_noise_theta_gamma0.3'][2][:,:,:].numpy().sum(2).sum(1)                   
                   ], \
                #positions=[1, 1.6, 2.2, 2.8, 3.4, 4],                                   
                positions=np.arange(1, 5.799, 0.6), \
                showmeans=True,
                patch_artist=True, boxprops=dict(facecolor="C0", alpha=0.3))
bp2 = plt.boxplot([points_depth['rational'][1][:,:,:].numpy().sum(2).sum(1),
                   points_depth['discount_noise_theta_gamma0.7'][1][:,:,:].numpy().sum(2).sum(1),
                   points_depth['discount_noise_theta_gamma0.3'][1][:,:,:].numpy().sum(2).sum(1),
                   points_depth['anchor_pruning'][1][:,:,:].numpy().sum(2).sum(1),
                   points_depth['discount_noise_theta_learnprobs'][1][:,:,:].numpy().sum(2).sum(1),
                   points_depth['discount_noise_theta_anchor_pruning'][1][:,:,:].numpy().sum(2).sum(1),
                   points_depth['discount_noise_theta_gamma0.7'][1][:,:,:].numpy().sum(2).sum(1),                   
                   points_depth['discount_noise_theta_gamma0.3'][1][:,:,:].numpy().sum(2).sum(1)                   
                   ], \
                #positions=[4.8, 5.4, 6.0, 6.6, 7.2, 7.8], 
                positions=np.arange(6.0, 10.799, 0.6), \
                showmeans=True,
                patch_artist=True, boxprops=dict(facecolor="C1", alpha=0.3))   
ax=plt.gca()
ax.tick_params(axis='y', labelsize=15)
ax.set_xticklabels(['PD3 \n rational', 'PD3 \n discount \n $\gamma=0.7$', 'PD3 \n discount \n $\gamma=0.3$', \
                    'PD3 \n anchor \n pruning', 'PD3 \n discount \n learnprobs \n $\gamma=0.7$',\
                    'PD3 \n discount \n anchor \n pruning \n $\gamma=0.7$', \
                    'PD3 \n discount \n realprobs \n $\gamma=0.7$', 'PD3 \n discount \n realprobs \n $\gamma=0.3$', \
                    'PD2 \n rational','PD2 \n discount \n $\gamma=0.7$', 'PD2 \n discount \n $\gamma=0.3$', \
                    'PD2 \n anchor \n pruning','PD2 \n discount \n learnprobs \n $\gamma=0.7$', \
                    'PD2 \n discount \n anchor \n pruning \n $\gamma=0.7$', \
                    'PD3 \n discount \n realprobs \n $\gamma=0.7$', 'PD2 \n discount \n realprobs \n $\gamma=0.3$', \
                    ], fontsize=fsize)
#ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['All'], loc='upper center')
plt.ylabel('Mean points', fontsize=15) #  per miniblock
#Tstat, pval = st.ttest_ind(points_depth['rational'][0][:,:,:].numpy().sum(2).mean(0)[index_difficult], \
#             points_depth['rational'][2][:,:,:].numpy().sum(2).mean(0)[index_difficult])
#plt.text(3.1, 44, '***')    
#plt.text(3.1, 42, 'p='+str(round(pval,5)), fontsize=13)    
#plt.savefig('Distribution_agents_points.png', bbox_inches='tight', dpi=600)  
#plt.savefig(datapath+'/Distribution_agents_points.png', bbox_inches='tight', dpi=600)  
plt.savefig(datapath+'/Distribution_agents_points_subset.png', bbox_inches='tight', dpi=600)  


#sim_number0 = 0                                             # here we set the planning depth for the 
                                                            # inference (where 0 referd to PD = 1)

sim_number0 = 2
# This is where inference starts

n_iter = 10 # 1000 # 100

agent2_rational = {}
agent2_anchor_pruning = {}
agent2_discount_noise_theta_gamma07 = {}

fitting_agents = {}
elbo = {}
modelfit = {}

datapath = 'P:/037/B3_BEHAVIORAL_STUDY/04_Experiment/Analysis_Scripts/SAT_Results/Results_crossfitting'

for i_fitted in range(len(agent_keys)):
    fitting_agents['fit-'+agent_keys[i_fitted]] = {}
    elbo['fit-'+agent_keys[i_fitted]] = {}    
    modelfit['fit-'+agent_keys[i_fitted]] = {}     
   
    for i_simulated in range(len(agent_keys)):    
        print('fitted: '+agent_keys[i_fitted] + ', simulated: '+agent_keys[i_simulated]) 
        
        fitting_agents['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ] = 'fit-'+str(i_fitted)+'-sim-'+str(i_simulated)
        elbo['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ] = 'fit-'+str(i_fitted)+'-sim-'+str(i_simulated)        
        modelfit['fit-'+agent_keys[i_fitted]][ 'sim-'+agent_keys[i_simulated] ] = {}

        responses0 = simulations[agent_keys[i_simulated]][sim_number0].responses.clone()
        mask0 = ~torch.isnan(responses0)

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

            
        # plotting the ELBO for the given inference 
        plt.figure()
        plt.plot(infer.loss[:]) #change section you want to see here 
        #plt.plot(np.log(infer.loss[100:])) #change section you want to see here 
        plt.ylabel('ELBO')  
        plt.xlabel('x')
        plt.savefig('ELBO_PD3_sim_' + simulated_agent_key + '_fit_' + fitting_agent_key + '.pdf', dpi=600, bbox_inches='tight')
        plt.savefig('ELBO_PD3_sim_' + simulated_agent_key + '_fit_' + fitting_agent_key + '.png', dpi=600, bbox_inches='tight')

        labels = [r'$\tilde{\beta}$', r'$\theta$',   r'$\tilde{\alpha}$'] #these refer to the next plots and calculations
'''        