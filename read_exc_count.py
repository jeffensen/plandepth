# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 09:38:31 2023

@author: goenner
"""

import pylab as plt
import numpy as np
import pandas as pd

datapath = 'H:\Sesyn\TRR265-B09\Analysis_SAT-PD2_Sophia\SAT_PD_Inference_Scripts\Results_discount_Noise_theta'

df_exc_pd1_oa = pd.read_csv(datapath + '\exc_PD1_oa_lossaversion_discount_Noise_theta_0cost.csv')
df_exc_pd2_oa = pd.read_csv(datapath + '\exc_PD2_oa_lossaversion_discount_Noise_theta_0cost.csv')
df_exc_pd3_oa = pd.read_csv(datapath + '\exc_PD3_oa_lossaversion_discount_Noise_theta_0cost.csv')