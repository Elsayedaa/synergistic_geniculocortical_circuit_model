"""
Python script for fitting the model to the PCA projected 
V1 population data from Skyberg (2022) and Elsayed (2025).
"""

import os
import pickle
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.join('c:\\', *os.getcwd().split('\\')[1:-1]))
from sgcc5 import *

# define the model input
X = tf.convert_to_tensor([0.02,0.04,0.08,0.1,0.12,0.16,0.2,0.24,0.28,0.32], dtype = tf.float32)

## load and preprocess V1 PCA data
v1_xs_file = os.path.join('c:\\', *os.getcwd().split('\\')[1:-1], 'project_datafiles', 'v1_ori_phase_condition_pcascores_wcomp.pkl')
v1_scores = pd.read_pickle(v1_xs_file)
v1_scores_condition_averaged = np.array([np.array(x) for x in v1_scores.scores.values]).mean(0)
Y_true = v1_scores_condition_averaged[:2,:,:].transpose(0,2,1)

## Initialize the parameter bounds
param_bounds = {
    "fts": [0, 175],
    "t": [40, 125],
    "ampc": [0.01, 0.1],
    'ampm': [0,3],
    "ampg": [0.1, 4],
    "ampw": [0.01,0.08],
    "d": [10, 40],
    "inh_d": [0, 40],
    "inh_w": [0, 3],
}

# initialize the model
model = SGCCircuit(param_bounds)
model.initialize_random_parameters(n_v1=Y_true.shape[0], n_lgn=3, n_sample=1000)

# initialize the optimizer
optimizer = Optimize(model, epochs=10000, loss_threshold=0)

# fit the model
optimizer.fit(X, Y_true)

# save the optimized model
optimizer.save_state('sgcc_mop_4_2_26', write=True)