"""
Python script for fitting the model to the full V1 population 
data from Skyberg (2022) and Elsayed (2025).
"""

import os
import pickle
import numpy as np
import pandas as pd
import sys
import mat73
sys.path.append(os.path.join('c:\\', *os.getcwd().split('\\')[1:-1]))
from sgcc import *

# define the model input
X = tf.convert_to_tensor([0.02,0.04,0.08,0.1,0.12,0.16,0.2,0.24,0.28,0.32], dtype = tf.float32)


## load the data
ringach_resp_file = os.path.join('c:\\', *os.getcwd().split('\\')[1:-1], 'project_datafiles', 'ringach_resp.pkl')
ringach_resp = pd.read_pickle(ringach_resp_file)
Y_true = ringach_resp.mean((0,2)).transpose(1,0,2)

# set the parameter bounds
param_bounds = {
    "fts": [-200, 200],
    "t": [40, 150],
    "ampc": [0.01, 0.1],
    'ampm': [0,3],
    "ampg": [0.1, 4],
    "ampw": [0.01,0.1],
    "d": [10, 40],
    "inh_d": [0, 40],
    "inh_w": [0, 3],
}

# initialize the model
model = SGCCircuit(param_bounds)
model.initialize_random_parameters(n_v1=Y_true.shape[0], n_lgn=3, n_sample=25)

# initialize the optimizer
# minimum loss is much smaller for this optimization
# so we set the loss threshold to 0
optimizer = Optimize(model, epochs=20000, loss_threshold=0)

# fit the model
optimizer.fit(X, Y_true)

# save the result
optimizer.save_state(os.path.join('c:\\', *os.getcwd().split('\\')[1:-1], 'project_datafiles', 'joint_exp_fit'), write=True)