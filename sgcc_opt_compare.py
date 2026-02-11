import os
import numpy as np
import pandas as pd
import pickle
from sgcc_model12 import *

## Set the selection of optimizers to compare
optimizer_selection = {
    'Adam': tf.keras.optimizers.Adam(),
    'AdamW': tf.keras.optimizers.AdamW(),
    'Adamax': tf.keras.optimizers.Adamax(),
    'Adagrad': tf.keras.optimizers.Adagrad(),
    'Adafactor': tf.keras.optimizers.Adafactor(),
    'Ftrl': tf.keras.optimizers.Ftrl(),
    'Lion': tf.keras.optimizers.Lion(),
    'Nadam': tf.keras.optimizers.Nadam(),
}

## Set the X inputs
X = tf.convert_to_tensor([0.02,0.04,0.08,0.1,0.12,0.16,0.2,0.24,0.28,0.32], dtype = tf.float32)

## Load and process Y true
v1_xs_file = f"v1_ori_phase_condition_pcascores.pkl"
#v1_scores = pd.read_pickle(os.path.join('c:\\',*os.getcwd().split('\\')[1:-1], v1_xs_file))
v1_scores = pd.read_pickle(os.path.join('/',*os.getcwd().split('/')[:-1], v1_xs_file))
v1_scores_condition_averaged = np.array([np.array(x) for x in v1_scores.scores.values]).mean(0)
Y_true = v1_scores_condition_averaged[:2,:,:].transpose(0,2,1)

## Set the bounds
param_bounds = {
    "fts": [20, 200],
    "t": [20, 100],
    "ats": [-3, 0],
    "a": [0.1, 4],
    "d": [10, 40],
    "inh_d": [0, 40],
    "inh_w": [0, 3],
}

## Run each optimization
for key, selection in optimizer_selection.items():

    sgcc = SGCCircuit(param_bounds)
    sgcc.initialize_random_parameters(n_v1=2, n_lgn=3, n_sample=500)
    
    optimizer = Optimize(sgcc, epochs=5000)
    optimizer.optimizer = selection
    loss_decay = optimizer.fit(X, Y_true)

    optimizer.save_state(f"sgcc_optimization_via_{key}.pkl", write=True)
    
    del sgcc
    del optimizer
    del loss_decay