import os
import tensorflow as tf
import pandas as pd
import sys
from itertools import permutations, combinations
sys.path.append(os.path.join('c:\\', *os.getcwd().split('\\')[1:-1]))
from sgcc7 import *

# set the parameter bounds
param_bounds = {
    "fts": [0, 175],
    "t": [40, 125],
    "ampc": [0.01, 0.1],
    'ampm': [0,3],
    "ampg": [0.1, 4],
    "ampw": [0.01, 0.08],
    "d": [10, 40],
    "inh_d": [0, 40],
    "inh_w": [0, 3],
}

# Make the permutation matrices
perm_combos = np.array(list(combinations((combinations(permutations(np.arange(3)), 2)), 7)))

perm_combos = tf.one_hot(
    perm_combos,
    3,
    axis=-1,
)
perm_combos = tf.transpose(perm_combos, (0,1,2,4,3))

perm_combos = tf.where(perm_combos == 1.0, 
                  tf.ones_like(perm_combos) * 10.0,   # large positive
                  tf.ones_like(perm_combos) * -10.0)  # large negative

# initialize X and Y
X = tf.convert_to_tensor([0.02,0.04,0.08,0.1,0.12,0.16,0.2,0.24,0.28,0.32], dtype = tf.float32)

v1_xs_file = os.path.join('c:\\', *os.getcwd().split('\\')[1:-1], 'project_datafiles', 'v1_ori_phase_condition_pcascores_wcomp.pkl')
v1_scores = pd.read_pickle(v1_xs_file)
v1_scores_condition_averaged = np.array([np.array(x) for x in v1_scores.scores.values]).mean(0)
Y_true = v1_scores_condition_averaged[:2,:,:].transpose(0,2,1)

new_params = {
    'dLGN_params':[],
    'V1_params':[]
}


## Run a low epoch optimization for all permutations with 100 samples per permutation
## Select the sample for each permutation with the lowest loss value for full optimization
index_batch = tf.data.Dataset.from_tensor_slices(np.arange(perm_combos.shape[0]))  
index_batch = index_batch.batch(10)
index_batch = index_batch.prefetch(tf.data.AUTOTUNE)
for i in index_batch:
    n_sample = 100
    batch_size = i.shape[0]

    model = SGCCircuit(bounds=param_bounds, enforce_permutation=tf.tile(perm_combos.numpy()[i], [n_sample,1,1,1,1]))
    model.initialize_random_parameters(2,3, n_sample=batch_size*n_sample)

    optimizer = Optimize(model, epochs=100)
    optimizer.fit(X, Y_true)

    sl = optimizer.outputs['loss_decay'][-1,:].reshape(n_sample,batch_size) # sample loss
    bli = np.array([np.where(sl[:,i] == sl[:,i].min())[0][0] for i in range(batch_size)]) # best loss index
    fe = optimizer.outputs['final_epoch_params'] ## final epoch
    dlgn = fe['dLGN_params'].reshape(n_sample,batch_size,2,3,7,1,1)
    v1 = fe['V1_params'].reshape(n_sample,batch_size,2,1,2,1,1)
    for i in range(batch_size):
        new_params['dLGN_params'].append(dlgn[bli[i],i])
        new_params['V1_params'].append(v1[bli[i],i])

## Fully optimize the best sample for each permutation
index_batch = tf.data.Dataset.from_tensor_slices(np.arange(perm_combos.shape[0]))  
index_batch = index_batch.batch(1000)
index_batch = index_batch.prefetch(tf.data.AUTOTUNE)  

dlgn_params = tf.stack(new_params['dLGN_params'], axis = 0)
v1_params = tf.stack(new_params['V1_params'], axis = 0)

optimized_permutation_params = {
    'dLGN_params':[],
    'V1_params':[]
}

permutation_loss_decay = []
for i in index_batch:
    perm = perm_combos.numpy()[i]
    lgn_param = dlgn_params.numpy()[i]
    v1_param = v1_params.numpy()[i]

    
    model = SGCCircuit(bounds = param_bounds, identify=True, enforce_permutation=perm)
    model.load_saved_parameters({'dLGN_params':lgn_param, 'V1_params': v1_param})
    optimizer = Optimize(model, epochs=5000, loss_threshold=0)
    optimizer.fit(X, Y_true)

    permutation_loss_decay.append(optimizer.outputs['loss_decay'])
    optimized_permutation_params['dLGN_params'].append(optimizer.outputs['final_epoch_params']['dLGN_params'])
    optimized_permutation_params['V1_params'].append(optimizer.outputs['final_epoch_params']['V1_params'])

optimized_permutation_params = {
    'dLGN_params': np.concatenate(optimized_permutation_params['dLGN_params'], axis=0), 
    'V1_params': np.concatenate(optimized_permutation_params['V1_params'], axis=0)
}
result =  {
    'loss_decay': np.concatenate(permutation_loss_decay, axis = 1), 
    'params': optimized_permutation_params
}

# with open('permutation_test_results.pkl', 'wb') as f:
#     pickle.dump(result, f)