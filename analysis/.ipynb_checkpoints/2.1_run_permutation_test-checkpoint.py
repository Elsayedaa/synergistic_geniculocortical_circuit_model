import os
import tensorflow as tf
import pandas as pd
import sys
from itertools import permutations
sys.path.append(os.path.join('c:\\', *os.getcwd().split('\\')[1:-1]))
from sgcc11 import *

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


## initialize X and Y
X = tf.convert_to_tensor([0.02,0.04,0.08,0.1,0.12,0.16,0.2,0.24,0.28,0.32], dtype = tf.float32)

v1_xs_file = os.path.join('c:\\', *os.getcwd().split('\\')[1:-1], 'project_datafiles', 'v1_ori_phase_condition_pcascores_wcomp.pkl')
v1_scores = pd.read_pickle(v1_xs_file)
v1_scores_condition_averaged = np.array([np.array(x) for x in v1_scores.scores.values]).mean(0)
Y_true = v1_scores_condition_averaged[:2,:,:].transpose(0,2,1)

#####################################################################################################################################
## Random optimization run (truly random set of permutations)
perm_choices = np.array(list(permutations(permutations(np.arange(3)), 2)))
random_index = np.random.randint(0,29,size=(4000,7))
random_perm = perm_choices[random_index]
P0 = tf.one_hot(
    random_perm,
    3,
    axis=-1,
)
P0 = tf.transpose(P0, (0,1,2,4,3))
P0 = tf.where(P0 == 1.0, 
                  tf.ones_like(P0) * 10.0,   # large positive
                  tf.ones_like(P0) * -10.0)  # large negative

model = SGCCircuit(bounds=param_bounds, enforce_permutation=P0)
model.initialize_random_parameters(n_v1=2,n_lgn=3,n_sample=4000)
optimizer = Optimize(model, epochs=5000, loss_threshold=0)
optimizer.fit(X, Y_true)
optimizer.save_state(os.path.join('c:\\', *os.getcwd().split('\\')[1:-1], 'project_datafiles', 'permutation_test_random'))

#####################################################################################################################################
## Free optimization run (no enforced identifiability or permutation)
model = SGCCircuit(bounds=param_bounds, identify=False)
model.initialize_random_parameters(n_v1=2,n_lgn=3,n_sample=4000)
optimizer = Optimize(model, epochs=5000, loss_threshold=0)
optimizer.fit(X, Y_true)
optimizer.save_state(os.path.join('c:\\', *os.getcwd().split('\\')[1:-1], 'project_datafiles', 'permutation_test_free'))

#####################################################################################################################################
## Optimize the best 120 permutations
lsi = np.argsort(optimizer.outputs['loss_decay'][-1]) # loss sorting index
params = optimizer.outputs['final_epoch_params']['dLGN_params'][lsi[:120],:,:,:,0,0]
op_perm = np.argsort(np.argsort(params, axis=2), axis=2).transpose(0,3,1,2)

P = tf.one_hot(
    op_perm,
    3,
    axis=-1,
)
P = tf.transpose(P, (0,1,2,4,3))

P = tf.where(P == 1.0, 
                  tf.ones_like(P) * 10.0,   # large positive
                  tf.ones_like(P) * -10.0)  # large negative

P = np.tile(P, [100, 1, 1, 1, 1, 1]).reshape(12000,7,2,3,3)

## Optimize 100 samples for each permutation
index_batch = tf.data.Dataset.from_tensor_slices(np.arange(P.shape[0]))  
index_batch = index_batch.batch(4000)
index_batch = index_batch.prefetch(tf.data.AUTOTUNE)  

optimized_permutation_params = {
    'dLGN_params':[],
    'V1_params':[]
}

permutation_loss_decay = []
for i in index_batch:
    perm = P[i]
    model = SGCCircuit(bounds = param_bounds, identify=True, enforce_permutation=perm)
    model.initialize_random_parameters(2,3,4000)
    optimizer = Optimize(model, epochs=5000, loss_threshold=0)
    optimizer.fit(X, Y_true)

    permutation_loss_decay.append(optimizer.outputs['loss_decay'])
    optimized_permutation_params['dLGN_params'].append(optimizer.outputs['param_history']['dLGN_params'][:,-1])
    optimized_permutation_params['V1_params'].append(optimizer.outputs['param_history']['V1_params'][:,-1])

optimized_permutation_params = {
    'dLGN_params': np.concatenate(optimized_permutation_params['dLGN_params'], axis=0), 
    'V1_params': np.concatenate(optimized_permutation_params['V1_params'], axis=0)
}
result =  {
    'loss_decay': np.concatenate(permutation_loss_decay, axis = 1), 
    'params': optimized_permutation_params
}

with open(os.path.join('c:\\', *os.getcwd().split('\\')[1:-1], 'project_datafiles', 'permutation_test_top120_multi_sample.pkl'), 'wb') as f:
    pickle.dump(result, f)
