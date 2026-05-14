import tensorflow as tf
import numpy as np
import pickle

DTYPE = tf.float32
class SGCCircuit(tf.Module):

    def __init__(
            self,
            bounds,
            identify = True,
            enforce_permutation = False
    ):
        super().__init__(name = 'SGCC')

        ## Set the bounds on the trainable parameters
        self.bounds = bounds
        self.GBOUND = 6

        self.identify = identify
        self.permutation = enforce_permutation
        if self.identify == False:
            if str(type(self.permutation)) in [
                "<class 'numpy.ndarray'>", 
                "<class 'tensorflow.python.framework.ops.EagerTensor'>"
            ]:
                raise _PermPassedWithNoID()
            elif str(type(self.permutation)) == "<class 'str'>":
                ## The default permutation: dLGN0<=dLGN1<=dLGN2 for all parameter classes
                if ((self.permutation == 'any') or (self.permutation == 'default')):
                    raise _PermPassedWithNoID()

    def variable_transformer(self, x, lower, upper, op = 'scale'):
        if op == 'scale':
            transformed = lower + (upper - lower) * tf.sigmoid(x)

        if op == 'normalize':
            n = (x - lower)/(upper-lower)
            transformed = tf.math.log(n/(1-n))
            
        return transformed
    
    def initialize_neuron_order_permutation(self):
        
        # skip if indentifiability constraint is not required
        if self.identify:
            pass
        else:
            return None
        
        if str(type(self.permutation)) in [
            "<class 'numpy.ndarray'>", 
            "<class 'tensorflow.python.framework.ops.EagerTensor'>"
        ]:
            ## implement a user-defined permutation matrix
            soft_perm = self.permutation
            self.soft_perm = tf.Variable(soft_perm,
                                        name = 'permutation_matrix', 
                                        trainable = False, 
                                        dtype = DTYPE,
                                    )
            
        elif str(type(self.permutation)) == "<class 'str'>" or (self.permutation==False):
            ## The default permutation: dLGN0<=dLGN1<=dLGN2 for all parameter classes
            if (self.permutation == 'default') or (self.permutation == False):
                self.soft_perm = None
            
            elif self.permutation == 'any':
                ## Enforces an identifiable permutation by optimizing a permutation
                ## matrix with values chosen randomly from a normal distribution & 
                ## applying Sinkhorn relaxation. Not gauranteed to have sharp 1s and 
                ## 0s and will probably distort the parameters going into model.predict().
                ## Use with caution. 
                soft_perm = tf.random.normal((7, self.n_v1, self.n_lgn, self.n_lgn), mean = 0, stddev = 1)
                soft_perm = tf.tile(
                    tf.expand_dims(soft_perm, axis=0),
                    [self.n_sample, 1, 1, 1, 1]
                )
                self.soft_perm = tf.Variable(soft_perm,
                                            name = 'permutation_matrix', 
                                            trainable = True, 
                                            dtype = DTYPE,
                                        )

    def initialize_random_parameters(self, n_v1, n_lgn, n_sample):
        self.n_sample = n_sample
        self.n_v1 = n_v1
        self.n_lgn = n_lgn

        ## Fixed parameters
        self.T = tf.reshape(tf.cast(tf.linspace(0,250,250), dtype = DTYPE), [1, 1, 1, 1,-1])
        self.mid = tf.cast(tf.fill([self.n_sample, n_v1, n_lgn, 1, 1], 0), dtype = DTYPE)

        ## Trainable parameters
        dlgn_raw = tf.random.normal([self.n_sample, n_v1, n_lgn, 7, 1, 1], mean = 0, stddev = 1)
        v1_scaled = tf.random.normal([self.n_sample, n_v1, 1, 2, 1, 1])

        self.dlgn_raw = tf.Variable(dlgn_raw, 
                                       name = 'dLGN_params', 
                                       trainable = True, 
                                       dtype = DTYPE,
                                       )
        
        self.v1_scaled = tf.Variable(v1_scaled, 
                                     name = 'V1_params', 
                                     trainable = True, 
                                     dtype = DTYPE,
                                     )
        
        self.initialize_neuron_order_permutation()
        self.update_transform()

    def load_saved_parameters(self, parameters):
        self.n_sample, self.n_v1, self.n_lgn = parameters['dLGN_params'].shape[:3]

        ## Fixed parameters
        self.T = tf.reshape(tf.cast(tf.linspace(0,250,250), dtype = DTYPE), [1, 1, 1, 1,-1])
        self.mid = tf.cast(tf.fill([self.n_sample, self.n_v1, self.n_lgn, 1, 1], 0), dtype = DTYPE)

        self.dlgn_raw = tf.Variable(parameters['dLGN_params'], name = 'dLGN_params', trainable = True, dtype = DTYPE)
        self.v1_scaled = tf.Variable(parameters['V1_params'], name = 'V1_params', trainable = True, dtype = DTYPE)

        self.initialize_neuron_order_permutation()
        self.update_transform()

    def sinkhorn(self, log_alpha, n_iters=10, temperature=1):
        log_alpha = log_alpha/temperature
        for i in range(n_iters):
            log_alpha = log_alpha - tf.math.reduce_logsumexp(log_alpha, axis=-1, keepdims=True)
            log_alpha = log_alpha - tf.math.reduce_logsumexp(log_alpha, axis=-2, keepdims=True)
        return tf.exp(log_alpha)

    @property # apply the necessary reparameterizations
    def dlgn_scaled(self):
        if self.identify:

            ## apply identifiability constraint (dLGN0<=dLGN1<=dLGN2)
            leading_unit = self.dlgn_raw[:, :, 0:1, ...] # leading dLGN unit
            trailing_offsets = tf.nn.softplus( # values added to consecutive dLGN units (enforce positive vals)
                self.dlgn_raw[:, :, 1:, ...]
            ) 

            # get the values of the trailing units by adding the offsets to the leading unit
            trailing_units = leading_unit + tf.cumsum(trailing_offsets, axis=2)
            ordered_units = tf.concat([leading_unit, trailing_units], axis=2)

            if np.any(self.soft_perm):
                # apply the order permutation
                P = self.sinkhorn(self.soft_perm)
                D = ordered_units[:,:,:,:,0,0]
                D = tf.transpose(D, (0,3,1,2))[:,:,:,None,:]
                D = tf.squeeze(tf.matmul(D, P))
                D = tf.transpose(D, (0,2,3,1))[:,:,:,:,None,None]
                print('Model initialized with pre-defined permutation.')
                if type(self.permutation) == str:
                    print(f'permutation = {self.permutation}')
                else:
                    print('permutation = user defined')
                return D
            
            else:
                print('Model initialized with default permutation')
                return ordered_units
        else:
            return self.dlgn_raw
    
    def inverse_dlgn_property(self, ordered_units):
        ## invert the dLGN reparameterization (for loading parameters and reinitialization)

        leading_unit = ordered_units[:,:,0]
        trailing_offsets = ordered_units[:,:,1:] - ordered_units[:,:,:-1]
        trailing_offsets = tf.math.log(tf.math.expm1(trailing_offsets)) # apply an approximate inverse softplus
        dlgn_recov = np.concatenate([leading_unit[:,:,None,:,:,:], trailing_offsets], axis = 2)

        return dlgn_recov

    def set_parameter(
            self,
            brain_area = str,
            samples = list,
            dlgn_units = list,
            v1_units = list,
            param = str,
            value = int
    ):
        
        control_map = {
            'dLGN': {'fts': 0, 't': 1, 'ampc': 2, 'ampm': 3, 'ampg': 4, 'ampw': 5, 'd': 6},
            'V1': {'inh_d': 0, 'inh_w': 1}
        }

        if brain_area == 'dLGN':
            bounds = [x for x in self.bounds.values()][:-2]
            lower, upper = bounds[control_map['dLGN'][param]]
            for v1_unit in v1_units:
                for dlgn_unit in dlgn_units:
                    for sample in samples:
                        
                        # update the scaled reparameterized variable
                        new_dlgn_scaled = tf.tensor_scatter_nd_update(
                            self.dlgn_scaled, 
                            [[sample,v1_unit,dlgn_unit,control_map['dLGN'][param],0,0]], 
                            [self.variable_transformer(value, lower, upper, op = 'normalize')]
                        )

                        if self.identify:
                            # apply the inverse property
                            self.dlgn_raw = tf.Variable(self.inverse_dlgn_property(new_dlgn_scaled), 
                                        name = 'dLGN_params', 
                                        trainable = True, 
                                        dtype = DTYPE,
                                        )
                            
                        else :
                            self.dlgn_raw = tf.Variable(new_dlgn_scaled, 
                                        name = 'dLGN_params', 
                                        trainable = True, 
                                        dtype = DTYPE,
                                        )                            

        if brain_area == 'V1':
            bounds = [x for x in self.bounds.values()][-2:]
            lower, upper = bounds[control_map['V1'][param]]

            for v1_unit in v1_units:
                for sample in samples:
                    self.v1_scaled = tf.tensor_scatter_nd_update(
                        self.v1_scaled, 

                        # the first 0 is the dLGN unit placeholder
                        [[sample,v1_unit,0,control_map['V1'][param],0,0]], 
                        [self.variable_transformer(value, lower, upper, op = 'normalize')]
                    )

    def gaussian(
        self,
        support, # domain of the function
        ctr, # mean of the function
        amp, # amplitude of the function
        mid, # midline of the function
        std # bandwidth parameter
    ):

        f = tf.math.exp(-1*(((support-ctr)**2)/(2*std**2)))

        return mid+(amp*f)
    
    def gaussian_gradient( # calculate the derivative of a gaussian
        self,
        support,
        ctr, 
        amp, 
        mid, 
        std 
    ):
        fg = self.gaussian(
            support, ctr, amp, mid, std 
        )
        
        dGdX = -fg * ((support - ctr)/(std**2))
        return dGdX

    def f_amp(self, sf, ampc, ampm, ampg, ampw):
        return self.gaussian(
            sf, ampc, ampg, ampm, ampw
        )

    def f_t(self, fts, t, sf):
        return fts*sf+t

    def frf(
        self,
        sfs, 
        fts, t,
        ampc, ampm, 
        ampg, ampw,
        d
    ):
        # Make the rest of the variables broadcastable
        f = self.gaussian(
            self.T,
            self.f_t(fts, t, sfs),
            self.f_amp(sfs,ampc, ampm, ampg, ampw),
            self.mid, d
        )

        return f

    def predict(
            self,
            sfs,
    ):
        # reshape the SF input to broadcast across multiple exploration samples
        self.sfs = tf.reshape(sfs, [1,1,1,-1,1])

        # Transform the scaled variables before doing calculations
        self.update_transform()

        self.dlgn_exc = self.frf(
            self.sfs,
            self.params['dLGN_params'][:,:,:,0,:,:], 
            self.params['dLGN_params'][:,:,:,1,:,:],
            self.params['dLGN_params'][:,:,:,2,:,:],
            self.params['dLGN_params'][:,:,:,3,:,:],
            self.params['dLGN_params'][:,:,:,4,:,:],
            self.params['dLGN_params'][:,:,:,5,:,:],
            self.params['dLGN_params'][:,:,:,6,:,:],
        )

        self.dlgn_inh = self.params['V1_params'][:,:,:,1,:,:] * self.frf(
            self.sfs,
            self.params['dLGN_params'][:,:,:,0,:,:],
            self.params['dLGN_params'][:,:,:,1,:,:] + self.params['V1_params'][:,:,:,0,:,:],
            self.params['dLGN_params'][:,:,:,2,:,:],
            self.params['dLGN_params'][:,:,:,3,:,:],
            self.params['dLGN_params'][:,:,:,4,:,:],
            self.params['dLGN_params'][:,:,:,5,:,:],
            self.params['dLGN_params'][:,:,:,6,:,:],
        )

        self.v1_exc = tf.reduce_sum(self.dlgn_exc, axis = 2)
        self.v1_inh = tf.reduce_sum(self.dlgn_inh, axis = 2)
        self.Y = self.v1_exc - self.v1_inh

        return self.Y
    
    def update_transform(self):
        if str(type(self.bounds)) == "<class 'tensorflow.python.trackable.data_structures.ListWrapper'>":
            self.update_transform_separate()
        else:
            self.update_transform_joint()
            
    
    def update_transform_joint(self):

        # Transform the normalized variables back into their meaningful scale
        dlgn_bounds = tf.convert_to_tensor([x for x in self.bounds.values()])[:-2]
        dlgn_lower = tf.reshape(dlgn_bounds[:,0], [1,1,1,-1,1,1])
        dlgn_upper = tf.reshape(dlgn_bounds[:,1], [1,1,1,-1,1,1])

        v1_bounds = tf.convert_to_tensor([x for x in self.bounds.values()])[-2:]
        v1_lower = tf.reshape(v1_bounds[:,0], [1,1,1,-1,1,1])
        v1_upper = tf.reshape(v1_bounds[:,1], [1,1,1,-1,1,1])


        self.params = {
            'dLGN_params': self.variable_transformer(self.dlgn_scaled, dlgn_lower, dlgn_upper),
            'V1_params': self.variable_transformer(self.v1_scaled, v1_lower, v1_upper)
        }

    def update_transform_separate(self):

        ## Extract dlgn bounds
        dlgn_bounds = [
            tf.convert_to_tensor([x for x in inner_bounds.values()])[:-2]
            for inner_bounds in self.bounds
        ]
        dlgn_lower = [
            tf.reshape(inner_bounds[:,0], [1,1,-1,1,1])
            for inner_bounds in dlgn_bounds
        ]
        dlgn_upper = [
            tf.reshape(inner_bounds[:,1], [1,1,-1,1,1])
            for inner_bounds in dlgn_bounds
        ]

        # Extract v1 bounds
        v1_bounds = [
            tf.convert_to_tensor([x for x in inner_bounds.values()])[-2:]
            for inner_bounds in self.bounds
        ]
        v1_lower = [
            tf.reshape(inner_bounds[:,0], [1,1,-1,1,1])
            for inner_bounds in v1_bounds
        ]
        v1_upper = [
            tf.reshape(inner_bounds[:,1], [1,1,-1,1,1])
            for inner_bounds in v1_bounds
        ]

        dlgn_unstacked = []
        v1_unstacked = []
        for i in range(self.n_v1):
            
            dlgn_unstacked.append(
                self.variable_transformer(
                self.dlgn_scaled[:,i,:,:,:,:], 
                dlgn_lower[i], 
                dlgn_upper[i]
                )
            )

            v1_unstacked.append(
                self.variable_transformer(
                self.v1_scaled[:,i,:,:,:,:], 
                v1_lower[i], 
                v1_upper[i]
                )
            )

        self.params = {
            'dLGN_params': tf.stack(dlgn_unstacked, axis=1),
            'V1_params': tf.stack(v1_unstacked, axis=1)
        }


class Optimize:
    def __init__(
            self, model, 
            epochs = 50, loss_threshold = 0.105,
            flatness_lambda = 1e-3, flatness_epsilon = 1e-6,
            highpass_lambda = 1e-3
    ):
        self.model = model
        self.epochs = epochs
        self.loss_threshold = loss_threshold
        self.flatness_lambda = flatness_lambda
        self.flatness_epsilon = flatness_epsilon
        self.highpass_lambda = highpass_lambda

        if hasattr(tf.keras.optimizers, "AdamW"):
            self.optimizer = tf.keras.optimizers.AdamW()
            print(f"Optimizer initialized with {self.optimizer}")
        else:
            self.optimizer = tf.keras.optimizers.Adam()
            print(f"Optimizer initialized with {self.optimizer}")
    
    def mse(self, Y_pred, Y_true):
        ## The dimension of Y_true has to be expanded to broadcast across multiple samples
        return tf.reduce_mean((Y_pred - Y_true[None, :, :, :,])**2, axis = [1,2,3])

    def train_step(self, opt, X, Y_true):
        
        with tf.GradientTape() as tape:

            ## calculate a penalty on the flatness of the SF amplitude tuning curve
            
            # get the amplitude tuning curve
            amp_tuning = self.model.f_amp(
                X,
                self.model.params['dLGN_params'][:,:,:,2,:,:],
                self.model.params['dLGN_params'][:,:,:,3,:,:],
                self.model.params['dLGN_params'][:,:,:,4,:,:],
                self.model.params['dLGN_params'][:,:,:,5,:,:],
            )

            Y_pred = self.model.predict(X)
            loss = self.mse(Y_pred, Y_true) 

        # First, find the exploration samples that have converged
        # 1 for all exploration samples that HAVE converged.
        self.converged = tf.cast(loss<=self.loss_threshold, dtype = DTYPE)

        # The gradients variable is a tuple containing the gradients for each trainable
        # variable in the model. In this specific model, there are two trainable variables:
        # 1) The matrix of dLGN parameters and 2) The matrix of V1 parameters. We want to
        # mask each variable separately:
        self.gradients = tape.gradient(loss, self.model.trainable_variables)

        self.masked_gradients = []
        for g in self.gradients: # iterate through each trainable variable

            # Expand mask to match parameter rank
            gmask = tf.reshape(
                tf.abs(self.converged-1), # 1 for all exploration samples that STILL HAVEN'T converged.
                [self.model.n_sample] # In this model, exploration samples are in the 0th dimension.
                + [1] * (len(g.shape) - 1) # Fill with ones to match the rank of the given trainable variable.
            )

            # Apply the mask to the gradients of the given trainable variable.
            # For exploration samples with a loss under the threshold, the gradients
            # will now be set to 0. 
            self.masked_gradients.append(g * gmask)
        
        opt.apply_gradients(zip(tuple(self.masked_gradients), self.model.trainable_variables))   
        return loss

    def fit(self, X, Y_true, output_dtype = np.float16):

        defined_params = [x for x in list(self.model.params.keys())]
        native_params = [x.name.split(':')[0] for x in self.model.trainable_variables]
        native_to_defined = np.array([native_params.index(x) for x in defined_params])

        elems = [(self.model.n_lgn,7), (1,2)]
        self.param_history = {
            key: np.zeros([self.model.n_sample, self.epochs, self.model.n_v1, elem[0], elem[1], 1, 1], dtype=output_dtype)
            for key, elem in zip(list(self.model.params.keys()), elems)
        }
        self.scaled_param_history = {
            key: np.zeros([self.model.n_sample, self.epochs, self.model.n_v1, elem[0], elem[1], 1, 1], dtype=output_dtype)
            for key, elem in zip(list(self.model.params.keys()), elems)
        }
        self.gradient_history = {
            key: np.zeros([self.model.n_sample, self.epochs, self.model.n_v1, elem[0], elem[1], 1, 1], dtype=output_dtype)
            for key, elem in zip(list(self.model.params.keys()), elems)
        }

        self.loss_decay = np.zeros((self.epochs, self.model.n_sample), dtype=output_dtype)

        for i in range(self.epochs):

            loss = self.train_step(self.optimizer, X, Y_true)

            minloss = loss.numpy().min()
            medloss = np.median(loss.numpy())
            maxloss = loss.numpy().max()

            if i%100 == 0:
                print(f"Training step = {i}, N_exploration_samples = {len(loss)},\nmin_loss = {minloss}\nmed_loss = {medloss}\nmax_loss = {maxloss}\n")

            self.loss_decay[i,:] = loss

            for key, value in self.model.params.items():
                idx = native_to_defined[list(self.model.params.keys()).index(key)]
                self.param_history[key][:,i,:,:,:] = tf.identity(value).numpy().astype(output_dtype)
                self.scaled_param_history[key][:,i,:,:,:] = tf.identity(self.model.trainable_variables[idx]).numpy().astype(output_dtype)
                self.gradient_history[key][:,i,:,:,:] = tf.identity(self.gradients[idx]).numpy().astype(output_dtype)
                
        self.save_state("_", write=False)
        return tf.convert_to_tensor(self.loss_decay)
    
    def save_state(self, file_identifier, dtype = np.float16, write = True):

        self.outputs = {
            'converged_samples': self.converged.numpy().astype(dtype),

            'param_history': {
                    x[0]: x[1].astype(dtype)
                    for x in self.param_history.items()
            },

            'scaled_param_history': {
                    x[0]: x[1].astype(dtype)
                    for x in self.scaled_param_history.items()
            },

            'gradient_history': {
                    x[0]: x[1].astype(dtype)
                    for x in self.gradient_history.items()
            },

            'loss_decay': self.loss_decay.astype(dtype),

            'final_epoch_params': {
                    x[0]: x[1][:,-1].astype(dtype)
                    for x in self.scaled_param_history.items()
            }
        }
        
        if write:
            with open(f"{file_identifier}.pkl", 'wb') as f:
                pickle.dump(self.outputs, f)

class _PermPassedWithNoID(Exception):
    """
    Exception raised for when the user sets identify = False
    when initializing the model but tries to pass something to
    enforce_permutation
    """

    def __init__(self):
        self.message = f"""
        User passed an argument to "enforce_permutation" while identify
        was set to False.
        """
        super().__init__(self.message)