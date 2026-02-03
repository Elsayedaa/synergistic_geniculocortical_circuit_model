## Multistart optimization enabled

import tensorflow as tf
import numpy as np
import pickle

DTYPE = tf.float32
class SGCCircuit(tf.Module):

    def __init__(
            self,
            bounds,
    ):
        super().__init__(name = 'SGCC')

        ## Set the bounds on the trainable parameters
        self.bounds = bounds

    def variable_transformer(self, x, lower, upper):
        transformed = lower + (upper - lower) * tf.sigmoid(x)
        return transformed

    def initialize_random_parameters(self, n_v1, n_lgn, n_sample):
        self.n_sample = n_sample

        ## Fixed parameters
        self.T = tf.reshape(tf.cast(tf.linspace(0,250,250), dtype = DTYPE), [1, 1, 1, 1,-1])
        self.mid = tf.cast(tf.fill([n_v1, n_lgn, self.n_sample, 1, 1], 0), dtype = DTYPE)

        ## Trainable parameters
        dlgn_scaled = tf.random.normal([n_v1, n_lgn, 5, self.n_sample, 1, 1], mean = 0, stddev = 1)
        v1_scaled = tf.random.normal([n_v1, n_lgn, 2, n_sample, 1, 1])

        self.dlgn_scaled = tf.Variable(dlgn_scaled, name = 'dLGN_parameters', dtype = DTYPE)
        self.v1_scaled = tf.Variable(v1_scaled, name = 'V1_parameters', dtype = DTYPE)

    def load_saved_parameters(self, parameters):
        self.n_sample = len(parameters['t0'])

        ## Fixed parameters
        self.T = tf.reshape(tf.cast(tf.linspace(0,250,250), dtype = DTYPE), [1,1,-1])
        self.mid = tf.cast(tf.fill([self.n_sample, 1, 1], 0), dtype = DTYPE)

        self.loaded_parameters = parameters
        self.set_parameters(initialization='saved')

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

    def f_amp(self, ats, a, sf):
        return ats*sf+a

    def f_t(self, fts, t, sf):
        return fts*sf+t

    def frf(
        self,
        sfs, 
        fts, t,
        ats, a,
        d
    ):
        # Make the rest of the variables broadcastable
        f = self.gaussian(
            self.T,
            self.f_t(fts, t, sfs),
            self.f_amp(ats, a, sfs),
            self.mid, d
        )

        return f

    def predict(
            self,
            sfs,
    ):
        # reshape the SF input to broadcast across multiple exploration samples
        sfs = tf.reshape(sfs, [1,1,1,-1,1])

        # Transform the scaled variables before doing calculations
        self.update_transform()

        self.dlgn_exc = self.frf(
            sfs,
            self.params['dLGN_params'][:,:,0,:,:,:],
            self.params['dLGN_params'][:,:,1,:,:,:],
            self.params['dLGN_params'][:,:,2,:,:,:],
            self.params['dLGN_params'][:,:,3,:,:,:],
            self.params['dLGN_params'][:,:,4,:,:,:]
        )

        self.dlgn_inh = self.params['V1_params'][:,:,1,:,:,:] * self.frf(
            sfs,
            self.params['dLGN_params'][:,:,0,:,:,:],
            self.params['dLGN_params'][:,:,1,:,:,:] + self.params['V1_params'][:,:,0,:,:,:],
            self.params['dLGN_params'][:,:,2,:,:,:],
            self.params['dLGN_params'][:,:,3,:,:,:],
            self.params['dLGN_params'][:,:,4,:,:,:]
        )

        Y = tf.reduce_sum(self.dlgn_exc, axis = 1) - tf.reduce_sum(self.dlgn_inh, axis = 1)

        return Y
    
    def update_transform(self):

        # Transform the normalized variables back into their meaningful scale
        dlgn_bounds = tf.convert_to_tensor([x for x in self.bounds.values()])[:5]
        dlgn_lower = tf.reshape(dlgn_bounds[:,0], [1,1,-1,1,1,1])
        dlgn_upper = tf.reshape(dlgn_bounds[:,1], [1,1,-1,1,1,1])

        v1_bounds = tf.convert_to_tensor([x for x in self.bounds.values()])[-2:]
        v1_lower = tf.reshape(v1_bounds[:,0], [1,1,-1,1,1,1])
        v1_upper = tf.reshape(v1_bounds[:,1], [1,1,-1,1,1,1])


        self.params = {
            'dLGN_params': self.variable_transformer(self.dlgn_scaled, dlgn_lower, dlgn_upper),
            'V1_params': self.variable_transformer(self.v1_scaled, v1_lower, v1_upper)
        }

class Optimize:
    def __init__(self, model, epochs = 50):
        self.model = model
        self.epochs = epochs
    
    def mse(self, Y_pred, Y_true):
        ## The dimension of Y_true has to be expanded to broadcast across multiple samples
        return tf.reduce_mean((Y_pred - Y_true[None, :, :, :,])**2, axis = [1,2,3])

    def train_step(self, model, opt, X, Y_true):

        with tf.GradientTape() as tape:

            Y_pred = model.predict(X)
            loss = self.mse(Y_pred, Y_true)

        self.gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(self.gradients, model.trainable_variables))

        return loss

    def fit(self, X, Y_true):

        optimizer = tf.keras.optimizers.Adam(1e-3)

        defined_params = [x[1] for x in list(self.model.params.values())]
        native_params = [x.name for x in self.model.trainable_variables]
        native_to_defined = np.array([native_params.index(x) for x in defined_params])

        self.param_history = {
            key: np.zeros([self.model.n_sample, self.epochs])
            for key in list(self.model.params.keys())
        }

        self.scaled_param_history = {
            key: np.zeros([self.model.n_sample, self.epochs])
            for key in list(self.model.params.keys())
        }

        self.gradient_history = {
            key: np.zeros([self.model.n_sample, self.epochs])
            for key in list(self.model.params.keys())
        }

        self.loss_decay = []

        for i in range(self.epochs):

            loss = self.train_step(self.model, optimizer, X, Y_true)

            minloss = loss.numpy().min()
            medloss = np.median(loss.numpy())
            maxloss = loss.numpy().max()

            if i%100 == 0:
                print(f"Training step = {i}, N_exploration_samples = {len(loss)},\nmin_loss = {minloss}\nmed_loss = {medloss}\nmax_loss = {maxloss}\n")

            self.loss_decay.append(loss)

            for key, value in self.model.params.items():
                idx = native_to_defined[list(self.model.params.keys()).index(key)]
                self.param_history[key][:,i] = tf.identity(value[0])[:,-1,-1]
                self.scaled_param_history[key][:,i] = tf.identity(self.model.trainable_variables)[idx][:,-1,-1]
                self.gradient_history[key][:,i] = tf.identity(self.gradients)[idx][:,-1,-1]

        return tf.convert_to_tensor(self.loss_decay)
    
    def save_state(self, file_identifier):

        self.outputs = {
            'param_history': self.param_history,
            'scaled_param_history': self.scaled_param_history,
            'gradient_history': self.gradient_history,
            'loss_decay': self.loss_decay,
            'final_epoch_params': {
                x[0]:tf.convert_to_tensor(x[1][:,-1][:,None, None], dtype=DTYPE) 
                for x in self.scaled_param_history.items()
            }
        }
        
        with open(f"{file_identifier}.pkl", 'wb') as f:
            pickle.dump(self.outputs, f)
