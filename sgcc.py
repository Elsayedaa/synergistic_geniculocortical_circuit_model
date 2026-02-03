## Multistart optimization enabled

import tensorflow as tf
import numpy as np

DTYPE = tf.float32
class SGCCircuit(tf.Module):

    def __init__(
            self,
            t0, t1, inh_d0, inh_d1,
            fts0_1, fts0_2, fts0_3,
            fts1_1, fts1_2, fts1_3,
            a0, a1, ats0, ats1,
            d0, d1, 
            inh_w0, inh_w1,
            n_sample = 1,
    ):
        super().__init__(name = 'SGCC')

        ## Fixed parameters

        # reshape to broadcast across multiple exploration samples
        self.n_sample = n_sample
        self.T = tf.reshape(tf.cast(tf.linspace(0,250,250), dtype = DTYPE), [1,1,-1])
        self.mid = tf.cast(tf.fill([self.n_sample, 1, 1], 0), dtype = DTYPE)
        
        ## Trainable parameters

        #### Initial response latencies ####
        """
        The initial response latencies must be the same for each dLGN unit
        into a particular V1 unit and must be the same for the corresponding
        inhibitory inputs. 
        """
        self.t0_lower, self.t0_upper = t0
        self.t0 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_0 input Latency'
        )

        self.t1_lower, self.t1_upper = t1
        self.t1 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_1 input Latency'
        )
        #################################################################

        #### Inhibition delay ####
        self.inh_d0_lower, self.inh_d0_upper = inh_d0
        self.inh_d0 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_0 inhibition delay'
        )

        self.inh_d1_lower, self.inh_d1_upper = inh_d1
        self.inh_d1 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_1 inhibition delay'
        )
        #################################################################

        #### Frequency-time slopes ####
        self.fts0_1_lower, self.fts0_1_upper = fts0_1
        self.fts0_1 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='dLGN1 into V1_0 - frequency-time slope'
        )

        self.fts0_2_lower, self.fts0_2_upper = fts0_2
        self.fts0_2 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='dLGN2 into V1_0 - frequency-time slope'
        )

        self.fts0_3_lower, self.fts0_3_upper = fts0_3
        self.fts0_3 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='dLGN3 into V1_0 - frequency-time slope'
        )

        self.fts1_1_lower, self.fts1_1_upper = fts1_1
        self.fts1_1 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='dLGN1 into V1_1 - frequency-time slope'
        )

        self.fts1_2_lower, self.fts1_2_upper = fts1_2
        self.fts1_2 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='dLGN2 into V1_1 - frequency-time slope'
        )

        self.fts1_3_lower, self.fts1_3_upper = fts1_3
        self.fts1_3 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='dLGN3 into V1_1 - frequency-time slope'
        )
        #################################################################

        #### Initial response amplitude ####
        self.a0_lower, self.a0_upper = a0
        self.a0 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_0 input amplitude'
        )

        self.a1_lower, self.a1_upper = a1
        self.a1 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_1 input amplitude'
        )
        #################################################################

        #### Amplitude-time slopes ####
        self.ats0_lower, self.ats0_upper = ats0
        self.ats0 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_0 input amplitude-time slope'
        )

        self.ats1_lower, self.ats1_upper = ats1
        self.ats1 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_1 input amplitude-time slope'
        )

        #### Initial response duration ####
        self.d0_lower, self.d0_upper = d0
        self.d0 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_0 input duration'
        )

        self.d1_lower, self.d1_upper = d1
        self.d1 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_1 input duration'
        )

        #### Inhibition weight ####
        self.inh_w0_lower, self.inh_w0_upper = inh_w0
        self.inh_w0 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_0 inhibition weight'
        )

        self.inh_w1_lower, self.inh_w1_upper = inh_w1
        self.inh_w1 = tf.Variable(
            self.variable_initializer(), dtype=DTYPE, trainable=True,
            name='V1_1 inhibition weight'
        )

        ## Apply transformation to all variables
        self.update_transform()

    def variable_transformer(self, x, lower, upper):
        transformed = lower + (upper - lower) * tf.sigmoid(x)
        return transformed

    def variable_initializer(self):
        scaled = tf.random.normal([self.n_sample, 1, 1], mean = 0, stddev = 1)
        return scaled

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
        sfs = tf.reshape(sfs, [1,-1,1])

        # Transform the scaled variables before doing calculations
        self.update_transform()

        # V1 Neuron 1 - excitatory dLGN inputs
        self.dlgn0_1e = self.frf(sfs, self.params['fts0_1'][0], self.params['t0'][0], self.params['ats0'][0], self.params['a0'][0], self.params['d0'][0])
        self.dlgn0_2e = self.frf(sfs, self.params['fts0_2'][0], self.params['t0'][0], self.params['ats0'][0], self.params['a0'][0], self.params['d0'][0])
        self.dlgn0_3e = self.frf(sfs, self.params['fts0_3'][0], self.params['t0'][0], self.params['ats0'][0], self.params['a0'][0], self.params['d0'][0])

        # V1 Neuron 1 - inhibitory dLGN inputs
        self.dlgn0_1i = self.frf(sfs, self.params['fts0_1'][0], self.params['t0'][0] + self.params['inh_d0'][0], self.params['ats0'][0], self.params['a0'][0], self.params['d0'][0])
        self.dlgn0_2i = self.frf(sfs, self.params['fts0_2'][0], self.params['t0'][0] + self.params['inh_d0'][0], self.params['ats0'][0], self.params['a0'][0], self.params['d0'][0])
        self.dlgn0_3i = self.frf(sfs, self.params['fts0_3'][0], self.params['t0'][0] + self.params['inh_d0'][0], self.params['ats0'][0], self.params['a0'][0], self.params['d0'][0])

        # V1 Neuron 2 - excitatory dLGN inputs
        self.dlgn1_1e = self.frf(sfs, self.params['fts1_1'][0], self.params['t1'][0], self.params['ats1'][0], self.params['a1'][0], self.params['d1'][0])
        self.dlgn1_2e = self.frf(sfs, self.params['fts1_2'][0], self.params['t1'][0], self.params['ats1'][0], self.params['a1'][0], self.params['d1'][0])
        self.dlgn1_3e = self.frf(sfs, self.params['fts1_3'][0], self.params['t1'][0], self.params['ats1'][0], self.params['a1'][0], self.params['d1'][0])

        # V1 Neuron 2 - inhibitory dLGN inputs
        self.dlgn1_1i = self.frf(sfs, self.params['fts1_1'][0], self.params['t1'][0] + self.params['inh_d1'][0], self.params['ats1'][0], self.params['a1'][0], self.params['d1'][0])
        self.dlgn1_2i = self.frf(sfs, self.params['fts1_2'][0], self.params['t1'][0] + self.params['inh_d1'][0], self.params['ats1'][0], self.params['a1'][0], self.params['d1'][0])
        self.dlgn1_3i = self.frf(sfs, self.params['fts1_3'][0], self.params['t1'][0] + self.params['inh_d1'][0], self.params['ats1'][0], self.params['a1'][0], self.params['d1'][0])


        # V1 Neuron 1 - excitatory and inhibitory components
        self.v1_0e = self.dlgn0_1e + self.dlgn0_2e + self.dlgn0_3e
        self.v1_0i = self.dlgn0_1i + self.dlgn0_2i + self.dlgn0_3i

        # V1 Neuron 2 - excitatory and inhibitory components
        self.v1_1e = self.dlgn1_1e + self.dlgn1_2e + self.dlgn1_3e
        self.v1_1i = self.dlgn1_1i + self.dlgn1_2i + self.dlgn1_3i

        # V1 Neuron 1 - full responses
        self.v1_0 = self.v1_0e - (self.params['inh_w0'][0] * self.v1_0i)

        # V1 Neuron 2 - full responses
        self.v1_1 = self.v1_1e - (self.params['inh_w1'][0] * self.v1_1i)

        Y = tf.stack([
            self.v1_0,
            self.v1_1,
        ], axis = 1)

        return Y
    
    def update_transform(self):

        # Transform the normalized variables back into their meaningful scale
        self.params = {
            "t0": (self.variable_transformer(self.t0, self.t0_lower, self.t0_upper), 'V1_0 input Latency:0'),
            "t1": (self.variable_transformer(self.t1, self.t1_lower, self.t1_upper), 'V1_1 input Latency:0'),

            "inh_d0": (self.variable_transformer(self.inh_d0, self.inh_d0_lower, self.inh_d0_upper), 'V1_0 inhibition delay:0'),
            "inh_d1": (self.variable_transformer(self.inh_d1, self.inh_d1_lower, self.inh_d1_upper), 'V1_1 inhibition delay:0'),

            "fts0_1": (self.variable_transformer(self.fts0_1, self.fts0_1_lower, self.fts0_1_upper), 'dLGN1 into V1_0 - frequency-time slope:0'),
            "fts0_2": (self.variable_transformer(self.fts0_2, self.fts0_2_lower, self.fts0_2_upper), 'dLGN2 into V1_0 - frequency-time slope:0'),
            "fts0_3": (self.variable_transformer(self.fts0_3, self.fts0_3_lower, self.fts0_3_upper), 'dLGN3 into V1_0 - frequency-time slope:0'),

            "fts1_1": (self.variable_transformer(self.fts1_1, self.fts1_1_lower, self.fts1_1_upper), 'dLGN1 into V1_1 - frequency-time slope:0'),
            "fts1_2": (self.variable_transformer(self.fts1_2, self.fts1_2_lower, self.fts1_2_upper), 'dLGN2 into V1_1 - frequency-time slope:0'),
            "fts1_3": (self.variable_transformer(self.fts1_3, self.fts1_3_lower, self.fts1_3_upper), 'dLGN3 into V1_1 - frequency-time slope:0'),

            "a0": (self.variable_transformer(self.a0, self.a0_lower, self.a0_upper), 'V1_0 input amplitude:0'),
            "a1": (self.variable_transformer(self.a1, self.a1_lower, self.a1_upper), 'V1_1 input amplitude:0'),

            "ats0": (self.variable_transformer(self.ats0, self.ats0_lower, self.ats0_upper), 'V1_0 input amplitude-time slope:0'),
            "ats1": (self.variable_transformer(self.ats1, self.ats1_lower, self.ats1_upper), 'V1_1 input amplitude-time slope:0'),

            "d0": (self.variable_transformer(self.d0, self.d0_lower, self.d0_upper), 'V1_0 input duration:0'),
            "d1": (self.variable_transformer(self.d1, self.d1_lower, self.d1_upper), 'V1_1 input duration:0'),

            "inh_w0": (self.variable_transformer(self.inh_w0, self.inh_w0_lower, self.inh_w0_upper), 'V1_0 inhibition weight:0'),
            "inh_w1": (self.variable_transformer(self.inh_w1, self.inh_w1_lower, self.inh_w1_upper), 'V1_1 inhibition weight:0'),
        }


class Optimize:
    def __init__(self, model, epochs = 50):
        self.model = model
        self.epochs = epochs
    
    def mse(self, Y_pred, Y_true):
        ## The dimension of Y_true has to be expanded to broadcast across multiple samples
        return tf.reduce_mean((Y_pred - Y_true[None, :, :, :])**2, axis = [1,2,3])

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
