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
            mode = 'loaded',
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
        self.t0 = tf.Variable(self.variable_initializer(t0, mode), dtype = DTYPE, trainable =  True,
                              name = 'V1_0 input Latency') # latency of the initial response in V1 #1 inputs
        
        self.t1 = tf.Variable(self.variable_initializer(t1, mode), dtype = DTYPE, trainable = True,
                              name = 'V1_1 input Latency') # latency of the initial response in V1 #2 inputs
        #################################################################

        #### Inhibition delay ####
        """
        The inhibition delays must be the same for each dLGN unit
        into a particular V1 unit and must be the same for the corresponding
        inhibitory inputs. 
        """
        self.inh_d0 = tf.Variable(self.variable_initializer(inh_d0, mode), dtype = DTYPE, trainable =  True,
                                  name = 'V1_0 inhibition delay') # inhibition delay of the initial response in V1 #1 inputs
        self.inh_d1 = tf.Variable(self.variable_initializer(inh_d1, mode), dtype = DTYPE, trainable =  True, 
                                  name = 'V1_1 inhibition delay') # inhibition delay of the initial response in V1 #2 inputs
        #################################################################

        #### Frequency-time slopes ####
        """
        The frequency-time slopes can vary for each dLGN unit into a particular
        V1 unit but inhibitory slopes must be the same as the excitatory slopes.
        Relu applied to keep positive.
        """
        self.fts0_1 = tf.Variable(self.variable_initializer(fts0_1, mode),
                                  dtype = DTYPE, trainable = True, constraint = tf.nn.relu,
                                  name = 'dLGN1 into V1_0 - frequency-time slope') # frequency-time slope of dLGN#1 into V1#1
        self.fts0_2 = tf.Variable(self.variable_initializer(fts0_2, mode), 
                                  dtype = DTYPE, trainable = True, constraint = tf.nn.relu,
                                  name = 'dLGN2 into V1_0 - frequency-time slope') # frequency-time slope of dLGN#2 into V1#1
        self.fts0_3 = tf.Variable(self.variable_initializer(fts0_3, mode), 
                                  dtype = DTYPE, trainable =  True, constraint = tf.nn.relu,
                                  name = 'dLGN3 into V1_0 - frequency-time slope') # frequency-time slope of dLGN#3 into V1#1

        self.fts1_1 = tf.Variable(self.variable_initializer(fts1_1, mode), 
                                  dtype = DTYPE, trainable = True, constraint = tf.nn.relu,
                                  name = 'dLGN1 into V1_1 - frequency-time slope') # frequency-time slope of dLGN#4 into V1#2
        self.fts1_2 = tf.Variable(self.variable_initializer(fts1_2, mode), 
                                  dtype = DTYPE, trainable = True, constraint = tf.nn.relu,
                                  name = 'dLGN2 into V1_1 - frequency-time slope') # frequency-time slope of dLGN#5 into V1#2
        self.fts1_3 = tf.Variable(self.variable_initializer(fts1_3, mode), 
                                  dtype = DTYPE, trainable = True, constraint = tf.nn.relu,
                                  name = 'dLGN3 into V1_1 - frequency-time slope') # frequency-time slope of dLGN#6 into V1#2
        #################################################################

        #### Initial response amplitude ####
        """
        The initial response amplitudes must be the same for each dLGN unit into a particular
        V1 unit and must be the same for the corresponding inhibitory inputs. Relu applied to
        keep positive.
        """
        self.a0 = tf.Variable(self.variable_initializer(a0, mode), 
                              dtype = DTYPE, trainable = True, constraint = tf.nn.relu,
                              name = 'V1_0 input amplitude') # initial amplitude of dLGN inputs into V1 #1
        self.a1 = tf.Variable(self.variable_initializer(a1, mode), 
                              dtype = DTYPE, trainable = True, constraint = tf.nn.relu,
                              name = 'V1_1 input amplitude') # initial amplitude of dLGN inputs into V1 #2
        #################################################################

        #### Amplitude-time slopes ####
        """
        The amplitude-time slopes must be the same for each dLGN unit into a particular
        V1 unit and must be the same for the corresponding inhibitory inputs.
        """
        self.ats0 = tf.Variable(self.variable_initializer(ats0, mode), 
                                dtype = DTYPE, trainable = True, constraint = lambda x: tf.clip_by_value(x, -4, 0),
                                name = 'V1_0 input amplitude-time slope') # amplitude-time slope of dLGN inputs into V1 #1
        self.ats1 = tf.Variable(self.variable_initializer(ats1, mode), 
                                dtype = DTYPE, trainable = True, constraint = lambda x: tf.clip_by_value(x, -4, 0),
                                name = 'V1_1 input amplitude-time slope') # amplitude-time slope of dLGN inputs into V1 #2

        #### Initial response duration ####
        """
        The initial response durations must be the same for each dLGN unit into a particular
        V1 unit and must be the same for the corresponding inhibitory inputs.
        """
        self.d0 = tf.Variable(self.variable_initializer(d0, mode), 
                              dtype = DTYPE, trainable = True, constraint = tf.nn.relu,
                              name = 'V1_0 input duration') # initial duration of dLGN inputs into V1 #1
        self.d1 = tf.Variable(self.variable_initializer(d1, mode), 
                              dtype = DTYPE, trainable = True, constraint = tf.nn.relu,
                              name = 'V1_1 input duration') # initial duration of dLGN inputs into V1 #2
        
        #### Inhibition weight ####
        """
        The inhibition weight is a single variable applied to the inhibitory component of each V1 unit.
        """
        self.inh_w0 = tf.Variable(self.variable_initializer(inh_w0, mode), 
                              dtype = DTYPE, trainable = True, constraint = tf.nn.relu,
                              name = 'V1_0 inhibition weight') # inhibition weight applied to V1 #1
        self.inh_w1 = tf.Variable(self.variable_initializer(inh_w1, mode), 
                              dtype = DTYPE, trainable = True, constraint = tf.nn.relu,
                              name = 'V1_1 inhibition weight') # inhibition weight applied to V1 #2
        self.update_variables()

    def variable_initializer(self, x, mode):
        if mode == 'random':
            ## Broadcast 
            return tf.Variable(tf.random.uniform([self.n_sample, 1], minval = x[0], maxval = x[1]))[:, None, :]
        
        if mode == 'loaded':
            return tf.cast(tf.fill([1,1,1], x), dtype = DTYPE)

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
            tracking = False
    ):
        # reshape the SF input to broadcast across multiple exploration samples
        sfs = tf.reshape(sfs, [1,-1,1])

        # V1 Neuron 1 - excitatory dLGN inputs
        self.dlgn0_1e = self.frf(sfs, self.fts0_1, self.t0, self.ats0, self.a0, self.d0)
        self.dlgn0_2e = self.frf(sfs, self.fts0_2, self.t0, self.ats0, self.a0, self.d0)
        self.dlgn0_3e = self.frf(sfs, self.fts0_3, self.t0, self.ats0, self.a0, self.d0)

        # V1 Neuron 1 - inhibitory dLGN inputs
        self.dlgn0_1i = self.frf(sfs, self.fts0_1, self.t0 + self.inh_d0, self.ats0, self.a0, self.d0)
        self.dlgn0_2i = self.frf(sfs, self.fts0_2, self.t0 + self.inh_d0, self.ats0, self.a0, self.d0)
        self.dlgn0_3i = self.frf(sfs, self.fts0_3, self.t0 + self.inh_d0, self.ats0, self.a0, self.d0)

        # V1 Neuron 2 - excitatory dLGN inputs
        self.dlgn1_1e = self.frf(sfs, self.fts1_1, self.t1, self.ats1, self.a1, self.d1)
        self.dlgn1_2e = self.frf(sfs, self.fts1_2, self.t1, self.ats1, self.a1, self.d1)
        self.dlgn1_3e = self.frf(sfs, self.fts1_3, self.t1, self.ats1, self.a1, self.d1)

        # V1 Neuron 2 - inhibitory dLGN inputs
        self.dlgn1_1i = self.frf(sfs, self.fts1_1, self.t1 + self.inh_d1, self.ats1, self.a1, self.d1)
        self.dlgn1_2i = self.frf(sfs, self.fts1_2, self.t1 + self.inh_d1, self.ats1, self.a1, self.d1)
        self.dlgn1_3i = self.frf(sfs, self.fts1_3, self.t1 + self.inh_d1, self.ats1, self.a1, self.d1)

        # V1 Neuron 1 - excitatory and inhibitory components
        self.v1_0e = self.dlgn0_1e + self.dlgn0_2e + self.dlgn0_3e
        self.v1_0i = self.dlgn0_1i + self.dlgn0_2i + self.dlgn0_3i

        # V1 Neuron 2 - excitatory and inhibitory components
        self.v1_1e = self.dlgn1_1e + self.dlgn1_2e + self.dlgn1_3e
        self.v1_1i = self.dlgn1_1i + self.dlgn1_2i + self.dlgn1_3i

        # V1 Neuron 1 - full responses
        self.v1_0 = self.v1_0e - (self.inh_w0 * self.v1_0i)

        # V1 Neuron 2 - full responses
        self.v1_1 = self.v1_1e - (self.inh_w1 * self.v1_1i)

        Y = tf.stack([
            self.v1_0,
            self.v1_1,
        ], axis = 1)

        if tracking:
            self.update_variables()

        return Y
    
    def update_variables(self):
        # For internal tracking
        self.params = {
            "t0": self.t0,
            "t1": self.t1,

            "inh_d0": self.inh_d0,
            "inh_d1": self.inh_d1,

            "fts0_1": self.fts0_1,
            "fts0_2": self.fts0_2,
            "fts0_3": self.fts0_3,

            "fts1_1": self.fts1_1,
            "fts1_2": self.fts1_2,
            "fts1_3": self.fts1_3,

            "a0": self.a0,
            "a1": self.a1,

            "ats0": self.ats0,
            "ats1": self.ats1,

            "d0": self.d0,
            "d1": self.d1,

            "inh_w0": self.inh_w0,
            "inh_w1": self.inh_w1,
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

            Y_pred = model.predict(X, tracking = True)
            loss = self.mse(Y_pred, Y_true)

        self.gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(self.gradients, model.trainable_variables))

        return loss

    def fit(self, X, Y_true):
        self.loss_decay = []
        self.param_history = []
        self.gradient_history = []

        optimizer = tf.keras.optimizers.Adam(1e-3)

        defined_params = [x.name for x in list(self.model.params.values())]
        native_params = [x.name for x in self.model.trainable_variables]
        native_to_defined = np.array([native_params.index(x) for x in defined_params])

        self.param_history = {
            key: np.zeros([self.model.n_sample, self.epochs])
            for key in list(self.model.params.keys())
        }

        self.gradient_history_history = {
            key: np.zeros([self.model.n_sample, self.epochs])
            for key in list(self.model.params.keys())
        }

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
                self.param_history[key][:,i] = tf.identity(value)[:,-1,-1]
                self.gradient_history_history[key][:,i] = tf.identity(self.gradients)[idx][:,-1,-1]

            # self.param_history.append({key:tf.identity(value) for key, value in self.model.params.items()})
            # self.gradient_history.append(np.array(self.gradients)[native_to_defined])

        return tf.convert_to_tensor(self.loss_decay)
