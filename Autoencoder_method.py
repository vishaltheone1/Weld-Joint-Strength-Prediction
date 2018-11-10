
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn import LSTMCell

class Autoencoder_Time():
    def __init__(self, config):
        #Hyperparameters
        num_layers = config['num_layers']
        hidden_size = config['hidden_size']
        max_grad_norm = config['max_grad_norm']
        batch_size = config['batch_size']
        sl = config['sl']
        crd = config['crd']
        num_l = config['num_l']
        learning_rate = config['learning_rate']
        self.sl = sl
        self.batch_size = batch_size

        #Nodes for input variable
        self.x = tf.placeholder('float', shape = [batch_size, sl], name = 'Input_data')
        self.x_exp = tf.expand_dims(self.x, 1)
        self.keep_prob = tf.placeholder('float')

        #Encoder cell, multi-layered with dropout
        with tf.variable_scope('Encoder') as scope:
            cell_enc = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_size) for _ in range(num_layers)])
            cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc, output_keep_prob = self.keep_prob)
            
            #Initial state
            initial_state_enc = cell_enc.zero_state(batch_size, tf.float32)
            
            #Layer for mean of z
            W_mu = tf.get_variable('W_mu', [hidden_size, num_l])
            outputs_enc, _ = tf.contrib.rnn.static_rnn(cell_enc, inputs = tf.unstack(self.x_exp, axis = 2), initial_state = initial_state_enc) 
            cell_output = outputs_enc[-1]
            b_mu = tf.get_variable('b_mu', [num_l])
            self.z_mu = tf.nn.xw_plus_b(cell_output, W_mu, b_mu, name = 'z_mu')
            
            #Train the point in latent space to have zero mean and unit variance on batch basis
            lat_mean, lat_var = tf.nn.moments(self.z_mu, axes = [1])
            self.loss_lat_batch = tf.reduce_mean(tf.square(lat_mean) + lat_var - tf.log(lat_var) - 1)
            

        #Layers to generate initial state
        with tf.name_scope("Lat_2_dec") as scope:
            W_state = tf.get_variable('W_state', [num_l, hidden_size])
            b_state = tf.get_variable('b_state', [hidden_size])
            z_state = tf.nn.xw_plus_b(self.z_mu, W_state, b_state, name = 'z_state')
            

        #Decoder cell, multi-layered
        with tf.variable_scope("Decoder") as scope:
            cell_dec = tf.contrib.rnn.MultiRNNCell([LSTMCell(hidden_size) for _ in range(num_layers)])
            
            #Initial state
            initial_state_dec = tuple([(z_state, z_state)] * num_layers)
            dec_inputs = [tf.zeros([batch_size, 1])] * sl
            outputs_dec, _ = tf.contrib.rnn.static_rnn(cell_dec, inputs = dec_inputs, initial_state = initial_state_dec)
            
            
        with tf.name_scope("Out_layer") as scope:
            params_o = 2*crd
            W_o = tf.get_variable('W_o', [hidden_size, params_o])
            b_o = tf.get_variable('b_o', [params_o])
            outputs = tf.concat(outputs_dec, axis = 0)
            h_out = tf.nn.xw_plus_b(outputs, W_o, b_o)
            h_mu, h_sigma_log = tf.unstack(tf.reshape(h_out, [sl, batch_size, params_o]), axis = 2)
            h_sigma = tf.exp(h_sigma_log)
            dist = tf.contrib.distributions.Normal(h_mu, h_sigma)
            px = dist.log_prob(tf.transpose(self.x))
            loss_seq = -px
            self.loss_seq = tf.reduce_mean(loss_seq)
            
            
        with tf.name_scope("train") as scope:
            #Use learning rate deacy
            global_step = tf.Variable(0, trainable = False)
            lr = tf.train.exponential_decay(learning_rate, global_step, 1000, 0.1, staircase = False)
            
            self.loss = self.loss_seq + self.loss_lat_batch
            
            
            tvars = tf.trainable_variables()
            grads = tf.gradients(self.loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)    #Gradient clipping to prevent explosion
            self.numel = tf.constant([[0]])
            
            #Apply the gradients
            optimizer = tf.train.AdamOptimizer(lr)
            gradients = zip(grads, tvars)
            self.train_step = optimizer.apply_gradients(gradients, global_step = global_step)
            self.numel = tf.constant([[0]])
            
            
        tf.summary.tensor_summary('lat_state', self.z_mu)
        self.merged = tf.summary.merge_all()
        self.init_op = tf.global_variables_initializer()

