"""
    Advantage Actor-Critic (A2C) Algorithm

"""

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

class A2C(object):
    def __init__(self, s_dim, a_dim, sess, policy_logvar=-1.0,
                 lr_a=0.0001, lr_c=0.001, gamma=0.99,
                 batch_size=256,
                 c_epochs=10, a_epochs=10):
        self.sess = sess

        self.c_epochs, self.a_epochs = c_epochs, a_epochs

        self.s_dim, self.a_dim = s_dim, a_dim
        self.lr_a, self.lr_c = lr_a, lr_c
        self.gamma = gamma
        self.policy_logvar = policy_logvar
        self.batch_size = batch_size

        self.train_cnt = 0

        self._placeholders()
        self.v = self._build_value_net(self.s_ph, scope='value_function', trainable=True)

        # actor
        self.means, self.log_vars = self._build_policy_net(self.s_ph, 'policy', trainable=True)
        self.logp = self._logprob()

        self.sampled_act = self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=[1, self.a_dim])

        self.c_loss = tf.reduce_mean(tf.square(self.v - self.val_ph))
        self.c_train_op = tf.train.AdamOptimizer(self.lr_c).minimize(self.c_loss)

        exp_v = self.logp * tf.stop_gradient(self.adv_ph)
        self.exp_v = exp_v
        self.a_loss = - tf.reduce_mean(self.exp_v)
        self.a_train_op = tf.train.AdamOptimizer(self.lr_a).minimize(self.a_loss)

        self.sess.run(tf.global_variables_initializer())
        self._print_hyperparams()

        print('-- INFO: A2C initialized.')
        print('==========================')

    def _print_hyperparams(self):
        print('------------------- Hyperparameters ----------------------')
        print('-- S_Dim:', self.s_dim)
        print('-- A_Dim:', self.a_dim)
        print('-- LR_Critic:', self.lr_c)
        print('-- LR_Actor:', self.lr_a)
        print('-- Gamma:', self.gamma)
        print('--')

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.s_ph = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        self.a_ph = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.adv_ph = tf.placeholder(tf.float32, [None, 1], 'advantages')
        self.val_ph = tf.placeholder(tf.float32, [None, 1], 'val_valfunc')

    def _build_value_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            fc1 = tf.layers.dense(s, 200, activation=tf.nn.relu, name='fc1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.s_dim)))
            fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='fc2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            v = tf.layers.dense(fc2, 1, activation=None, name='v_value', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 100)))
        return v

    def _build_policy_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            fc1 = tf.layers.dense(s, 200, activation=tf.nn.relu, name='fc1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.s_dim)))
            fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='fc2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            means = tf.layers.dense(fc2, self.a_dim, activation=tf.nn.tanh, name='means', trainable=trainable,
                                    kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 100)))

            logvar_speed = (10 * 64) // 48
            spd_log_vars = tf.get_variable('spd_logvars', [logvar_speed, self.a_dim], tf.float32,
                                           tf.constant_initializer(0.0))
            log_vars = tf.reduce_sum(spd_log_vars, axis=0, keepdims=True) + self.policy_logvar

        return means, log_vars

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.a_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1, keepdims=True)

        return logp

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sampled_act, feed_dict={self.s_ph: s})[0]
        return np.clip(a, -1, 1)

    def predict_v(self, s):
        """ Predict method """
        y_hat = self.sess.run(self.v, feed_dict={self.s_ph: s})

        return np.squeeze(y_hat)

    def update_p(self, observes, actions, advantages):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
        """
        feed_dict = {self.s_ph: observes,
                     self.a_ph: actions,
                     self.adv_ph: advantages.reshape(-1, 1),
                     }

        self.sess.run(self.a_train_op, feed_dict)
        a_loss = self.sess.run(self.a_loss, feed_dict)

        return a_loss

    def update_v(self, x, y):
        """ Fit model to current data batch + previous data batch

        Args:
            x: features
            y: target
            logger: logger to save training loss and % explained variance
        """
        num_batches = max(x.shape[0] // self.batch_size, 1)
        batch_size = x.shape[0] // num_batches

        x_train, y_train = x, y

        for e in range(self.c_epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.s_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end].reshape(-1, 1)}
                _, l = self.sess.run([self.c_train_op, self.c_loss], feed_dict=feed_dict)

        y_hat = self.predict_v(x)
        c_loss = np.mean(np.square(y_hat - y))  # explained variance after update

        return c_loss
