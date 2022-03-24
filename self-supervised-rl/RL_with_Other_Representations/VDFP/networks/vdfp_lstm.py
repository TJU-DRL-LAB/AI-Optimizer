"""
    Value Decomposed DDPG with Future Prediction Algorithm
    with LSTM for the trajectory representation model
    (VDFP_LSTM)
"""

import tensorflow as tf
import numpy as np
import time
import math


class VDFP(object):
    def __init__(self, s_dim, a_dim, sess,
                 lr_a=0.00025, lr_vae=0.001, lr_r=0.0005,
                 gamma=0.99, kl_coef=1000.0,
                 a_batch_size=64, m_batch_size=64,
                 memory_size=100000,
                 m_dim=100, z_dim=20,
                 clip_value=0.2,
                 sequence_length=64, min_sequence_length=16):

        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length

        self.s_dim, self.a_dim = s_dim, a_dim
        self.m_dim, self.z_dim = m_dim, z_dim

        self.lr_a, self.lr_r, self.lr_vae = lr_a, lr_r, lr_vae
        self.gamma = gamma
        self.clip_value = clip_value
        self.a_batch_size, self.m_batch_size = a_batch_size, m_batch_size

        self.memory_size = memory_size
        self.memory = np.zeros((memory_size, s_dim + a_dim + (s_dim + a_dim) * sequence_length + 2), dtype=np.float32)
        self.pointer = 0
        self.sess = sess

        self.train_cnt = 0
        self.kl_coef = kl_coef

        # -------------------------- placeholders ------------------------------
        self.s = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.ss = tf.placeholder(tf.float32, [None, sequence_length, s_dim + a_dim], 's_sequence')
        self.batch_sequence_length = tf.placeholder(tf.int16, [None, ], 'batch_sequence_length')

        self.u = tf.placeholder(tf.float32, [None, 1], 'utility')
        self.noise = tf.placeholder(tf.float32, [None, self.z_dim], 'noise')

        self.a = self._build_a(self.s, trainable=True)
        self.m = self._build_m(self.ss, self.batch_sequence_length)

        self.mean, log_std = self._build_encoder(self.s, self.a, self.m)
        self.std = tf.exp(log_std)
        self.z = self.mean + self.std * self.noise

        self.d = self._build_decoder(self.s, self.a, self.z)

        self.r = self._build_r(self.m)

        self.mm_gen = self._build_decoder(self.s, self.a, self.noise, reuse=True)
        self.objective = self._build_r(self.mm_gen, reuse=True)

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        m_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Measurement')
        p_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')
        e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
        r_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Return')
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')

        self.a_loss = - tf.reduce_mean(self.objective)
        self.a_train = tf.train.AdamOptimizer(learning_rate=self.lr_a).minimize(self.a_loss, var_list=a_params)

        self.recon_loss = tf.reduce_mean(tf.square(self.m - self.d))
        self.kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + tf.log(tf.square(self.std))
                                                           - tf.square(self.mean) - tf.square(self.std), axis=1))
        self.vae_loss = self.recon_loss + self.kl_coef * self.kl_loss
        self.vae_train = tf.train.AdamOptimizer(learning_rate=self.lr_vae).minimize(self.vae_loss, var_list=e_params + d_params)

        self.r_loss = tf.reduce_mean(tf.square(self.u - self.r))
        # self.r_train = tf.train.AdamOptimizer(learning_rate=self.lr_r).minimize(self.r_loss, var_list=m_params + r_params)
        self.r_train = tf.train.RMSPropOptimizer(learning_rate=self.lr_r).minimize(self.r_loss, var_list=m_params + r_params)

        self.sess.run(tf.global_variables_initializer())
        self._print_hyperparams()

        print('-- INFO: VDFP_LSTM initialized.')
        print('==========================')

    def _print_hyperparams(self):
        print('------------------- Hyperparameters ----------------------')
        print('-- Sequence_Length:', self.sequence_length, self.min_sequence_length)
        print('-- z_dim:', self.z_dim)
        print('-- S_Dim:', self.s_dim)
        print('-- A_Dim:', self.a_dim)
        print('-- LR_VAE:', self.lr_vae)
        print('-- LR_Return:', self.lr_r)
        print('-- LR_Actor:', self.lr_a)
        print('-- Gamma:', self.gamma)
        print('-- KL_Coef:', self.kl_coef)
        print('-- Batch_Size:', self.a_batch_size, self.m_batch_size)
        print('-- Memory_Size:', self.memory_size)
        print('--')

    def _build_a(self, s, trainable):
        with tf.variable_scope('Actor'):
            fc1 = tf.layers.dense(s, 200, activation=tf.nn.relu, name='l1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.s_dim)))
            fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='l2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            a = tf.layers.dense(fc2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 100)))
            return a

    def _build_encoder(self, s, a, m, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Encoder', reuse=reuse):
            concat = tf.concat([s, a], axis=1, name='s_a_concat')
            efsa = tf.layers.dense(concat, 400, activation=tf.nn.sigmoid, name='efsa', trainable=trainable,
                                   kernel_initializer=tf.random_normal_initializer(
                                       stddev=np.sqrt(1 / (self.s_dim + self.a_dim))))
            efm = tf.layers.dense(m, 400, activation=tf.nn.relu, name='efm', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.m_dim)))
            ef1 = efsa * efm
            ef2 = tf.layers.dense(ef1, 200, activation=tf.nn.relu, name='ef2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 400)))
            mean = tf.layers.dense(ef2, self.z_dim, activation=None, name='mean', trainable=trainable,
                                   kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            log_std = tf.layers.dense(ef2, self.z_dim, activation=None, name='std', trainable=trainable,
                                      kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))

            log_std_clipped = tf.clip_by_value(log_std, -4, 15)
            return mean, log_std_clipped

    def _build_decoder(self, s, a, z, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Decoder', reuse=reuse):
            concat = tf.concat([s, a], axis=1, name='s_a_concat')
            efsa = tf.layers.dense(concat, 200, activation=tf.nn.sigmoid, name='efsa', trainable=trainable,
                                   kernel_initializer=tf.random_normal_initializer(
                                       stddev=np.sqrt(1 / (self.s_dim + self.a_dim))))
            dfz = tf.layers.dense(z, 200, activation=tf.nn.relu, name='df1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.z_dim)))
            df1 = efsa * dfz
            df2 = tf.layers.dense(df1, 400, activation=tf.nn.relu, name='df2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            d = tf.layers.dense(df2, self.m_dim, activation=None, name='decoder', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 400)))
            return d

    def _build_m(self, ss, batch_sequence_length, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Measurement', reuse=reuse):
            lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=100)
            init_state = lstm_cell.zero_state(self.m_batch_size, dtype=tf.float32)

            outputs, last_state = tf.nn.dynamic_rnn(cell=lstm_cell, dtype=tf.float32, initial_state=init_state,
                                                    sequence_length=batch_sequence_length, inputs=ss)

            highway = tf.layers.dense(last_state[1], 100, activation=None, name='highway', trainable=trainable,
                                      kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 100)))
            joint = tf.nn.sigmoid(highway) * tf.nn.relu(highway) + (1. - tf.nn.sigmoid(highway)) * last_state[1]

            m = tf.layers.dense(joint, self.m_dim, activation=None, name='m', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 100)))

            return m

    def _build_r(self, m, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Return', reuse=reuse):
            r = tf.layers.dense(m, 1, activation=None, name='return', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.m_dim)))

            return r

    def choose_action(self, s):
        return self.sess.run(self.a, {self.s: s[np.newaxis, :]})[0]

    def train_reward_model(self):
        indices = np.random.choice(min(self.memory_size, self.pointer), size=self.m_batch_size)
        batch_exps = self.memory[indices, :]

        b_s = batch_exps[:, :self.s_dim]
        b_u = batch_exps[:, self.s_dim + self.a_dim]
        b_saoff = batch_exps[:, -(self.s_dim + self.a_dim) * self.sequence_length - 1:-1]
        b_seq_len = batch_exps[:, -1].astype(np.int16)
        b_ss = np.reshape(b_saoff, [-1, self.sequence_length, self.s_dim + self.a_dim])

        r_loss, _ = self.sess.run([self.r_loss, self.r_train],
                                  feed_dict={self.s: b_s, self.ss: b_ss,
                                             self.u: b_u.reshape((-1, 1)),
                                             # self.drop_rate: 0.2,
                                             self.batch_sequence_length: b_seq_len,
                                             })

        return r_loss

    def train_predictor(self):
        indices = np.random.choice(min(self.memory_size, self.pointer), size=self.m_batch_size)
        batch_exps = self.memory[indices, :]

        b_s = batch_exps[:, :self.s_dim]
        b_a = batch_exps[:, self.s_dim: self.s_dim + self.a_dim]
        b_saoff = batch_exps[:, -(self.s_dim + self.a_dim) * self.sequence_length - 1: -1]
        b_seq_len = batch_exps[:, -1].astype(np.int16)
        # b_saoff = batch_exps[:, -(self.s_dim + self.a_dim) * self.sequence_length:]
        b_ss = np.reshape(b_saoff, [-1, self.sequence_length, self.s_dim + self.a_dim])
        b_noise = np.clip(np.random.normal(0, 1, size=[self.m_batch_size, self.z_dim]), -2, 2)

        self.sess.run(self.vae_train, feed_dict={self.s: b_s, self.a: b_a,
                                                 self.ss: b_ss,
                                                 self.noise: b_noise,
                                                 self.batch_sequence_length: b_seq_len,
                                                 # self.drop_rate: 0.0,
                                                 })

        # recon_loss, kl_loss, vae_loss = self.sess.run([self.recon_loss, self.kl_loss, self.vae_loss],
        #                                               feed_dict={self.s: b_s, self.a: b_a,
        #                                                          self.ss: b_ss,
        #                                                          self.noise: b_noise,
        #                                                          self.drop_rate: 0.0})

        # return vae_loss
        return 0

    def train_actor(self):
        indices = np.random.choice(min(self.memory_size, self.pointer), size=self.m_batch_size)
        batch_exps = self.memory[indices, :]

        b_s = batch_exps[:, :self.s_dim]
        # b_noise = np.clip(np.random.normal(0, 1, size=[self.a_batch_size, self.z_dim]), -0.2, 0.2)
        b_noise = np.clip(np.random.normal(0, 1, size=[self.a_batch_size, self.z_dim]), -self.clip_value, self.clip_value)

        self.sess.run(self.a_train, feed_dict={self.s: b_s, self.noise: b_noise})

        self.train_cnt += 1

        # return actor_loss
        return 0

    def store_experience(self, trajectory, is_padding=False):
        s_traj, a_traj, r_traj = trajectory

        # for the convenience of manipulation
        arr_s_traj = np.array(s_traj)
        arr_a_traj = np.array(a_traj)
        arr_r_traj = np.array(r_traj)

        zero_pads = np.zeros(shape=[self.sequence_length, self.s_dim + self.a_dim])

        # for i in range(len(s_traj) - self.sequence_length):
        for i in range(len(s_traj) - self.min_sequence_length):
            tmp_s = arr_s_traj[i]
            tmp_a = arr_a_traj[i]
            tmp_soff = arr_s_traj[i:i + self.sequence_length]
            tmp_aoff = arr_a_traj[i:i + self.sequence_length]
            tmp_saoff = np.concatenate([tmp_soff, tmp_aoff], axis=1)

            tmp_sequence_length = tmp_saoff.shape[0]
            tmp_saoff_padded = np.concatenate([tmp_saoff, zero_pads], axis=0)
            tmp_saoff_padded_clip = tmp_saoff_padded[:self.sequence_length]
            tmp_soff_resh = tmp_saoff_padded_clip.reshape((self.s_dim + self.a_dim) * self.sequence_length)

            tmp_roff = arr_r_traj[i:i + self.sequence_length]
            tmp_u = np.matmul(tmp_roff, np.power(self.gamma, [j for j in range(len(tmp_roff))]))

            tmp_exp = tmp_s.tolist() + tmp_a.tolist() + [tmp_u] + tmp_soff_resh.tolist() + [tmp_sequence_length]

            index = self.pointer % self.memory_size
            self.memory[index, :] = tmp_exp
            self.pointer += 1

