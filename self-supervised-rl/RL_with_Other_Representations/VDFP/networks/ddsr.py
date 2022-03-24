"""
    Deterministic Deep Succesor Representation (DDSR) Algorithm

"""

import tensorflow as tf
import numpy as np


class DDSR(object):
    def __init__(self, s_dim, a_dim, sess,
                 lr_a=0.0001, lr_c=0.001, lr_r=0.001,
                 gamma=0.99, tau=0.001,
                 m_dim=50,
                 batch_size=64, memory_size=50000):

        self.s_dim, self.a_dim = s_dim, a_dim
        self.m_dim = m_dim
        self.lr_a, self.lr_c= lr_a, lr_c
        self.lr_r = lr_r

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.memory_size = memory_size
        self.memory = np.zeros((memory_size, s_dim * 2 + a_dim + 2), dtype=np.float32)
        self.pointer = 0
        self.sess = sess

        self.train_cnt = 0

        self.s = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.s_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')
        self.done = tf.placeholder(tf.float32, [None, 1], 'done_flag')

        self.m = self._build_m(self.s)
        self.recon = self._build_recon(self.m)

        self.m_ = self._build_m(self.s_, reuse=True)
        self.r_ime = self._build_r(self.m)

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.s, scope='eval', trainable=True)
            a_ = self._build_a(self.s_, scope='target', trainable=False)
        with tf.variable_scope('Successor'):
            sr = self._build_sr(self.m, self.a, scope='eval', trainable=True)
            sr_ = self._build_sr(self.m_, a_, scope='target', trainable=False)

        # networks parameters
        m_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Measurement')
        r_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Reward')
        recon_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Reconstruction')
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.se_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Successor/eval')
        self.st_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Successor/target')

        # target net replacement
        self.a_soft_replace = [tf.assign(ta, (1 - self.tau) * ta + self.tau * ea)
                               for ta, ea in zip(self.at_params, self.ae_params)]
        self.sr_soft_replace = [tf.assign(ts, (1 - self.tau) * ts + self.tau * es)
                                for ts, es in zip(self.st_params, self.se_params)]
        sr_target = self.m + self.done * self.gamma * sr_

        q = self._build_r(sr, reuse=True)

        self.recon_loss = tf.reduce_mean(tf.square(self.s - self.recon))
        self.r_loss = tf.reduce_mean(tf.square(self.r_ime - self.r))
        self.r_loss_total = self.recon_loss + self.r_loss
        self.r_train = tf.train.AdamOptimizer(learning_rate=self.lr_r).minimize(self.r_loss_total, var_list=m_params + r_params + recon_params)

        self.sr_td_error = tf.losses.mean_squared_error(labels=sr_target, predictions=sr)
        self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(self.sr_td_error, var_list=self.se_params)

        self.a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())
        self._print_hyperparams()

        print('-- INFO: DDSR initialized.')
        print('==========================')

    def _print_hyperparams(self):
        print('------------------- Hyperparameters ----------------------')
        print('-- S_Dim:', self.s_dim)
        print('-- A_Dim:', self.a_dim)
        print('-- M_Dim:', self.m_dim)
        print('-- LR_Critic:', self.lr_c)
        print('-- LR_Actor:', self.lr_a)
        print('-- LR_Reward:', self.lr_r)
        print('-- Gamma:', self.gamma)
        print('-- Tau:', self.tau)
        print('-- Batch_Size:', self.batch_size)
        print('-- Memory_Size:', self.memory_size)
        print('--')

    def choose_action(self, s):
        return self.sess.run(self.a, {self.s: s[np.newaxis, :]})[0]

    def learn(self):
        if self.pointer < self.batch_size:
            print('-- INFO: Memory less than batch size. Current num:', self.pointer)
            return

        indices = np.random.choice(min(self.memory_size, self.pointer), size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim]
        bs_ = bt[:, -self.s_dim - 1:-1]
        bdone = bt[:, -1]

        recon_loss, r_loss, _ = self.sess.run([self.recon_loss, self.r_loss, self.r_train], feed_dict={self.s: bs,
                                                                                      self.r: np.reshape(br, [-1,1])})
        c_loss, _ = self.sess.run([self.sr_td_error, self.ctrain], feed_dict={self.s: bs, self.a: ba,
                                                                              self.r: np.reshape(br, [-1,1]),
                                                                              self.s_: bs_,
                                                                              self.done: np.reshape(bdone, [-1, 1])})
        a_loss, _ = self.sess.run([self.a_loss, self.atrain], {self.s: bs})


        # soft target replacement
        self.sess.run([self.a_soft_replace, self.sr_soft_replace])

        self.train_cnt += 1

        return recon_loss, r_loss, a_loss, c_loss

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, [done]))
        index = self.pointer % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            fc1 = tf.layers.dense(s, 200, activation=tf.nn.relu, name='l1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.s_dim)))
            fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='l2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            a = tf.layers.dense(fc2, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 100)))
            return a


    def _build_m(self, s, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Measurement', reuse=reuse):
            fc1 = tf.layers.dense(s, 200, activation=tf.nn.relu, name='l1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.s_dim)))
            fc2 = tf.layers.dense(fc1, 100, activation=tf.nn.relu, name='l2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            m = tf.layers.dense(fc2, self.m_dim, activation=None, name='m', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 100)))

            return m

    def _build_recon(self, m, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Reconstruction', reuse=reuse):
            fc1 = tf.layers.dense(m, 100, activation=tf.nn.relu, name='l1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.m_dim)))
            fc2 = tf.layers.dense(fc1, 200, activation=tf.nn.relu, name='l2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 100)))
            recon = tf.layers.dense(fc2, self.s_dim, activation=None, name='recon', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))

            return recon

    def _build_r(self, m, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Reward', reuse=reuse):
            r = tf.layers.dense(m, 1, activation=None, name='reward', trainable=trainable, use_bias=False,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.m_dim)))
            return r

    def _build_sr(self, m, a, scope, trainable):
        with tf.variable_scope(scope):
            fc1 = tf.layers.dense(m, 200, activation=tf.nn.relu, name='fc1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.s_dim)))
            concat = tf.concat([fc1, a], axis=1, name='fc1_a_concat')
            fc2 = tf.layers.dense(concat, 100, activation=tf.nn.relu, name='fc2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / (200 + self.a_dim))))
            sr = tf.layers.dense(fc2, self.m_dim, activation=None, trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 100)))
            return sr

