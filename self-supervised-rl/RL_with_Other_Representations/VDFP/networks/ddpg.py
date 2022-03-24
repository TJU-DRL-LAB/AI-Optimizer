"""
    Deep Deterministic Policy Gradient (DDPG) Algorithm

"""

import tensorflow as tf
import numpy as np


class DDPG(object):
    def __init__(self, s_dim, a_dim, sess,
                 lr_a=0.0001, lr_c=0.001,
                 gamma=0.99, tau=0.001,
                 batch_size=64, memory_size=50000):
        self.s_dim, self.a_dim = s_dim, a_dim
        self.lr_a, self.lr_c= lr_a, lr_c
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

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.s, scope='eval', trainable=True)
            a_ = self._build_a(self.s_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            q = self._build_c(self.s, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.s_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.a_soft_replace = [tf.assign(ta, (1 - self.tau) * ta + self.tau * ea)
                               for ta, ea in zip(self.at_params, self.ae_params)]
        self.c_soft_replace = [tf.assign(tc, (1 - self.tau) * tc + self.tau * ec)
                                for tc, ec in zip(self.ct_params, self.ce_params)]

        q_target = self.r + self.done * self.gamma * q_

        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(self.td_error, var_list=self.ce_params)

        self.a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())
        self._print_hyperparams()

        print('-- INFO: DDPG initialized.')
        print('==========================')

    def _print_hyperparams(self):
        print('------------------- Hyperparameters ----------------------')
        print('-- S_Dim:', self.s_dim)
        print('-- A_Dim:', self.a_dim)
        print('-- LR_Critic:', self.lr_c)
        print('-- LR_Actor:', self.lr_a)
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

        a_loss, _ = self.sess.run([self.a_loss, self.atrain], {self.s: bs})
        c_loss, _ = self.sess.run([self.td_error, self.ctrain], {self.s: bs, self.a: ba,
                                                     self.r: np.reshape(br, [-1,1]),
                                                     self.s_: bs_,
                                                     self.done: np.reshape(bdone, [-1, 1])})

        # soft target replacement
        self.sess.run([self.a_soft_replace, self.c_soft_replace])

        self.train_cnt += 1

        return a_loss, c_loss

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

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            fc1 = tf.layers.dense(s, 200, activation=tf.nn.relu, name='fc1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.s_dim)))
            concat = tf.concat([fc1, a], axis=1, name='fc1_a_concat')
            fc2 = tf.layers.dense(concat, 100, activation=tf.nn.relu, name='fc2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / (200 + self.a_dim))))
            q = tf.layers.dense(fc2, 1, activation=None, trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 100)))
            return q
