import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


class VDFP_PPO(object):
    def __init__(self, s_dim, a_dim, sess, policy_logvar=-1.0,
                 lr_a=0.0005, lr_vae=0.001, lr_r=0.001,
                 gamma=0.99, kl_coef=1.0,
                 epsilon=0.2,
                 vae_batch_size=256, r_batch_size=64,
                 memory_size=50000,
                 m_dim=32, z_dim=16,
                 clip_value=0.2,
                 sequence_length=64, min_sequence_length=16,
                 aggregate_sequence_length=64,
                 filter_sizes=None, filters_nums=None,
                 c_epochs=10, a_epochs=10, clipping_range=0.2):
        self.sequence_length = sequence_length
        self.min_sequence_length = min_sequence_length
        self.aggregate_sequence_length = aggregate_sequence_length

        self.filter_sizes = [1, 2, 4, 8, 16, 32, 64] if filter_sizes is None else filter_sizes
        self.filters_nums = [20, 20, 10, 10, 5, 5, 5] if filters_nums is None else filters_nums

        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.c_epochs, self.a_epochs = c_epochs, a_epochs

        self.s_dim, self.a_dim = s_dim, a_dim
        self.m_dim, self.z_dim = m_dim, z_dim
        self.lr_a, self.lr_r, self.lr_vae = lr_a, lr_r, lr_vae
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_logvar = policy_logvar
        self.vae_batch_size, self.r_batch_size = vae_batch_size, r_batch_size
        self.clip_value = clip_value

        self.memory_size = memory_size
        # self.memory = []
        self.memory = np.zeros((memory_size, s_dim + a_dim + (s_dim + a_dim) * sequence_length + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = sess

        self.policy_logvar = policy_logvar
        self.clipping_range = clipping_range

        self.train_cnt = 0
        # self.are_coef = 0.2
        self.kl_coef = kl_coef

        self._placeholders()

        self.m = self._build_m(self.ss_ph)

        self.mean, log_std = self._build_encoder(self.s_ph, self.m)
        self.std = tf.exp(log_std)
        self.z = self.mean + self.std * self.noise_ph
        self.d = self._build_decoder(self.s_ph, self.z)
        self.r = self._build_r(self.m)

        # FIXME 0322
        self.mm_gen = self._build_decoder(self.s_ph, self.noise_ph, reuse=True)
        self.v = self._build_r(self.mm_gen, reuse=True)

        # actor
        self.means, self.log_vars = self._build_policy_net(self.s_ph, 'policy', trainable=True)
        self.logp, self.logp_old = self._logprob()

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        m_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Measurement')
        p_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Predictor')
        e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Encoder')
        r_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Return')
        d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Decoder')

        self.sampled_act = self.means + tf.exp(self.log_vars / 2.0) * tf.random_normal(shape=[self.a_dim,])

        # clipped surrogate objective
        pg_ratio = tf.exp(self.logp - self.logp_old)
        clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.clipping_range, 1 + self.clipping_range)
        surrogate_loss = tf.minimum(self.adv_ph * pg_ratio, self.adv_ph * clipped_pg_ratio)
        self.a_loss = - tf.reduce_mean(surrogate_loss)
        self.a_train_op = tf.train.AdamOptimizer(self.lr_a).minimize(self.a_loss)

        self.recon_loss = tf.reduce_mean(tf.square(self.m - self.d))
        self.kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + tf.log(tf.square(self.std))
                                                           - tf.square(self.mean) - tf.square(self.std), axis=1))
        self.vae_loss = self.recon_loss + self.kl_coef * self.kl_loss
        self.vae_train = tf.train.AdamOptimizer(learning_rate=self.lr_vae).minimize(self.vae_loss,
                                                                                    var_list=e_params + d_params)

        self.r_loss = tf.reduce_mean(tf.square(self.u_ph - self.r))
        self.r_train = tf.train.AdamOptimizer(learning_rate=self.lr_r).minimize(self.r_loss,
                                                                                var_list=m_params + r_params)

        self.sess.run(tf.global_variables_initializer())
        self._print_hyperparams()

        print('-- INFO: VDFP_PPO initialized.')
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
        # print('-- Batch_Size:', self.a_batch_size, self.m_batch_size)
        print('--')

    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.s_ph = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        self.a_ph = tf.placeholder(tf.float32, [None, self.a_dim], 'action')

        self.ss_ph = tf.placeholder(tf.float32, [None, self.sequence_length, self.s_dim + self.a_dim], 's_sequence')

        self.u_ph = tf.placeholder(tf.float32, [None, 1], 'utility')
        self.noise_ph = tf.placeholder(tf.float32, [None, self.z_dim], 'noise')
        self.drop_rate = tf.placeholder(tf.float32, None, name='drop_rate')  # input Action

        self.adv_ph = tf.placeholder(tf.float32, [None, ], 'advantages')
        # self.dcr_ph = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.val_ph = tf.placeholder(tf.float32, [None, 1], 'val_valfunc')

        self.old_log_vars_ph = tf.placeholder(tf.float32, [self.a_dim, ], 'old_log_vars')
        self.old_means_ph = tf.placeholder(tf.float32, [None, self.a_dim], 'old_means')

    def _build_encoder(self, s, m, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Encoder', reuse=reuse):
            efs = tf.layers.dense(s, 400, activation=tf.nn.sigmoid, name='efs', trainable=trainable,
                                   kernel_initializer=tf.random_normal_initializer(
                                       stddev=np.sqrt(1 / (self.s_dim + self.a_dim))))
            efm = tf.layers.dense(m, 400, activation=tf.nn.relu, name='efm', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.m_dim)))
            ef1 = efs * efm
            ef2 = tf.layers.dense(ef1, 200, activation=tf.nn.relu, name='ef2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 400)))
            mean = tf.layers.dense(ef2, self.z_dim, activation=None, name='mean', trainable=trainable,
                                   kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            log_std = tf.layers.dense(ef2, self.z_dim, activation=None, name='std', trainable=trainable,
                                      kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            #
            log_std_clipped = tf.clip_by_value(log_std, -4, 15)
            return mean, log_std_clipped

    def _build_decoder(self, s, z, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Decoder', reuse=reuse):
            efs = tf.layers.dense(s, 200, activation=tf.nn.sigmoid, name='efs', trainable=trainable,
                                   kernel_initializer=tf.random_normal_initializer(
                                       stddev=np.sqrt(1 / (self.s_dim + self.a_dim))))
            dfz = tf.layers.dense(z, 200, activation=tf.nn.relu, name='df1', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.z_dim)))
            df1 = efs * dfz
            df2 = tf.layers.dense(df1, 400, activation=tf.nn.relu, name='df2', trainable=trainable,
                                  kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 200)))
            # FIXME 1226 1-dimension or more (?)
            d = tf.layers.dense(df2, self.m_dim, activation=None, name='decoder', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 400)))
            return d

    def _build_m(self, ss, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Measurement', reuse=reuse):
            if self.sequence_length > 64:
                # FIXME -- aggregate long-trajectory
                ss_resh = tf.reshape(ss, [-1, self.sequence_length * (self.s_dim + self.a_dim)
                                          // self.aggregate_sequence_length])

                ss_agg = tf.layers.dense(ss_resh, self.s_dim + self.a_dim, activation=tf.nn.relu, name='ss_aggre',
                                         trainable=trainable,
                                         kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / (
                                             self.sequence_length * (self.s_dim + self.a_dim)
                                             // self.aggregate_sequence_length
                                         ))))

                ss = tf.reshape(ss_agg, [-1, self.aggregate_sequence_length, self.s_dim + self.a_dim, 1])
            else:
                ss = tf.reshape(ss, [-1, self.sequence_length, self.s_dim + self.a_dim, 1])
            convs = [tf.squeeze(tf.layers.conv2d(inputs=ss,
                                                 filters=n,
                                                 kernel_size=[h, self.s_dim + self.a_dim],
                                                 activation=tf.nn.relu),
                                axis=2)
                     for h, n in zip(self.filter_sizes, self.filters_nums)]

            pools = [tf.squeeze(tf.layers.max_pooling1d(inputs=conv, pool_size=[conv.shape[1]], strides=1),
                                axis=1)
                     for conv in convs]

            concat = tf.concat(pools, axis=1, name='conv_concat')
            highway = tf.layers.dense(concat, sum(self.filters_nums), activation=None, name='highway',
                                      trainable=trainable,
                                      kernel_initializer=tf.random_normal_initializer(
                                          stddev=np.sqrt(1 / sum(self.filters_nums))))
            joint = tf.nn.sigmoid(highway) * tf.nn.relu(highway) + (1. - tf.nn.sigmoid(highway)) * concat
            dropout = tf.layers.dropout(inputs=joint, rate=self.drop_rate, name='dropout')

            m = tf.layers.dense(dropout, self.m_dim, activation=None, name='m', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(
                                    stddev=np.sqrt(1 / sum(self.filters_nums))))

            return m

    def _build_r(self, m, reuse=False):
        trainable = True if not reuse else False
        with tf.variable_scope('Return', reuse=reuse):
            r = tf.layers.dense(m, 1, activation=None, name='return', trainable=trainable,
                                kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / self.m_dim)))

            return r

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
            log_vars = tf.reduce_sum(spd_log_vars, axis=0) + self.policy_logvar

        return means, log_vars

    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.a_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.a_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)

        return logp, logp_old

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sampled_act, feed_dict={self.s_ph: s})[0]
        return np.clip(a, -1, 1)

    def predict_v(self, s):
        """ Predict method """
        noise = np.clip(np.random.normal(0, 1, size=[s.shape[0], self.z_dim]), -self.clip_value, self.clip_value)
        y_hat = self.sess.run(self.v, feed_dict={self.s_ph: s, self.noise_ph: noise})

        return np.squeeze(y_hat)

    def train_actor(self, observes, actions, advantages):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)
        """
        feed_dict = {self.s_ph: observes,
                     self.a_ph: actions,
                     self.adv_ph: advantages,
                     }

        old_means_np, old_log_vars_np = self.sess.run([self.means, self.log_vars], feed_dict)
        feed_dict[self.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.old_means_ph] = old_means_np

        a_loss = 0
        for e in range(self.a_epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            self.sess.run(self.a_train_op, feed_dict)
            a_loss = self.sess.run(self.a_loss, feed_dict)

        return a_loss

    def train_reward_model(self):
        indices = np.random.choice(min(self.memory_size, self.pointer), size=self.r_batch_size)
        batch_exps = self.memory[indices, :]

        b_s = batch_exps[:, :self.s_dim]
        b_u = batch_exps[:, self.s_dim + self.a_dim]
        b_saoff = batch_exps[:, -(self.s_dim + self.a_dim) * self.sequence_length:]
        b_ss = np.reshape(b_saoff, [-1, self.sequence_length, self.s_dim + self.a_dim])

        r_loss, _ = self.sess.run([self.r_loss, self.r_train],
                                  feed_dict={self.s_ph: b_s, self.ss_ph: b_ss,
                                             self.u_ph: b_u.reshape((-1, 1)),
                                             self.drop_rate: 0.2})

        return r_loss

    def train_predictor(self, trajs):
        x, y = [], []
        zero_pads = np.zeros(shape=[self.sequence_length, self.s_dim + self.a_dim])
        for t in trajs:
            arr_s_traj = t[0]
            arr_a_traj = t[1]

            for i in range(arr_s_traj.shape[0] - self.min_sequence_length):
                tmp_s = arr_s_traj[i]
                tmp_a = arr_a_traj[i]
                tmp_soff = arr_s_traj[i:i + self.sequence_length]
                tmp_aoff = arr_a_traj[i:i + self.sequence_length]
                tmp_saoff = np.concatenate([tmp_soff, tmp_aoff], axis=1)

                tmp_saoff_padded = np.concatenate([tmp_saoff, zero_pads], axis=0)
                tmp_saoff_padded_clip = tmp_saoff_padded[:self.sequence_length, :]
                tmp_saoff_resh = tmp_saoff_padded_clip.reshape((self.s_dim + self.a_dim) * self.sequence_length)

                x.append(tmp_s)
                y.append(tmp_saoff_resh)

        x = np.array(x)
        y = np.array(y)
        num_batches = max(x.shape[0] // self.vae_batch_size, 1)
        batch_size = x.shape[0] // num_batches

        if self.replay_buffer_x is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate([x, self.replay_buffer_x])
            y_train = np.concatenate([y, self.replay_buffer_y])
        self.replay_buffer_x = x
        self.replay_buffer_y = y

        vae_losses = []

        for e in range(self.c_epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                b_s = x_train[start:end]
                b_ss = y_train[start:end].reshape([-1, self.sequence_length, self.s_dim + self.a_dim])
                b_noise = np.clip(np.random.normal(0, 1, size=[b_s.shape[0], self.z_dim]), -2.0, 2.0)
                feed_dict = {self.s_ph: b_s,
                             self.ss_ph: b_ss,
                             self.noise_ph: b_noise,
                             self.drop_rate: 0.0,
                             }
                _, l = self.sess.run([self.vae_train, self.vae_loss], feed_dict=feed_dict)
                vae_losses.append(l)

        vae_loss = sum(vae_losses) / (len(vae_losses) + 0.00001)
        return vae_loss, vae_losses

    def store_experience(self, trajectory, is_padding=False):
        s_traj, a_traj, r_traj = trajectory

        arr_s_traj = np.array(s_traj)
        arr_a_traj = np.array(a_traj)
        arr_r_traj = np.array(r_traj)

        zero_pads = np.zeros(shape=[self.sequence_length, self.s_dim + self.a_dim])

        # FIXME 0116 discard tails
        # for i in range(len(s_traj) - self.sequence_length):
        for i in range(len(s_traj) - self.min_sequence_length):
            tmp_s = arr_s_traj[i]
            tmp_a = arr_a_traj[i]
            tmp_soff = arr_s_traj[i:i + self.sequence_length]
            tmp_aoff = arr_a_traj[i:i + self.sequence_length]
            tmp_saoff = np.concatenate([tmp_soff, tmp_aoff], axis=1)

            tmp_saoff_padded = np.concatenate([tmp_saoff, zero_pads], axis=0)
            tmp_saoff_padded_clip = tmp_saoff_padded[:self.sequence_length, :]
            tmp_soff_resh = tmp_saoff_padded_clip.reshape((self.s_dim + self.a_dim) * self.sequence_length)
            # tmp_soff_resh = tmp_saoff.reshape((self.s_dim + self.a_dim) * self.sequence_length)

            tmp_roff = arr_r_traj[i:i + self.sequence_length]
            tmp_u = np.matmul(tmp_roff, np.power(self.gamma, [j for j in range(len(tmp_roff))]))

            # tmp_exp = [tmp_s, tmp_a, tmp_soff_resh, tmp_u]
            tmp_exp = tmp_s.tolist() + tmp_a.tolist() + [tmp_u] + tmp_soff_resh.tolist()

            index = self.pointer % self.memory_size
            self.memory[index, :] = tmp_exp
            self.pointer += 1