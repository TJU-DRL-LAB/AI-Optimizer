import numpy as np
import scipy.signal


# from mpi_tools import mpi_statistics_scalar


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBufferPPO(object):
    """
    original from: https://github.com/bluecontra/tsallis_actor_critic_mujoco/blob/master/spinup/algos/ppo/ppo.py
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, discrete_action_dim, parameter_action_dim, size, gamma=0.99, lam=0.95):
        self.obs_dim = obs_dim
        self.discrete_action_dim = discrete_action_dim
        self.parameter_action_dim = parameter_action_dim

        self.size = size
        self.gamma, self.lam = gamma, lam
        self.ptr = 0
        self.path_start_idx, self.max_size = 0, size

        self.reset()

    def reset(self):
        self.obs_buf = np.zeros([self.size, self.obs_dim], dtype=np.float32)
        self.discrete_act_buf = np.zeros([self.size, self.discrete_action_dim], dtype=np.float32)
        self.parameter_act_buf = np.zeros([self.size, self.parameter_action_dim], dtype=np.float32)
        self.adv_buf = np.zeros(self.size, dtype=np.float32)
        self.rew_buf = np.zeros(self.size, dtype=np.float32)
        self.ret_buf = np.zeros(self.size, dtype=np.float32)
        self.val_buf = np.zeros(self.size, dtype=np.float32)
        self.discrete_logp_buf = np.zeros(self.size, dtype=np.float32)
        self.parameter_logp_buf = np.zeros(self.size, dtype=np.float32)

    def add(self, obs, discrete_action, parameter_action, rew, val, discrete_logp, parameter_logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.discrete_act_buf[self.ptr] = discrete_action
        self.parameter_act_buf[self.ptr] = parameter_action
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.discrete_logp_buf[self.ptr] = discrete_logp
        self.parameter_logp_buf[self.ptr] = parameter_logp

        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        # the next two lines implement GAE-Lambda advantage calculation
        # print("rews,vals",rews,vals)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        # print("deltas",deltas)
        self.adv_buf[path_slice] = discount(deltas, self.gamma * self.lam)
        # print("self.adv_buf[path_slice]",self.adv_buf[path_slice])

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        # assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.discrete_act_buf, self.parameter_act_buf, self.adv_buf,
                self.ret_buf, self.discrete_logp_buf, self.parameter_logp_buf]


class ReplayBuffer_MC(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, u, r = [], [], []

        for i in ind:
            X, U, R = self.storage[i]
            x.append(np.array(X, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))

        return np.array(x), np.array(u), np.array(r).reshape(-1, 1)


class ReplayBuffer_VDFP(object):
    def __init__(self, max_size=1e5):
        self.storage = []
        self.max_size = int(max_size)
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        s, a, u, x = [], [], [], []

        for i in ind:
            S, A, U, X = self.storage[i]
            s.append(np.array(S, copy=False))
            a.append(np.array(A, copy=False))
            u.append(np.array(U, copy=False))
            x.append(np.array(X, copy=False))

        return np.array(s), np.array(a), np.array(u).reshape(-1, 1), np.array(x)

    def sample_traj(self, batch_size, offset=0):
        ind = np.random.randint(0, len(self.storage) - int(offset), size=batch_size)
        if len(self.storage) == self.max_size:
            ind = (self.ptr + self.max_size - ind) % self.max_size
        else:
            ind = len(self.storage) - ind - 1
        # ind = (self.ptr - ind + len(self.storage)) % len(self.storage)
        s, a, x = [], [], []

        for i in ind:
            S, A, _, X = self.storage[i]
            s.append(np.array(S, copy=False))
            a.append(np.array(A, copy=False))
            x.append(np.array(X, copy=False))

        return np.array(s), np.array(a), np.array(x)

    def sample_traj_return(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        u, x = [], []

        for i in ind:
            _, _, U, X = self.storage[i]
            u.append(np.array(U, copy=False))
            x.append(np.array(X, copy=False))

        return np.array(u).reshape(-1, 1), np.array(x)


def store_experience(replay_buffer, trajectory, s_dim, a_dim,
                     sequence_length, min_sequence_length=0, is_padding=False, gamma=0.99,
                     ):
    s_traj, a_traj, r_traj = trajectory

    # for the convenience of manipulation
    arr_s_traj = np.array(s_traj)
    arr_a_traj = np.array(a_traj)
    arr_r_traj = np.array(r_traj)

    zero_pads = np.zeros(shape=[sequence_length, s_dim + a_dim])

    # for i in range(len(s_traj) - self.sequence_length):
    for i in range(len(s_traj) - min_sequence_length):
        tmp_s = arr_s_traj[i]
        tmp_a = arr_a_traj[i]
        tmp_soff = arr_s_traj[i:i + sequence_length]
        tmp_aoff = arr_a_traj[i:i + sequence_length]
        tmp_saoff = np.concatenate([tmp_soff, tmp_aoff], axis=1)

        tmp_saoff_padded = np.concatenate([tmp_saoff, zero_pads], axis=0)
        tmp_saoff_padded_clip = tmp_saoff_padded[:sequence_length, :]

        tmp_roff = arr_r_traj[i:i + sequence_length]
        tmp_u = np.matmul(tmp_roff, np.power(gamma, [j for j in range(len(tmp_roff))]))

        replay_buffer.add((tmp_s, tmp_a, tmp_u, tmp_saoff_padded_clip))


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    # return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0
        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)
        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) + (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            self.vars = np.maximum(0.0, self.vars)  # occasionally goes negative, clip
            self.means = new_means
            self.m += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1 / (np.sqrt(self.vars) + 0.1) / 3, self.means
