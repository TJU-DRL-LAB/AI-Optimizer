"""
    run VDFP_PPO in MuJoCo tasks

"""

import sys
import argparse
import tensorflow as tf
import numpy as np
import gym
import time
import math

sys.path.append('./')
sys.path.append('../')
sys.path.append('../../')

from networks.vdfp_ppo import VDFP_PPO
import scipy.signal

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# print(os.environ.get('MUJOCO_PY_MJKEY_PATH'))
# print(os.environ.get('MUJOCO_PY_MJPRO_PATH'))

def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

def build_train_set(trajectories):
    """
    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t[0] for t in trajectories])
    actions = np.concatenate([t[1] for t in trajectories])
    disc_sum_rew = np.concatenate([t[4] for t in trajectories])
    advantages = np.concatenate([t[5] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


#####################  hyper parameters  ####################

MAX_TOTAL_STEPS = 2000000
PRETRAIN_STEPS = 20000
RANDOM_STEPS = 5000

RENDER = False
# RENDER = True

###############################  training  ####################################
parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default='Ant-v1', help='env')
parser.add_argument("--seed", type=int, default=111, help='random_seed')
parser.add_argument("--gamma", type=float, default=0.99, help='discount_factor')
parser.add_argument("--seq_length", type=int, default=256, help='sequence_length')
parser.add_argument("--min_seq_length", type=int, default=0, help='min_seq_length')
parser.add_argument("--kl_weight", type=float, default=1000, help='kl_weight')
parser.add_argument("--clip_value", type=float, default=0.2, help='clip_value')
parser.add_argument("--ti", type=int, default=2, help='train_interval')
parser.add_argument("--lamb", type=float, default=0, help='gae_lambda')
parser.add_argument("--gpu-no", type=str, default='-1', help='cuda_device')
args = parser.parse_args()
seed = args.seed
seq_length = args.seq_length
min_seq_length = args.min_seq_length
kl_weight = args.kl_weight
clip_value = args.clip_value

env_name = args.env
train_interval = args.ti
lamb = args.lamb
gpu_no = args.gpu_no
gamma = args.gamma

import os
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_no
if gpu_no == '-1':
    sess = tf.Session()
else:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

env = gym.make(env_name)
env = env.unwrapped

print('-- Env:', env_name)
print('-- Seed:', seed)

env.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

MAX_EP_STEPS = 1000
t_inc = 1.0 / MAX_EP_STEPS

s_dim = env.observation_space.shape[0] + 1
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

vdppo = VDFP_PPO(s_dim, a_dim, sess, gamma=gamma,
                 sequence_length=seq_length, min_sequence_length=min_seq_length,
                 vae_batch_size=128, r_batch_size=64,
                 lr_vae=0.001, lr_r=0.0005, lr_a=0.0001,
                 m_dim=100, z_dim=50,
                 clip_value=clip_value,
                 kl_coef=kl_weight, memory_size=100000,
                 c_epochs=10)

# --------------------------------- summary operations ------------------------------
# IS_SAVE_LOG = True
IS_SAVE_LOG = False
if IS_SAVE_LOG:
    var1 = tf.placeholder(tf.float32, None, name='episode_reward')
    var2 = tf.placeholder(tf.float32, None, name='r_loss')
    var3 = tf.placeholder(tf.float32, None, name='p_loss')
    var4 = tf.placeholder(tf.float32, None, name='a_loss')
    tf.summary.scalar('ep_reward', var1)
    tf.summary.scalar('ep_r_loss', var2)
    tf.summary.scalar('ep_p_loss', var3)
    tf.summary.scalar('ep_a_loss', var4)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tf_logs/' + env_name
                                   + '/vdfp_ppo/m100k_b64128_m100_z50_gamma' + gamma
                                   + '_lamd' + str(lamb * 100)
                                   + '_ce10_lr0001x105_tg1050_ti' + str(train_interval)
                                   + 'seqlen' + str(seq_length) + '_' + str(min_seq_length)
                                   + '_kl_' + str(kl_weight) + '_c_' + str(clip_value)
                                   + '/seed' + str(seed), sess.graph)

t1 = time.time()

train_count = 0
global_step_count = 0

r_loss_his, p_loss_his, a_loss_his = [], [], []

ep_num = 0

trajs = []

while global_step_count < MAX_TOTAL_STEPS:
    s = env.reset()
    ep_reward = 0

    obs, acts, rews = [], [], []

    ep_step_count = 0
    t = 0

    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()
            time.sleep(0.03)

        s = s.tolist()
        s.append(t)
        s = np.array(s)

        a = vdppo.choose_action(s)
        scaled_a = np.multiply(a, a_bound)
        s_, r, done, info = env.step(scaled_a)

        t += t_inc

        obs.append(s)
        acts.append(a)
        rews.append(r)

        ep_step_count += 1
        global_step_count += 1

        s = s_
        ep_reward += r

        if done or global_step_count == MAX_TOTAL_STEPS:
            break

    vdppo.store_experience(trajectory=[obs, acts, rews], is_padding=False)

    traj = [np.array(obs), np.array(acts), np.array(rews)]
    trajs.append(traj)
    trajs = trajs[-train_interval:]

    # FIXME -- train reward model
    if global_step_count > RANDOM_STEPS:
        if global_step_count <= PRETRAIN_STEPS:
            train_num = global_step_count // 10 - (global_step_count - ep_step_count) // 10
            for i in range(train_num):
                r_loss = vdppo.train_reward_model()
                r_loss_his.append(r_loss)
            if (ep_num + 1) % train_interval == 0:
                p_loss, p_losses = vdppo.train_predictor(trajs)
                p_loss_his += p_losses
                trajs = []

                train_count += 1

        else:
            train_num = global_step_count // 50 - (global_step_count - ep_step_count) // 50
            for i in range(train_num):
                r_loss = vdppo.train_reward_model()
                r_loss_his.append(r_loss)

            # TODO
            if (ep_num + 1) % train_interval == 0:
                for traj in trajs:
                    t_s, t_a, t_r = traj
                    t_v = vdppo.predict_v(t_s)
                    t_dsc_r = discount(t_r, gamma)
                    tds = t_r - t_v + np.append(t_v[1:] * gamma, 0)
                    t_adv = discount(tds, gamma * lamb)

                    traj.append(t_v)
                    traj.append(t_dsc_r)
                    traj.append(t_adv)

                observes, actions, advantages, disc_sum_rew = build_train_set(trajs)
                a_loss = vdppo.train_actor(observes, actions, advantages)
                p_loss, p_losses = vdppo.train_predictor(trajs)

                a_loss_his.append(a_loss)
                p_loss_his += p_losses

                trajs = []
                train_count += 1

    r_loss_his = r_loss_his[-100:]
    p_loss_his = p_loss_his[-100:]
    a_loss_his = a_loss_his[-100:]

    print('- Episode:', ep_num, ' Reward: %i' % int(ep_reward),
          'Total steps:', global_step_count,
          'Train count:', train_count,
          'Loss:',
          sum(r_loss_his) / 100.0,
          sum(p_loss_his) / 100.0,
          sum(a_loss_his) / 100.0)

    ep_num += 1

    if IS_SAVE_LOG:
        summary = sess.run(merged, feed_dict={var1: ep_reward,
                                              var2: sum(r_loss_his) / 100.0,
                                              var3: sum(p_loss_his) / 100.0,
                                              var4: sum(a_loss_his) / 100.0,
                                              })
        writer.add_summary(summary, global_step_count)


print('Running time: ', time.time() - t1)

