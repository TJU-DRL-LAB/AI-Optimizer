"""
    run VDFP_ReLU in MuJoCo tasks

"""

import sys
import argparse
import tensorflow as tf
import numpy as np
import gym
import time

sys.path.append('./')
sys.path.append('../')

from networks.vdfp_relu import VDFP

#####################  hyper parameters  ####################

MAX_TOTAL_STEPS = 2000000
PRETRAIN_STEPS = 20000
RANDOM_STEPS = 5000

RENDER = False
# RENDER = True

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default='HalfCheetah-v1', help='env')
parser.add_argument("--seed", type=int, default=111, help='random_seed')
parser.add_argument("--seq_length", type=int, default=64, help='sequence_length')
parser.add_argument("--min_seq_length", type=int, default=16, help='min_seq_length')
parser.add_argument("--kl_weight", type=float, default=1000, help='kl_weight')
parser.add_argument("--clip_value", type=float, default=0.2, help='clip_value')
parser.add_argument("--gpu-no", type=str, default='-1', help='cuda_device')
parser.add_argument("--slope", type=float, default=0.8, help='slope_of_LeakyReLu')

args = parser.parse_args()
seed = args.seed
seq_length = args.seq_length
min_seq_length = args.min_seq_length
kl_weight = args.kl_weight
clip_value = args.clip_value
gpu_no = args.gpu_no
slope = args.slope

env_name = args.env

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

vdfp = VDFP(s_dim, a_dim, sess, gamma=0.99,
            sequence_length=seq_length, min_sequence_length=min_seq_length,
            m_batch_size=64,
            lr_vae=0.001, lr_r=0.0005, lr_a=0.00025,
            m_dim=100, z_dim=50,
            clip_value=clip_value,
            kl_coef=kl_weight, memory_size=100000,
            slope=slope)

########################### summary operations ##############################
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
                                   + '/vdfp_relu/m100k_b6464_m100_z50_gamma99_lr00025x42_tg1050_'
                                   + 'seqlen' + str(seq_length) + '_' + str(min_seq_length)
                                   + '_kl_' + str(kl_weight) + '_c_' + str(clip_value)
                                   + '_slope' + str(slope)
                                   + '/seed' + str(seed), sess.graph)

t1 = time.time()

train_count = 0
global_step_count = 0

r_loss_his, p_loss_his, a_loss_his = [], [], []

ep_num = 0

while global_step_count < MAX_TOTAL_STEPS:
    s = env.reset()
    ep_reward = 0

    traj = [[], [], []]

    ep_step_count = 0
    t = 0

    for ep_steps in range(MAX_EP_STEPS):
        # FIXME 0114 add time-varying feature
        s = s.tolist()
        s.append(t)
        s = np.array(s)

        a = vdfp.choose_action(s)

        noise = np.random.normal(0, 0.1, size=a_dim)
        a = np.clip(a + noise, -1, 1)

        scaled_a = np.multiply(a, a_bound)   # add randomness to action selection for exploration
        s_, r, done, info = env.step(scaled_a)

        t += t_inc

        traj[0].append(s)
        traj[1].append(a)
        traj[2].append(r)

        ep_step_count += 1
        global_step_count += 1

        if global_step_count > PRETRAIN_STEPS:
            # FIXME - losses are not calculated to reduce the training time (modify this if necessary)
            p_loss = vdfp.train_predictor()
            if p_loss is not None:
                p_loss_his.append(p_loss)
            a_loss = vdfp.train_actor()
            if a_loss is not None:
                a_loss_his.append(a_loss)
            train_count += 1

        s = s_
        ep_reward += r

        if done or global_step_count == MAX_TOTAL_STEPS:
            break

    vdfp.store_experience(trajectory=traj, is_padding=False)

    if global_step_count > RANDOM_STEPS:
        if global_step_count <= PRETRAIN_STEPS:
            train_num = global_step_count // 10 - (global_step_count - ep_step_count) // 10
            for i in range(train_num):
                r_loss = vdfp.train_reward_model()
                r_loss_his.append(r_loss)
            for i in range(ep_step_count):
                p_loss = vdfp.train_predictor()
                p_loss_his.append(p_loss)
                train_count += 1
        else:
            train_num = global_step_count // 50 - (global_step_count - ep_step_count) // 50
            for i in range(train_num):
                r_loss = vdfp.train_reward_model()
                r_loss_his.append(r_loss)

    print('- Episode:', ep_num, ' Reward: %i' % int(ep_reward),
          'Total steps:', global_step_count,
          'Train count:', train_count,
          'Loss:', sum(r_loss_his[-100:]) / 100.0, sum(p_loss_his[-100:]) / 100.0, sum(a_loss_his[-100:]) / 100.0)

    ep_num += 1

    if IS_SAVE_LOG:
        summary = sess.run(merged, feed_dict={var1: ep_reward,
                                              var2: sum(r_loss_his[-100:]) / 100.0,
                                              var3: sum(p_loss_his[-100:]) / 100.0,
                                              var4: sum(a_loss_his[-100:]) / 100.0,
                                              })
        writer.add_summary(summary, global_step_count)


print('Running time: ', time.time() - t1)

