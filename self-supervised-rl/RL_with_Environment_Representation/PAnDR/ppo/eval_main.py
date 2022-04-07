import os, random, sys
import gym
import numpy as np
import pdvf_utils
import env_utils

import embedding_networks
import myant
import myswimmer
import myspaceship

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from pandr_storage import ReplayMemoryPDVF_vl, ReplayMemoryPolicyDecoder
from pdvf_networks import PDVF, PDVF_ln

from pandr_arguments import get_args

from ppo.model import Policy
from ppo.envs import make_vec_envs

import env_utils
import pandr_utils
import train_utils

import myant
import myswimmer
from tensorboardX import SummaryWriter

from torch.autograd import Variable
import random
import time
from numbers import Number
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    
def evaluate_ppo(all_envs, all_policies_index, args)    
        print(torch.cuda.is_available())
    os.environ['OMP_NUM_THREADS'] = '1'
    norm = True
    device = torch.device("cuda")
    if device != 'cpu':
        torch.cuda.empty_cache()
    print(args.kl_lambda, args.seed)
    print(device)
    # Create the environment
    envs = make_vec_envs(args, device)
    seed_list = [2, 0, 2, 1, 4, 0, 4, 0, 2, 3, 2, 1, 3, 1, 4, 1, 3, 2, 4, 0]  # ds0
    # seed_list = [0, 4, 4, 3, 0, 4, 0, 4, 4, 4, 0, 1, 4, 1, 1, 3, 4, 0, 4, 4]
    # seed_list = [1, 1, 0, 1, 1, 2, 5, 1, 3, 6, 1, 3, 1, 3, 5, 2, 5, 2, 1, 1] # space1
    # seed_list = [2, 3, 1, 3, 1, 3, 1, 3, 2, 2, 3, 2, 4, 0, 0, 3, 4, 0, 1, 3]
    # seed_list = [2, 3, 2, 1, 0, 1, 1, 3, 2, 3, 2, 1, 3, 1, 4, 1, 3, 2, 4, 3]  # ant 2
    # seed_list = [0, 1, 4, 3, 0, 1, 1, 4, 3, 1, 0, 4, 1, 3, 1, 2, 4, 3, 3, 1] # swimmer
    # seed_list = [1, 2, 3, 4, 1, 2, 2, 4, 3, 1, 0, 0, 2, 3, 1, 3, 0, 2, 3, 2] spapceship 0
    # seed_list = [0, 1, 4, 3, 0, 1, 1, 4]
    names = []
    for e in range(args.num_envs):
        names.append('ppo.{}.env{}.seed{}.pt'.format(args.env_name, e, int(seed_list[e])))
    all_policies = []
    for name in names:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': False})
        actor_critic.to(device)
        model = os.path.join(args.save_dir, name)
        actor_critic.load_state_dict(torch.load(model))
        all_policies.append(actor_critic)

    env_sampler = env_utils.EnvSamplerPDVF(envs, all_policies, args)
    
    reward_list = [[], [], [], [], []]
    for i in range(NUM_EVAL_EPS):
        for ei in eval_envs:
            init_obs = torch.FloatTensor(env_sampler.env.reset(env_id=ei))

            if 'ant' in args.env_name or 'swimmer' in args.env_name:
                init_state = env_sampler.env.sim.get_state()
                res = env_sampler.zeroshot_sample_src_from_pol_state_mujoco(args, init_obs, sizes, policy_idx=ei,
                                                                            env_idx=ei)
            else:
                init_state = env_sampler.env.state
                res = env_sampler.zeroshot_sample_src_from_pol_state(args, init_obs, sizes, policy_idx=ei,
                                                                     env_idx=ei)

            source_env = res['source_env']
            mask_env = res['mask_env']
            source_policy = res['source_policy']
            init_episode_reward = res['episode_reward']
            reward_list[ei - 15].append(init_episode_reward)
    for ei in eval_envs:
        print(np.mean(np.array(reward_list[ei - 15])))
    envs.close()
    
    
if __name__ == '__main__':
    # a = [0, 5, 2, 7, 4, 6, 1, 3]
    # a = [6, 7, 3, 0, 4, 1, 5, 2]
    # a = [0, 7, 5, 1, 4, 3, 2, 6]
    # a = [0, 6, 5, 1, 2, 3, 4, 7]
    # a = [2, 3, 5, 7, 4, 0, 1, 6]
    # a = [0,1,2,3,4,5,6,7]
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # 0
    # a = [6, 10, 8, 4, 17, 19, 2, 0,7,18,1,16,9,12,5,3,15,11,14,13]
    # a = [19, 0, 14, 1, 15, 17, 18, 3, 9, 12, 2, 8, 11, 16, 6, 5, 4, 13, 7, 10] #2
    # a = [ 14, 2, 13, 16, 17,  1, 19, 18,  3,  0, 11, 15,  4, 7, 5, 10, 12, 6,  8, 9] # 3

    # loaded = np.load("mini-ant-10-50-sas-maskenv-sa-maskpi.npz",allow_pickle=True)
    # loaded = np.load("ant-100-200-sas-maskenv-sa-maskpi.npz",allow_pickle=True)
    # loaded = np.load("np3-ant-100-50-sas-maskenv-sa-maskpi.npz",allow_pickle=True)
    # loaded = np.load("np0-total-swim-100-max-sas-maskenv-sa-maskpi.npz", allow_pickle=True)
    # loaded = np.load("np0-total-ant-100-50-sas-maskenv-sa-maskpi.npz",allow_pickle=True)
    # loaded = np.load("np2-swimmer-100-50-sas-maskenv-sa-maskpi.npz",allow_pickle=True)
    # loaded = np.load("np0-spaceship-100-50-sas-maskenv-sa-maskpi.npz",allow_pickle=True)
    # loaded = np.load("np1-total-space-100-max-sas-maskenv-sa-maskpi.npz", allow_pickle=True)

    args = get_args()
    for i in range(2):
        # for i in range(5,10):

        args.num_epochs_emb_policy = 3000
        args.num_epochs_emb_env = 3000
        args.num_epochs_pdvf_phase1 = 3000
        args.num_epochs_emb_z = 1500
        args.seed = int(i+9)
        args.shuffle = 0
        args.data_set = 0
        args.num_eval_eps = 3
        # args.max_mutual_information=True

        args.use_information_bottleneck = False
        train_pdvf_ds(a, a, args)