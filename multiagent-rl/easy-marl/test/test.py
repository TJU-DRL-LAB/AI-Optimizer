
import os
import numpy as np

from envs.discrete_env.smac.smac_env import SMACEnv
#from envs.discrete_env.continuous_mpe.mpe_env import MPEEnv
#from envs.continuous_env.continuous_mpe.mpe_env import MPEEnv
#from envs.discrete_env.discrete_magym.ma_gym_env import MAGYMEnv
from utils.read_yaml import get_yaml_args


def test_env():
    args = get_yaml_args("./test.yaml")

    # ==================================
    # SMAC Test
    env = SMACEnv(args)

    obs = env.reset()

    done = False
    action_list = [1 for _ in range(env.n_agent)]
    while not done:
        obs, reward, done, info = env.step(action_list)
        available_action = info['available_action']
        action_list = [action.index(1) for action in available_action]

        print(f'current_step: {env.step_cnt}, reward: {reward}')

    
    # ==================================
    # MPE Test
    # env = MPEEnv(args)

    # obs = env.reset()
    # print(env.action_space)

    # done = False
    # action_list = [np.random.random(2) for _ in range(env.n_agent)]
    # while not done:
    #     obs, reward, done, info = env.step(action_list)
    #     print(f'current_step: {env.step_cnt}, reward: {reward}, done: {done}')

    # ==================================
    # MA-GYM Test
    # env = MAGYMEnv(args)

    # obs = env.reset()
    # print(env.action_space)

    # done = False
    # action_list = [1 for _ in range(env.n_agent)]
    # while not done:
    #     obs, reward, done, info = env.step(action_list)
    #     print(obs)
    #     print(f'current_step: {env.step_cnt}, reward: {reward}, done: {done}')

    # ==================================
    # GFootball Test
    # env = GFootballEnv(args)

    # obs = env.reset()
    # print(env.action_space)

    # done = False
    # action_list = [1 for _ in range(env.n_agent)]
    # while not done:
    #     obs, reward, done, info = env.step(action_list)
    #     print(f'current_step: {env.step_cnt}, reward: {reward}, done: {done}')





# def test_env_with_algos():
#     import numpy as np
#     from hyperparameters.IDQN_MPE import Hyperparameter  # need to be selectable based on command line
#     from algorithms.DQN_based.IDQN import IDQN as Agent  # need to be selectable based on command line
#     from buffer.buffer_step import Buffer

#     def exploration(q_list, args):
#         if np.random.random() < args.epsilon:
#             action_id_list = []
#             for i in range(args.agent_count):
#                 action_id = np.random.randint(0, args.action_dim_list[i])  # [low, high)
#                 action_id_list.append(action_id)
#             return action_id_list
#         else:
#             action_id_list = []
#             for i in range(args.agent_count):
#                 action_id_list.append(np.argmax(q_list[i]))
#             return action_id_list

    
#     def run(args, agent, env, buffer, logger):
#         episode_reward_list = []
#         for episode in range(args.episode_count):
#             # TODO: process args.epsilon

#             episode_reward = 0
#             obs_state_dict = env.reset()  # each observation in observation_list has a shape of (1, -1)
#             observation_list = [obs.reshape(1, -1) for obs in obs_state_dict["obs"]]

#             state = obs_state_dict["state"]
#             for step in range(args.episode_max_step):
#                 # env.render()

#                 q_list = agent.generate_q_list(observation_list)
#                 action_id_list = exploration(q_list, args)

#                 obs_state_dict, reward_list, done, info = env.step(action_id_list)
#                 next_observation_list = [obs.reshape(1, -1) for obs in obs_state_dict["obs"]]
#                 state = obs_state_dict["state"]

#                 buffer.append([observation_list, action_id_list, reward_list, next_observation_list, done, state])
#                 episode_reward += reward_list[-1]  # rewards[-1] is the global reward

#                 if done:
#                     break
#                 observation_list = next_observation_list

#                 # train the agent
#                 # TODO: if it is the time to train the agent
#                 batch = buffer.sample()
#                 loss = agent.train(batch)
#                 if reward_list[-1] != -0.2:
#                     print(episode, step, '==> reward and loss:', reward_list[-1], loss)

#             episode_reward_list.append(episode_reward)

#     args = Hyperparameter()

#     for exp_id in range(1, args.exp_count + 1):
#         args.epsilon = 1.0  # always reset this value!!!
#         args.exp_id = exp_id

#         agent = Agent(args)

#         env_args = get_yaml_args("./test.yaml")
#         env = MPEEnv(env_args)

#         buffer = Buffer(args)
#         logger = None
#         run(args, agent, env, buffer, logger)

if __name__ == '__main__':
    test_env()
    #test_env_with_algos()
    
    