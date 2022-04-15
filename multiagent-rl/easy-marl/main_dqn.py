import numpy as np
from hyperparameters import hyper_param_setting
from buffer import Buffer
from tensorboardX import SummaryWriter


def test_one_episode(args, agent):
    episode_reward = 0
    (observation_list, state) = env.reset()
    for step in range(args.episode_max_step):
        # env.render()
        q_list = agent.generate_q_list(observation_list)
        action_id_list = [np.argmax(q_list[i]) for i in range(args.agent_count)]
        (next_observation_list, next_state), (reward_list, team_reward), done, info = env.step(action_id_list)
        episode_reward += team_reward
        if done:
            break
        observation_list = next_observation_list
    return episode_reward


def exploration(q_list, epsilon, args):
    if np.random.random() < args.epsilon:
        action_id_list = []
        for i in range(args.agent_count):
            action_id = np.random.randint(0, args.action_dim_list[i])  # [low, high)
            action_id_list.append(action_id)
    else:
        action_id_list = []
        for i in range(args.agent_count):
            action_id_list.append(np.argmax(q_list[i]))
    # update epsilon(linear decay)
    if epsilon > args.min_epsilon:
        epsilon -= np.max([args.min_epsilon, epsilon - args.epsilon_decay])

    return action_id_list, epsilon


def run(args, agent, env, buffer, logger):
    total_step = 0
    train_step = 0
    epsilon = args.epsilon

    for episode in range(1, args.episode_count+1):
        episode_step = 0
        episode_reward = 0
        (observation_list, state) = env.reset()
        for step in range(args.episode_max_step):
            # env.render()
            total_step += 1
            episode_step += 1

            q_list = agent.generate_q_list(observation_list)
            action_id_list, epsilon = exploration(q_list, epsilon, args)
            (next_observation_list, next_state), (reward_list, team_reward), done, info = env.step(action_id_list)

            agent_specific = {"observation_list": observation_list, "action_id_list": action_id_list,
                              "reward_list": reward_list, "next_observation_list": next_observation_list}
            shared = {"state": state, "team_reward": team_reward, "next_state": next_state, "done": done}
            experience_dict = {"agent_specific": agent_specific, "shared": shared}
            buffer.append(experience_dict)
            episode_reward += team_reward

            if done:
                break
            observation_list = next_observation_list
            state = next_state

            # train the agent
            if total_step % args.train_interval == 0:
                batch_experience_dict = buffer.sample()
                loss = agent.train(batch_experience_dict)
                logger.add_scalar('loss', loss, train_step)
                train_step += 1
                if train_step % args.save_interval == 0:
                    agent.save_model(model_dir=f"./logs/{args.exp_name}/{exp_id}/{train_step}")

        print("Training episode: {}, Reward: {:.4f}, Episode_step: {}".format(episode, episode_reward, episode_step))
        logger.add_scalar('training_episode_reward', episode_reward, episode)

        # evaluate the agent
        if args.test_interval > 0 and episode % args.test_interval == 0:
            episode_reward_list = []
            for test_episode in range(args.test_episode_count):
                episode_reward_list.append(test_one_episode(args, agent))
            mean_episode_reward = np.mean(episode_reward_list)
            print("Training episode: {}, Test episode reward: {:.4f}".format(episode, mean_episode_reward))
            # add mean_episode_reward to logger
            logger.add_scalar('test_episode_reward', mean_episode_reward, episode)


if __name__ == '__main__':
    args = hyper_param_setting.parse_arguments()
    if args.env_name == "discrete_meeting":
        from envs.discrete_meeting import Environment as Env
        env = Env()
    elif args.env_name == "discrete_magym":
        from envs.discrete_magym.ma_gym_env import MAGYMEnv as Env
        env = Env(args)
    else:
        raise ValueError('{} does not exist.'.format(args.env_name))

    args.agent_count = env.agent_count
    args.observation_dim_list = [obs_space.shape[0] for obs_space in env.observation_space]
    args.state_dim = env.state_space.shape[0]
    args.action_dim_list = [act_space.n for act_space in env.action_space]

    print(args.exp_name)
    if args.agent_name == 'IDQN':
        from algorithms.DQN_based.IDQN import IDQN as Agent
    elif args.agent_name == 'VDN':
        from algorithms.DQN_based.VDN import VDN as Agent
    elif args.agent_name == 'QMIX':
        from algorithms.DQN_based.QMIX import QMIX as Agent
    elif args.agent_name == 'CommNet':
        from algorithms.DQN_based.CommNet import CommNet as Agent
    else:
        raise ValueError('{} does not exist.'.format(args.agent_name))

    for exp_id in range(1, args.exp_count + 1):
        # args.epsilon = 1.0  # always reset this value!!!
        args.exp_id = exp_id

        agent = Agent(args)
        buffer = Buffer(args)
        logger = SummaryWriter(log_dir=f"./logs/{args.exp_name}/{exp_id}")

        run(args, agent, env, buffer, logger)
        logger.close()