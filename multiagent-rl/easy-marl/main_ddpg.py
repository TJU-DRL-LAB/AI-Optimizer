import numpy as np
from buffer import Buffer
from hyperparameters import hyper_param_setting
from tensorboardX import SummaryWriter


def test_one_episode(args, agent):
    episode_reward = 0
    (observation_list, state) = env.reset()
    for step in range(args.episode_max_step):
        # env.render()
        continuous_action_list = agent.generate_action(observation_list)
        (next_observation_list, next_state), (reward_list, team_reward), done, info = env.step(continuous_action_list)
        episode_reward += team_reward
        if done:
            break
        observation_list = next_observation_list
    return episode_reward


def exploration(continuous_action_list, args):
    continuous_action_list = np.array(continuous_action_list)  # ndarray:(n_agent, action_dim)
    dim0 = args.action_dim_list[0]
    dim1 = args.action_dim_list[1]
    noise_size = dim0 * dim1
    exploration_noise = np.random.normal(0, args.exploration_var, noise_size).reshape((dim0, dim1))
    continuous_action_list = continuous_action_list + exploration_noise
    continuous_action_list = continuous_action_list.tolist()
    return continuous_action_list


def run(args, agent, env, buffer, logger):
    total_step = 0
    train_step = 0

    for episode in range(1, args.episode_count+1):
        episode_step = 0
        episode_reward = 0
        (observation_list, state) = env.reset()
        for step in range(args.episode_max_step):
            total_step += 1
            episode_step += 1
            # env.render()

            # sample action
            continuous_action_list = agent.generate_action(observation_list)

            # TODO: add exploration for continuous action space
            # continuous_action_list = exploration(continuous_action_list, args)
            (next_observation_list, next_state), (reward_list, team_reward), done, info = env.step(continuous_action_list)

            agent_specific = {"observation_list": observation_list, "continuous_action_list": continuous_action_list,
                              "reward_list": reward_list, "next_observation_list": next_observation_list}
            shared = {"state": state, "team_reward": team_reward, "done": done}
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
                actor_loss, critic_loss = agent.train(batch_experience_dict)
                # print("actor loss={}, critic_loss={}".format(actor_loss, critic_loss))
                logger.add_scalar('actor_loss', actor_loss, train_step)
                logger.add_scalar('critic_loss', critic_loss, train_step)
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
    if args.env_name == "continuous_meeting":
        from envs.continuous_meeting import Environment as Env
        env = Env()
    elif args.env_name == "continuous_mpe":
        from envs.continuous_mpe.mpe_env import MPEEnv as Env
        env = Env(args)
    else:
        raise ValueError('{} does not exist.'.format(args.env_name))

    # init agent's info according to the env's specific config
    args.agent_count = env.agent_count
    args.observation_dim_list = [obs_space.shape[0] for obs_space in env.observation_space]
    args.action_dim_list = [act_space.shape[0] for act_space in env.action_space]

    print(args.exp_name)
    if args.agent_name == 'IDDPG':
        from algorithms.DDPG_based.IDDPG import IDDPG as Agent
    elif args.agent_name == 'MADDPG':
        from algorithms.DDPG_based.MADDPG import MADDPG as Agent
    else:
        raise ValueError('{} does not exist.'.format(args.agent_name))

    # run exp_count times experiments
    for exp_id in range(1, args.exp_count + 1):
        args.exp_id = exp_id

        agent = Agent(args)
        buffer = Buffer(args)
        logger = SummaryWriter(log_dir=f"./logs/{args.exp_name}/{exp_id}")

        run(args, agent, env, buffer, logger)
        logger.close()
