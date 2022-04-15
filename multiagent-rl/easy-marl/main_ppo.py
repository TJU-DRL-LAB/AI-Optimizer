import numpy as np
from hyperparameters import hyper_param_setting
from buffer import Buffer
from tensorboardX import SummaryWriter


def test_one_episode(args, agent):
    episode_reward = 0
    (observation_list, state) = env.reset()
    for step in range(args.episode_max_step):
        action_id_list, action_prob_list = agent.generate_action_list(observation_list)
        (next_observation_list, next_state), (reward_list, team_reward), done, info = env.step(action_id_list)
        episode_reward += team_reward
        if done:
            break
        observation_list = next_observation_list
    return episode_reward


def run(args, agent, env, logger):
    total_step = 0
    train_step = 0
    for episode in range(1, args.episode_count+1):
        buffer = Buffer(args)
        episode_step = 0
        episode_reward = 0
        (observation_list, state) = env.reset()  # each observation in observation_list has a shape of (1, -1)
        for step in range(args.episode_max_step):
            # env.render()
            total_step += 1
            episode_step += 1

            action_id_list, action_prob_list = agent.generate_action_list(observation_list)
            (next_observation_list, next_state), (reward_list, team_reward), done, info = env.step(action_id_list)

            agent_specific = {
                "observation_list": observation_list,
                "action_id_list": action_id_list,
                "action_prob_list": action_prob_list,
                "reward_list": reward_list,
                "next_observation_list": next_observation_list
            }
            shared = {
                "state": state,
                "team_reward": team_reward,
                "done": done
            }
            experience_dict = {
                "agent_specific": agent_specific,
                "shared": shared
            }

            buffer.append(experience_dict)
            episode_reward += team_reward # team_reward....instead of reward_list[-1]

            if done:
                break
            observation_list = next_observation_list
            state = next_state

        # train the agent
        batch = buffer.sample()
        loss = agent.train(batch)
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
    elif args.env_name == "continuous_meeting":
        from envs.continuous_meeting import Environment as Env
        env = Env()
    elif args.env_name == "continuous_mpe":
        from envs.continuous_mpe.mpe_env import MPEEnv as Env
        env = Env(args)
    else:
        raise ValueError('{} does not exist.'.format(args.env_name))

    if args.env_name.startswith("discrete_"):
        args.agent_count = env.agent_count
        args.observation_dim_list = [obs_space.shape[0] for obs_space in env.observation_space]
        args.state_dim = env.state_space.shape[0]
        args.action_dim_list = [act_space.n for act_space in env.action_space]
    elif args.env_name.startswith("continuous_"):
        args.agent_count = env.agent_count
        args.observation_dim_list = [obs_space.shape[0] for obs_space in env.observation_space]
        args.state_dim = env.state_space.shape[0]
        args.action_dim_list = [act_space.shape[0] for act_space in env.action_space]

    print(args.exp_name)
    if args.agent_name == 'IPPO':
        from algorithms.PPO_based.IPPO import IPPO as Agent
    elif args.agent_name == 'MAPPO':
        from algorithms.PPO_based.MAPPO import MAPPO as Agent
    else:
        raise ValueError('{} does not exist.'.format(args.agent_name))

    for exp_id in range(1, args.exp_count + 1):

        agent = Agent(args)
        # buffer = Buffer(args)
        logger = SummaryWriter(log_dir=f"./logs/{args.exp_name}/{exp_id}")
        run(args, agent, env, logger)

