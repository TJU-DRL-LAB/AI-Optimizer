import numpy as np
import time
import torch
import rlkit.torch.pytorch_util as ptu
reward_weight = 1


def rollout(env, agent, exploration_agent, max_path_length=np.inf, accum_context=True, resample_z=False, animated=False, save_frames=False,
            explore=False, context=None, rsample=1):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :param accum_context: if True, accumulate the collected context
    :return:
    """
    observations = []
    actions = []
    rewards = []
    rewards_exp = []
    z_previous = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    goal_step = 0
    path_length = 0
    success_num = 0
    i = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        # print(o)
        if explore==False:
            a, agent_info = agent.get_action(o)
        else:
            a, agent_info = agent.get_action(o)

            exploration_agent.copy_z(agent.z)

            a, agent_info = exploration_agent.get_action(o)

        next_o, r, d, env_info = env.step(a)
        z_prev = agent.clear_z_sample()
        # update the agent's current context
        if explore == True:
            if agent.context is not None:
                z_prev = agent.encode(agent.context)

            agent.update_context([o, a, r, next_o, d, env_info])
            r_e = agent.explore_reward(z_prev, context)
            z_prev = torch.squeeze(z_prev, dim=0)

            r_e = r + reward_weight * r_e
        else:
            r_e = r
        if explore==False and accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        ####TODO
        if explore:
        #    if i % rsample==0:
            agent.infer_posterior(agent.context)
        ####
        i = i + 1
        z_prev = ptu.get_numpy(z_prev)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        rewards_exp.append(r_e)
        z_previous.append(z_prev)
        path_length += 1
        goal_step += 1

        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        # if env_info['success'] == 1:
        success_num += 1
        #    break
        if goal_step > max_path_length or d:
            break
        o = next_o
        if animated:
            env.render()
    z_previous = np.array(z_previous)
    if len(z_previous.shape) == 1:
        z_previous = np.expand_dims(z_previous, 1)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    #z_prev = np.array(z_previous)
    #if len(z_prev.shape) == 1:
    #    z_prev = np.expand_dims(z_prev, 1)
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        success=success_num,
        rewards_exp=np.array(rewards_exp).reshape(-1,1),
        z_previous=z_previous
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards_exp = [path["rewards_exp"] for path in paths]
    z_previous = [path["z_previous"] for path in paths]
    rewards = np.vstack(rewards)
    rewards_exp = np.vstack(rewards_exp)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    z_previous = np.vstack(z_previous)
    assert len(rewards.shape) == 2
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    assert len(z_previous.shape) == 2
    return rewards, rewards_exp, terminals, obs, actions, next_obs, z_previous


def split_paths_to_dict(paths):
    rewards, rewards_exp, terminals, obs, actions, next_obs, z_previous = split_paths(paths)
    return dict(
        rewards=rewards,
        rewards_exp=rewards_exp,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
        z_previous=z_previous,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
