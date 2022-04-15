import gym


class BaseEnvironment(gym.Env):
    """
    task description:
    XXXXXX
    """
    def __init__(self):
        raise NotImplementedError()

    def reset(self):
        # TODO: return (observation_list, state)
        # note that: each observation in observation_list has a shape of (-1, ); and state shape is (-1, )
        # always normalize the observation
        raise NotImplementedError()

    def step(self, action_list):
        # TODO: return (next_observation_list, next_state), (reward_list, team_reward), done, info
        # note that: team_reward = sum(reward_list)
        raise NotImplementedError()

    def render(self):
        # TODO
        raise NotImplementedError()

