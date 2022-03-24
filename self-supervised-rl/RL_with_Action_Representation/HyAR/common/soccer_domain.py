import gym
from gym.spaces import Tuple, Box
import numpy as np
import psutil


def kill_soccer_server():
    # kill any old HFO server processes
    try:
        process_name = "rcssserver"
        for proc in psutil.process_iter():
            # check whether the process name matches
            if process_name in proc.name():
                proc.kill()
    except Exception as e:
        print(e)


class SoccerParameterisedActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    """
    def __init__(self, env):
        super(SoccerParameterisedActionWrapper, self).__init__(env)
        old_as = env.action_space
        num_actions = old_as.spaces[0].n
        self.action_space = Tuple((
            old_as.spaces[0],  # actions
            Tuple(  # parameters
                tuple(Box(old_as.spaces[i].low, old_as.spaces[i].high, dtype=np.float32)
                      for i in range(1, num_actions+1))
            )
        ))

    def action(self, action):
        """
        Convert ragged array action input to 1-D array for Soccer environment.

        :param action:
        :return:
        """
        act = np.concatenate((np.array([action[0],]), np.concatenate(action[1])))
        return act


class SoccerScaledParameterisedActionWrapper(gym.ActionWrapper):
    """
    Changes the scale of the continuous action parameters to [-1,1].
    Parameter space must be flattened!

    Tuple((
        Discrete(n),
        Box(c_1),
        Box(c_2),
        ...
        Box(c_n)
        )
    """

    def __init__(self, env):
        super(SoccerScaledParameterisedActionWrapper, self).__init__(env)
        self.old_as = env.action_space
        self.num_actions = self.old_as.spaces[0].n
        self.high = [self.old_as.spaces[i].high for i in range(1, self.num_actions + 1)]
        self.low = [self.old_as.spaces[i].low for i in range(1, self.num_actions + 1)]
        self.range = [self.old_as.spaces[i].high - self.old_as.spaces[i].low for i in range(1, self.num_actions + 1)]
        new_params = [  # parameters
            Box(-np.ones(self.old_as.spaces[i].low.shape), np.ones(self.old_as.spaces[i].high.shape), dtype=np.float32)
            for i in range(1, self.num_actions + 1)
        ]
        self.action_space = Tuple((
            self.old_as.spaces[0],  # actions
            *new_params,
        ))

    def action(self, action):
        """
        Rescale from [-1,1] to original action-parameter range.

        :param action:
        :return:
        """
        import copy
        action = copy.deepcopy(action)
        # action input is flattened...
        p = action[0]
        if p == 0:
            action[1] = self.range[0][0] * (action[1] + 1) / 2. + self.low[0][0]
            action[2] = self.range[0][1] * (action[2] + 1) / 2. + self.low[0][1]
        elif p == 1:
            action[3] = self.range[1] * (action[3] + 1) / 2. + self.low[1]
        elif p == 2:
            action[4] = self.range[2][0] * (action[4] + 1) / 2. + self.low[2][0]
            action[5] = self.range[2][1] * (action[5] + 1) / 2. + self.low[2][1]
        else:
            raise ValueError("Unhandled action", p)
        return action
