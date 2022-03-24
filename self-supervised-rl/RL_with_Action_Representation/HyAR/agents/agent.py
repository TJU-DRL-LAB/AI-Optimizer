class Agent(object):
    """
    Defines a basic reinforcement learning agent for OpenAI Gym environments
    """

    NAME = "Abstract Agent"

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def act(self, state):
        """
        Determines the action to take in the given state.

        :param state:
        :return:
        """
        raise NotImplementedError

    def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
        """
        Performs a learning step given a (s,a,r,s',a') sample.

        :param state: previous observed state (s)
        :param action: action taken in previous state (a)
        :param reward: reward for the transition (r)
        :param next_state: the resulting observed state (s')
        :param next_action: action taken in next state (a')
        :param terminal: whether the episode is over
        :param time_steps: number of time steps the action took to execute (default=1)
        :return:
        """
        raise NotImplementedError

    def start_episode(self):
        """
        Perform any initialisation for the start of an episode.

        :return:
        """
        raise NotImplementedError

    def end_episode(self):
        """
        Performs any cleanup before the next episode.

        :return:
        """
        raise NotImplementedError

    def __str__(self):
        desc = self.NAME
        return desc
