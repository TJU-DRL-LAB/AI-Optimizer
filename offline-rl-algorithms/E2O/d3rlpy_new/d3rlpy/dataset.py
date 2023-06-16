# encoding: utf-8
# module d3rlpy.dataset
# from /home/zk/anaconda3/envs/mujoco37/lib/python3.7/site-packages/d3rlpy-1.0.0-py3.7-linux-x86_64.egg/d3rlpy/dataset.cpython-37m-x86_64-linux-gnu.so
# by generator 1.147
# no doc

# imports
import builtins as __builtins__  # <module 'builtins' (built-in)>
import copy as copy  # /home/zk/anaconda3/envs/mujoco37/lib/python3.7/copy.py
import warnings as warnings  # /home/zk/anaconda3/envs/mujoco37/lib/python3.7/warnings.py
import numpy as np  # /home/zk/anaconda3/envs/mujoco37/lib/python3.7/site-packages/numpy/__init__.py
import h5py as h5py  # /home/zk/anaconda3/envs/mujoco37/lib/python3.7/site-packages/h5py/__init__.py


# functions

def trace_back_and_clear(*args, **kwargs):  # real signature unknown
    """
    Traces transitions and clear all links.

        Args:
            transition (d3rlpy.dataset.Transition): transition.
    """
    pass


def _check_discrete_action(*args, **kwargs):  # real signature unknown
    pass


def _safe_size(*args, **kwargs):  # real signature unknown
    pass


def _to_episodes(*args, **kwargs):  # real signature unknown
    pass


def _to_transitions(*args, **kwargs):  # real signature unknown
    pass


def __pyx_unpickle_Enum(*args, **kwargs):  # real signature unknown
    pass


def __reduce_cython__(*args, **kwargs):  # real signature unknown
    pass


def __setstate_cython__(*args, **kwargs):  # real signature unknown
    pass


# classes

class Episode(object):
    """
    Episode class.

        This class is designed to hold data collected in a single episode.

        Episode object automatically splits data into list of
        :class:`d3rlpy.dataset.Transition` objects.
        Also Episode object behaves like a list object for ease of access to
        transitions.

        .. code-block:: python

            # return the number of transitions
            len(episode)

            # access to the first transition
            transitions = episode[0]

            # iterate through all transitions
            for transition in episode:
                pass

        Args:
            observation_shape (tuple): observation shape.
            action_size (int): dimension of action-space.
            observations (numpy.ndarray): observations.
            actions (numpy.ndarray): actions.
            rewards (numpy.ndarray): scalar rewards.
            terminal (bool): binary terminal flag. If False, the episode is not
                terminated by the environment (e.g. timeout).
    """

    def build_transitions(self, *args, **kwargs):  # real signature unknown
        """
        Builds transition objects.

                This method will be internally called when accessing the transitions
                property at the first time.
        """
        pass

    def compute_return(self, *args, **kwargs):  # real signature unknown
        """
        Computes sum of rewards.

                .. math::

                    R = \sum_{i=1} r_i

                Returns:
                    float: episode return.
        """
        pass

    def get_action_size(self, *args, **kwargs):  # real signature unknown
        """
        Returns dimension of action-space.

                Returns:
                    int: dimension of action-space.
        """
        pass

    def get_observation_shape(self, *args, **kwargs):  # real signature unknown
        """
        Returns observation shape.

                Returns:
                    tuple: observation shape.
        """
        pass

    def size(self, *args, **kwargs):  # real signature unknown
        """
        Returns the number of transitions.

                Returns:
                    int: the number of transitions.
        """
        pass

    def __getitem__(self, *args, **kwargs):  # real signature unknown
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __iter__(self, *args, **kwargs):  # real signature unknown
        pass

    def __len__(self, *args, **kwargs):  # real signature unknown
        pass

    actions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns the actions.

        Returns:
            numpy.ndarray: array of actions.

        """

    observations = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns the observations.

        Returns:
            numpy.ndarray: array of observations.

        """

    rewards = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns the rewards.

        Returns:
            numpy.ndarray: array of rewards.

        """

    terminal = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns the terminal flag.

        Returns:
            bool: the terminal flag.

        """

    transitions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns the transitions.

        Returns:
            list(d3rlpy.dataset.Transition):
                list of :class:`d3rlpy.dataset.Transition` objects.

        """

    __weakref__ = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """list of weak references to the object (if defined)"""

    __dict__ = None  # (!) real value is "mappingproxy({'__module__': 'd3rlpy.dataset', '__doc__': ' Episode class.\\n\\n    This class is designed to hold data collected in a single episode.\\n\\n    Episode object automatically splits data into list of\\n    :class:`d3rlpy.dataset.Transition` objects.\\n    Also Episode object behaves like a list object for ease of access to\\n    transitions.\\n\\n    .. code-block:: python\\n\\n        # return the number of transitions\\n        len(episode)\\n\\n        # access to the first transition\\n        transitions = episode[0]\\n\\n        # iterate through all transitions\\n        for transition in episode:\\n            pass\\n\\n    Args:\\n        observation_shape (tuple): observation shape.\\n        action_size (int): dimension of action-space.\\n        observations (numpy.ndarray): observations.\\n        actions (numpy.ndarray): actions.\\n        rewards (numpy.ndarray): scalar rewards.\\n        terminal (bool): binary terminal flag. If False, the episode is not\\n            terminated by the environment (e.g. timeout).\\n\\n    ', '__init__': <cyfunction Episode.__init__ at 0x7fd96ff1b1f0>, 'observations': <property object at 0x7fd96ff174d0>, 'actions': <property object at 0x7fd96ff17530>, 'rewards': <property object at 0x7fd96ff17590>, 'terminal': <property object at 0x7fd96ff175f0>, 'transitions': <property object at 0x7fd96ff17650>, 'build_transitions': <cyfunction Episode.build_transitions at 0x7fd96ff1b6d0>, 'size': <cyfunction Episode.size at 0x7fd96ff1b7a0>, 'get_observation_shape': <cyfunction Episode.get_observation_shape at 0x7fd96ff1b870>, 'get_action_size': <cyfunction Episode.get_action_size at 0x7fd96ff1b940>, 'compute_return': <cyfunction Episode.compute_return at 0x7fd96ff1ba10>, '__len__': <cyfunction Episode.__len__ at 0x7fd96ff1bae0>, '__getitem__': <cyfunction Episode.__getitem__ at 0x7fd96ff1bbb0>, '__iter__': <cyfunction Episode.__iter__ at 0x7fd96ff1bc80>, '__dict__': <attribute '__dict__' of 'Episode' objects>, '__weakref__': <attribute '__weakref__' of 'Episode' objects>})"


class MDPDataset(object):
    """
    Markov-Decision Process Dataset class.

        MDPDataset is deisnged for reinforcement learning datasets to use them like
        supervised learning datasets.

        .. code-block:: python

            from d3rlpy.dataset import MDPDataset

            # 1000 steps of observations with shape of (100,)
            observations = np.random.random((1000, 100))
            # 1000 steps of actions with shape of (4,)
            actions = np.random.random((1000, 4))
            # 1000 steps of rewards
            rewards = np.random.random(1000)
            # 1000 steps of terminal flags
            terminals = np.random.randint(2, size=1000)

            dataset = MDPDataset(observations, actions, rewards, terminals)

        The MDPDataset object automatically splits the given data into list of
        :class:`d3rlpy.dataset.Episode` objects.
        Furthermore, the MDPDataset object behaves like a list in order to use with
        scikit-learn utilities.

        .. code-block:: python

            # returns the number of episodes
            len(dataset)

            # access to the first episode
            episode = dataset[0]

            # iterate through all episodes
            for episode in dataset:
                pass

        Args:
            observations (numpy.ndarray): N-D array. If the
                observation is a vector, the shape should be
                `(N, dim_observation)`. If the observations is an image, the shape
                should be `(N, C, H, W)`.
            actions (numpy.ndarray): N-D array. If the actions-space is
                continuous, the shape should be `(N, dim_action)`. If the
                action-space is discrete, the shape should be `(N,)`.
            rewards (numpy.ndarray): array of scalar rewards. The reward function
                should be defined as :math:`r_t = r(s_t, a_t)`.
            terminals (numpy.ndarray): array of binary terminal flags.
            episode_terminals (numpy.ndarray): array of binary episode terminal
                flags. The given data will be splitted based on this flag.
                This is useful if you want to specify the non-environment
                terminations (e.g. timeout). If ``None``, the episode terminations
                match the environment terminations.
            discrete_action (bool): flag to use the given actions as discrete
                action-space actions. If ``None``, the action type is automatically
                determined.
    """

    def append(self, *args, **kwargs):  # real signature unknown
        """
        Appends new data.

                Args:
                    observations (numpy.ndarray): N-D array.
                    actions (numpy.ndarray): actions.
                    rewards (numpy.ndarray): rewards.
                    terminals (numpy.ndarray): terminals.
                    episode_terminals (numpy.ndarray): episode terminals.
        """
        pass

    def build_episodes(self, *args, **kwargs):  # real signature unknown
        """
        Builds episode objects.

                This method will be internally called when accessing the episodes
                property at the first time.
        """
        pass

    def compute_stats(self):  # real signature unknown; restored from __doc__
        """
        Computes statistics of the dataset.

                .. code-block:: python

                    stats = dataset.compute_stats()

                    # return statistics
                    stats['return']['mean']
                    stats['return']['std']
                    stats['return']['min']
                    stats['return']['max']

                    # reward statistics
                    stats['reward']['mean']
                    stats['reward']['std']
                    stats['reward']['min']
                    stats['reward']['max']

                    # action (only with continuous control actions)
                    stats['action']['mean']
                    stats['action']['std']
                    stats['action']['min']
                    stats['action']['max']

                    # observation (only with numpy.ndarray observations)
                    stats['observation']['mean']
                    stats['observation']['std']
                    stats['observation']['min']
                    stats['observation']['max']

                Returns:
                    dict: statistics of the dataset.
        """
        pass

    def dump(self, *args, **kwargs):  # real signature unknown
        """
        Saves dataset as HDF5.

                Args:
                    fname (str): file path.
        """
        pass

    def extend(self, *args, **kwargs):  # real signature unknown
        """
        Extend dataset by another dataset.

                Args:
                    dataset (d3rlpy.dataset.MDPDataset): dataset.
        """
        pass

    def get_action_size(self, *args, **kwargs):  # real signature unknown
        """
        Returns dimension of action-space.

                If `discrete_action=True`, the return value will be the maximum index
                +1 in the give actions.

                Returns:
                    int: dimension of action-space.
        """
        pass

    def get_observation_shape(self, *args, **kwargs):  # real signature unknown
        """
        Returns observation shape.

                Returns:
                    tuple: observation shape.
        """
        pass

    def is_action_discrete(self, *args, **kwargs):  # real signature unknown
        """
        Returns `discrete_action` flag.

                Returns:
                    bool: `discrete_action` flag.
        """
        pass

    @classmethod
    def load(cls, dataset_h5):  # real signature unknown; restored from __doc__
        """
        Loads dataset from HDF5.

                .. code-block:: python

                    import numpy as np
                    from d3rlpy.dataset import MDPDataset

                    dataset = MDPDataset(np.random.random(10, 4),
                                         np.random.random(10, 2),
                                         np.random.random(10),
                                         np.random.randint(2, size=10))

                    # save as HDF5
                    dataset.dump('dataset.h5')

                    # load from HDF5
                    new_dataset = MDPDataset.load('dataset.h5')

                Args:
                    fname (str): file path.
        """
        pass

    def size(self, *args, **kwargs):  # real signature unknown
        """
        Returns the number of episodes in the dataset.

                Returns:
                    int: the number of episodes.
        """
        pass

    def __getitem__(self, *args, **kwargs):  # real signature unknown
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    def __iter__(self, *args, **kwargs):  # real signature unknown
        pass

    def __len__(self, *args, **kwargs):  # real signature unknown
        pass

    actions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns the actions.

        Returns:
            numpy.ndarray: array of actions.

        """

    episodes = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns the episodes.

        Returns:
            list(d3rlpy.dataset.Episode):
                list of :class:`d3rlpy.dataset.Episode` objects.

        """

    episode_terminals = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns the episode terminal flags.

        Returns:
            numpy.ndarray: array of episode terminal flags.

        """

    observations = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns the observations.

        Returns:
            numpy.ndarray: array of observations.

        """

    rewards = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns the rewards.

        Returns:
            numpy.ndarray: array of rewards

        """

    terminals = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns the terminal flags.

        Returns:
            numpy.ndarray: array of terminal flags.

        """

    __weakref__ = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """list of weak references to the object (if defined)"""

    __dict__ = None  # (!) real value is "mappingproxy({'__module__': 'd3rlpy.dataset', '__doc__': ' Markov-Decision Process Dataset class.\\n\\n    MDPDataset is deisnged for reinforcement learning datasets to use them like\\n    supervised learning datasets.\\n\\n    .. code-block:: python\\n\\n        from d3rlpy.dataset import MDPDataset\\n\\n        # 1000 steps of observations with shape of (100,)\\n        observations = np.random.random((1000, 100))\\n        # 1000 steps of actions with shape of (4,)\\n        actions = np.random.random((1000, 4))\\n        # 1000 steps of rewards\\n        rewards = np.random.random(1000)\\n        # 1000 steps of terminal flags\\n        terminals = np.random.randint(2, size=1000)\\n\\n        dataset = MDPDataset(observations, actions, rewards, terminals)\\n\\n    The MDPDataset object automatically splits the given data into list of\\n    :class:`d3rlpy.dataset.Episode` objects.\\n    Furthermore, the MDPDataset object behaves like a list in order to use with\\n    scikit-learn utilities.\\n\\n    .. code-block:: python\\n\\n        # returns the number of episodes\\n        len(dataset)\\n\\n        # access to the first episode\\n        episode = dataset[0]\\n\\n        # iterate through all episodes\\n        for episode in dataset:\\n            pass\\n\\n    Args:\\n        observations (numpy.ndarray): N-D array. If the\\n            observation is a vector, the shape should be\\n            `(N, dim_observation)`. If the observations is an image, the shape\\n            should be `(N, C, H, W)`.\\n        actions (numpy.ndarray): N-D array. If the actions-space is\\n            continuous, the shape should be `(N, dim_action)`. If the\\n            action-space is discrete, the shape should be `(N,)`.\\n        rewards (numpy.ndarray): array of scalar rewards. The reward function\\n            should be defined as :math:`r_t = r(s_t, a_t)`.\\n        terminals (numpy.ndarray): array of binary terminal flags.\\n        episode_terminals (numpy.ndarray): array of binary episode terminal\\n            flags. The given data will be splitted based on this flag.\\n            This is useful if you want to specify the non-environment\\n            terminations (e.g. timeout). If ``None``, the episode terminations\\n            match the environment terminations.\\n        discrete_action (bool): flag to use the given actions as discrete\\n            action-space actions. If ``None``, the action type is automatically\\n            determined.\\n\\n    ', '__init__': <cyfunction MDPDataset.__init__ at 0x7fd96ff1a050>, 'observations': <property object at 0x7fd97457da10>, 'actions': <property object at 0x7fd96ff17290>, 'rewards': <property object at 0x7fd96ff172f0>, 'terminals': <property object at 0x7fd96ff17350>, 'episode_terminals': <property object at 0x7fd96ff173b0>, 'episodes': <property object at 0x7fd96ff17410>, 'size': <cyfunction MDPDataset.size at 0x7fd96ff1a6d0>, 'get_action_size': <cyfunction MDPDataset.get_action_size at 0x7fd96ff1a7a0>, 'get_observation_shape': <cyfunction MDPDataset.get_observation_shape at 0x7fd96ff1a870>, 'is_action_discrete': <cyfunction MDPDataset.is_action_discrete at 0x7fd96ff1a940>, 'compute_stats': <cyfunction MDPDataset.compute_stats at 0x7fd96ff1aa10>, 'append': <cyfunction MDPDataset.append at 0x7fd96ff1aae0>, 'extend': <cyfunction MDPDataset.extend at 0x7fd96ff1abb0>, 'dump': <cyfunction MDPDataset.dump at 0x7fd96ff1ac80>, 'load': <classmethod object at 0x7fd970310750>, 'build_episodes': <cyfunction MDPDataset.build_episodes at 0x7fd96ff1ae20>, '__len__': <cyfunction MDPDataset.__len__ at 0x7fd96ff1aef0>, '__getitem__': <cyfunction MDPDataset.__getitem__ at 0x7fd96ff1b050>, '__iter__': <cyfunction MDPDataset.__iter__ at 0x7fd96ff1b120>, '__dict__': <attribute '__dict__' of 'MDPDataset' objects>, '__weakref__': <attribute '__weakref__' of 'MDPDataset' objects>})"


class Transition(object):
    """
    Transition class.

        This class is designed to hold data between two time steps, which is
        usually used as inputs of loss calculation in reinforcement learning.

        Args:
            observation_shape (tuple): observation shape.
            action_size (int): dimension of action-space.
            observation (numpy.ndarray): observation at `t`.
            action (numpy.ndarray or int): action at `t`.
            reward (float): reward at `t`.
            next_observation (numpy.ndarray): observation at `t+1`.
            terminal (int): terminal flag at `t+1`.
            prev_transition (d3rlpy.dataset.Transition):
                pointer to the previous transition.
            next_transition (d3rlpy.dataset.Transition):
                pointer to the next transition.
    """

    def clear_links(self, *args, **kwargs):  # real signature unknown
        """
        Clears links to the next and previous transitions.

                This method is necessary to call when freeing this instance by GC.
        """
        pass

    def get_action_size(self, *args, **kwargs):  # real signature unknown
        """
        Returns dimension of action-space.

                Returns:
                    int: dimension of action-space.
        """
        pass

    def get_observation_shape(self, *args, **kwargs):  # real signature unknown
        """
        Returns observation shape.

                Returns:
                    tuple: observation shape.
        """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        pass

    action = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns action at `t`.

        Returns:
            (numpy.ndarray or int): action at `t`.

        """

    is_discrete = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """Returns flag of discrete action-space.

        Returns:
            bool: ``True`` if action-space is discrete.

        """

    next_observation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns observation at `t+1`.

        Returns:
            numpy.ndarray or torch.Tensor: observation at `t+1`.

        """

    next_transition = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns pointer to the next transition.

        If this is the last transition, this method should return ``None``.

        Returns:
            d3rlpy.dataset.Transition: next transition.

        """

    observation = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns observation at `t`.

        Returns:
            numpy.ndarray or torch.Tensor: observation at `t`.

        """

    prev_transition = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns pointer to the previous transition.

        If this is the first transition, this method should return ``None``.

        Returns:
            d3rlpy.dataset.Transition: previous transition.

        """

    reward = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns reward at `t`.

        Returns:
            float: reward at `t`.

        """

    terminal = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns terminal flag at `t+1`.

        Returns:
            int: terminal flag at `t+1`.

        """

    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x7fd974576660>'


class TransitionMiniBatch(object):
    """
    mini-batch of Transition objects.

        This class is designed to hold :class:`d3rlpy.dataset.Transition` objects
        for being passed to algorithms during fitting.

        If the observation is image, you can stack arbitrary frames via
        ``n_frames``.

        .. code-block:: python

            transition.observation.shape == (3, 84, 84)

            batch_size = len(transitions)

            # stack 4 frames
            batch = TransitionMiniBatch(transitions, n_frames=4)

            # 4 frames x 3 channels
            batch.observations.shape == (batch_size, 12, 84, 84)

        This is implemented by tracing previous transitions through
        ``prev_transition`` property.

        Args:
            transitions (list(d3rlpy.dataset.Transition)):
                mini-batch of transitions.
            n_frames (int): the number of frames to stack for image observation.
            n_steps (int): length of N-step sampling.
            gamma (float): discount factor for N-step calculation.
    """

    def size(self, *args, **kwargs):  # real signature unknown
        """
        Returns size of mini-batch.

                Returns:
                    int: mini-batch size.
        """
        pass

    def add_additional_data(self, *args, **kwargs):  # real signature unknown
        pass

    def __getitem__(self, *args, **kwargs):  # real signature unknown
        """ Return self[key]. """
        pass

    def __init__(self, transitions, n_frames=4):  # real signature unknown; restored from __doc__
        pass

    def __iter__(self, *args, **kwargs):  # real signature unknown
        """ Implement iter(self). """
        pass

    def __len__(self, *args, **kwargs):  # real signature unknown
        """ Return len(self). """
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):  # real signature unknown
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __reduce__(self, *args, **kwargs):  # real signature unknown
        pass

    def __setstate__(self, *args, **kwargs):  # real signature unknown
        pass

    actions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns mini-batch of actions at `t`.

        Returns:
            numpy.ndarray: actions at `t`.

        """

    next_observations = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns mini-batch of observations at `t+n`.

        Returns:
            numpy.ndarray or torch.Tensor: observations at `t+n`.

        """

    n_steps = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns mini-batch of the number of steps before next observations.

        This will always include only ones if ``n_steps=1``. If ``n_steps`` is
        bigger than ``1``. the values will depend on its episode length.

        Returns:
            numpy.ndarray: the number of steps before next observations.

        """

    observations = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns mini-batch of observations at `t`.

        Returns:
            numpy.ndarray or torch.Tensor: observations at `t`.

        """

    rewards = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns mini-batch of rewards at `t`.

        Returns:
            numpy.ndarray: rewards at `t`.

        """

    terminals = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns mini-batch of terminal flags at `t+n`.

        Returns:
            numpy.ndarray: terminal flags at `t+n`.

        """

    transitions = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """ Returns transitions.

        Returns:
            d3rlpy.dataset.Transition: list of transitions.

        """

    __pyx_vtable__ = None  # (!) real value is '<capsule object NULL at 0x7fd974576690>'


# variables with complex values

LOG = None  # (!) real value is "<BoundLoggerLazyProxy(logger=None, wrapper_class=None, processors=None, context_class=None, initial_values={}, logger_factory_args=('d3rlpy.logger',))>"

__loader__ = None  # (!) real value is '<_frozen_importlib_external.ExtensionFileLoader object at 0x7fd974569b90>'

__spec__ = None  # (!) real value is "ModuleSpec(name='d3rlpy.dataset', loader=<_frozen_importlib_external.ExtensionFileLoader object at 0x7fd974569b90>, origin='/home/zk/anaconda3/envs/mujoco37/lib/python3.7/site-packages/d3rlpy-1.0.0-py3.7-linux-x86_64.egg/d3rlpy/dataset.cpython-37m-x86_64-linux-gnu.so')"

__test__ = {}

