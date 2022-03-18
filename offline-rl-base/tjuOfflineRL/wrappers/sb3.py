from typing import TYPE_CHECKING, Any, List, Tuple, Union

import numpy as np
from gym.spaces import Discrete

from ..algos import AlgoBase
from ..dataset import MDPDataset

if TYPE_CHECKING:
    from stable_baselines3.common.buffers import ReplayBuffer


class SB3Wrapper:
    """A wrapper for tjuOfflineRL algorithms so they can be used with Stable-Baselines3 (SB3).

    Args:
        algo (tjuOfflineRL.algos.base.AlgoBase): algorithm.

    Attributes:
        algo (tjuOfflineRL.algos.base.AlgoBase): algorithm.

    """

    def __init__(self, algo: AlgoBase):
        # Avoid infinite recursion due to override of setattr
        self.__dict__["algo"] = algo

    def predict(
        self,
        observation: Union[np.ndarray, List[Any]],
        state: Any = None,
        mask: Any = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """Returns actions.

        Args:
            observation: observation.
            state: this argument is just ignored.
            mask: this argument is just ignored.
            deterministic: flag to return greedy actions.

        Returns:
            ``(actions, None)``.

        """
        if deterministic:
            return self.algo.predict(observation), None
        return self.algo.sample_action(observation), None

    def __getattr__(self, attr: str) -> Any:
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.algo, attr)

    def __setattr__(self, attr_name: str, value: Any) -> None:
        if attr_name != "algo":
            self.algo.__setattr__(attr_name, value)
        else:
            self.__dict__["algo"] = value


def to_mdp_dataset(replay_buffer: "ReplayBuffer") -> MDPDataset:
    """Returns tjuOfflineRL's MDPDataset from SB3's ReplayBuffer

    Args:
        replay_buffer: SB3's replay buffer.

    Returns:
        tjuOfflineRL's MDPDataset.

    """
    pos = replay_buffer.size()
    discrete_action = isinstance(replay_buffer.action_space, Discrete)
    dataset = MDPDataset(
        observations=replay_buffer.observations[:pos, 0],
        actions=replay_buffer.actions[:pos, 0],
        rewards=replay_buffer.rewards[:pos, 0],
        terminals=replay_buffer.dones[:pos, 0],
        discrete_action=discrete_action,
    )
    return dataset
