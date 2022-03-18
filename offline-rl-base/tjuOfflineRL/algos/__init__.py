from typing import Any, Dict, Type

from .awac import AWAC
from .base import AlgoBase
from .bc import BC, DiscreteBC
from .bcq import BCQ, DiscreteBCQ
from .bear import BEAR
from .combo import COMBO
from .cql import CQL, DiscreteCQL
from .crr import CRR
from .ddpg import DDPG
from .dqn import DQN, DoubleDQN
from .iql import IQL
from .mopo import MOPO
from .plas import PLAS, PLASWithPerturbation
from .random_policy import DiscreteRandomPolicy, RandomPolicy
from .redq import REDQ, REDQImpl
from .sac import SAC, DiscreteSAC
from .td3 import TD3
from .td3_plus_bc import TD3PlusBC

__all__ = [
    "AlgoBase",
    "AWAC",
    "BC",
    "DiscreteBC",
    "BCQ",
    "DiscreteBCQ",
    "BEAR",
    "COMBO",
    "CQL",
    "DiscreteCQL",
    "CRR",
    "DDPG",
    "DQN",
    "DoubleDQN",
    "IQL",
    "MOPO",
    "PLAS",
    "PLASWithPerturbation",
    "REDQ",
    "REDQImpl",
    "SAC",
    "DiscreteSAC",
    "TD3",
    "TD3PlusBC",
    "RandomPolicy",
    "DiscreteRandomPolicy",
    "get_algo",
    "create_algo",
]


DISCRETE_ALGORITHMS: Dict[str, Type[AlgoBase]] = {
    "bc": DiscreteBC,
    "bcq": DiscreteBCQ,
    "cql": DiscreteCQL,
    "dqn": DQN,
    "double_dqn": DoubleDQN,
    "sac": DiscreteSAC,
    "random": DiscreteRandomPolicy,
}

CONTINUOUS_ALGORITHMS: Dict[str, Type[AlgoBase]] = {
    "awac": AWAC,
    "bc": BC,
    "bcq": BCQ,
    "bear": BEAR,
    "combo": COMBO,
    "cql": CQL,
    "crr": CRR,
    "ddpg": DDPG,
    "iql": IQL,
    "mopo": MOPO,
    "plas": PLASWithPerturbation,
    "redq": REDQ,
    "sac": SAC,
    "td3": TD3,
    "td3_plus_bc": TD3PlusBC,
    "random": RandomPolicy,
}


def get_algo(name: str, discrete: bool) -> Type[AlgoBase]:
    """Returns algorithm class from its name.

    Args:
        name (str): algorithm name in snake_case.
        discrete (bool): flag to use discrete action-space algorithm.

    Returns:
        type: algorithm class.

    """
    if discrete:
        if name in DISCRETE_ALGORITHMS:
            return DISCRETE_ALGORITHMS[name]
        raise ValueError(f"{name} does not support discrete action-space.")
    if name in CONTINUOUS_ALGORITHMS:
        return CONTINUOUS_ALGORITHMS[name]
    raise ValueError(f"{name} does not support continuous action-space.")


def create_algo(name: str, discrete: bool, **params: Any) -> AlgoBase:
    """Returns algorithm object from its name.

    Args:
        name (str): algorithm name in snake_case.
        discrete (bool): flag to use discrete action-space algorithm.
        params (any): arguments for algorithm.

    Returns:
        tjuOfflineRL.algos.base.AlgoBase: algorithm.

    """
    return get_algo(name, discrete)(**params)
