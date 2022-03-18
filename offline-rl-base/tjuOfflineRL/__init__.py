import random

import numpy as np
import torch

from . import (
    algos,
    constants,
    dataset,
    datasets,
    dynamics,
    envs,
    metrics,
    models,
    online,
    ope,
    preprocessing,
    wrappers,
)
from ._version import __version__


def seed(n: int) -> None:
    """Sets random seed value.

    Args:
        n (int): seed value.

    """
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.backends.cudnn.deterministic = True
