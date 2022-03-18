from typing import List, cast

import numpy as np

from ..dataset import Transition
from .base import TransitionIterator


class RandomIterator(TransitionIterator):

    _n_steps_per_epoch: int

    def __init__(
        self,
        transitions: List[Transition],
        n_steps_per_epoch: int,
        batch_size: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_frames: int = 1,
        real_ratio: float = 1.0,
        generated_maxlen: int = 100000,
    ):
        super().__init__(
            transitions=transitions,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            n_frames=n_frames,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
        )
        self._n_steps_per_epoch = n_steps_per_epoch

    def _reset(self) -> None:
        pass

    def _next(self) -> Transition:
        index = cast(int, np.random.randint(len(self._transitions)))
        transition = self._transitions[index]
        return transition

    def _has_finished(self) -> bool:
        return self._count >= self._n_steps_per_epoch

    def __len__(self) -> int:
        return self._n_steps_per_epoch
