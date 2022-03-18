from abc import ABCMeta, abstractmethod
from typing import Iterator, List, cast

import numpy as np

from ..containers import FIFOQueue
from ..dataset import Transition, TransitionMiniBatch


class TransitionIterator(metaclass=ABCMeta):

    _transitions: List[Transition]
    _generated_transitions: FIFOQueue[Transition]
    _batch_size: int
    _n_steps: int
    _gamma: float
    _n_frames: int
    _real_ratio: float
    _real_batch_size: int
    _count: int

    def __init__(
        self,
        transitions: List[Transition],
        batch_size: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        n_frames: int = 1,
        real_ratio: float = 1.0,
        generated_maxlen: int = 100000,
    ):
        self._transitions = transitions
        self._generated_transitions = FIFOQueue(generated_maxlen)
        self._batch_size = batch_size
        self._n_steps = n_steps
        self._gamma = gamma
        self._n_frames = n_frames
        self._real_ratio = real_ratio
        self._real_batch_size = batch_size
        self._count = 0

    def __iter__(self) -> Iterator[TransitionMiniBatch]:
        self.reset()
        return self

    def __next__(self) -> TransitionMiniBatch:
        if len(self._generated_transitions) > 0:
            real_batch_size = self._real_batch_size
            fake_batch_size = self._batch_size - self._real_batch_size
            transitions = [self.get_next() for _ in range(real_batch_size)]
            transitions += self._sample_generated_transitions(fake_batch_size)
        else:
            transitions = [self.get_next() for _ in range(self._batch_size)]

        batch = TransitionMiniBatch(
            transitions,
            n_frames=self._n_frames,
            n_steps=self._n_steps,
            gamma=self._gamma,
        )

        self._count += 1

        return batch

    def reset(self) -> None:
        self._count = 0
        if len(self._generated_transitions) > 0:
            self._real_batch_size = int(self._real_ratio * self._batch_size)
        self._reset()

    @abstractmethod
    def _reset(self) -> None:
        pass

    @abstractmethod
    def _next(self) -> Transition:
        pass

    @abstractmethod
    def _has_finished(self) -> bool:
        pass

    def add_generated_transitions(self, transitions: List[Transition]) -> None:
        self._generated_transitions.extend(transitions)

    def get_next(self) -> Transition:
        if self._has_finished():
            raise StopIteration
        return self._next()

    def _sample_generated_transitions(
        self, batch_size: int
    ) -> List[Transition]:
        transitions: List[Transition] = []
        n_generated_transitions = len(self._generated_transitions)
        for _ in range(batch_size):
            index = cast(int, np.random.randint(n_generated_transitions))
            transitions.append(self._generated_transitions[index])
        return transitions

    @abstractmethod
    def __len__(self) -> int:
        pass

    def size(self) -> int:
        return len(self._transitions) + len(self._generated_transitions)

    @property
    def transitions(self) -> List[Transition]:
        return self._transitions

    @property
    def generated_transitions(self) -> FIFOQueue[Transition]:
        return self._generated_transitions
