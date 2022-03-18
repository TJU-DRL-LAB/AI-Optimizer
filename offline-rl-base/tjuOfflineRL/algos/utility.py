from typing import List, Optional, Tuple, cast

import numpy as np

from ..constants import DYNAMICS_NOT_GIVEN_ERROR, IMPL_NOT_INITIALIZED_ERROR
from ..dataset import Transition, TransitionMiniBatch
from ..dynamics import DynamicsBase
from .base import AlgoImplBase


class ModelBaseMixin:
    _grad_step: int
    _impl: Optional[AlgoImplBase]
    _dynamics: Optional[DynamicsBase]

    def generate_new_data(
        self, transitions: List[Transition]
    ) -> Optional[List[Transition]]:
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        assert self._dynamics, DYNAMICS_NOT_GIVEN_ERROR

        if not self._is_generating_new_data():
            return None

        init_transitions = self._sample_initial_transitions(transitions)

        rets: List[Transition] = []

        # rollout
        batch = TransitionMiniBatch(init_transitions)
        observations = batch.observations
        actions = self._sample_rollout_action(observations)
        prev_transitions: List[Transition] = []
        for _ in range(self._get_rollout_horizon()):
            # predict next state
            pred = self._dynamics.predict(observations, actions, True)
            pred = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], pred)
            next_observations, rewards, variances = pred

            # regularize by uncertainty
            next_observations, rewards = self._mutate_transition(
                next_observations, rewards, variances
            )

            # sample policy action
            next_actions = self._sample_rollout_action(next_observations)

            # append new transitions
            new_transitions = []
            for i in range(len(init_transitions)):
                transition = Transition(
                    observation_shape=self._impl.observation_shape,
                    action_size=self._impl.action_size,
                    observation=observations[i],
                    action=actions[i],
                    reward=float(rewards[i][0]),
                    next_observation=next_observations[i],
                    terminal=0.0,
                )

                if prev_transitions:
                    prev_transitions[i].next_transition = transition
                    transition.prev_transition = prev_transitions[i]

                new_transitions.append(transition)

            prev_transitions = new_transitions
            rets += new_transitions
            observations = next_observations.copy()
            actions = next_actions.copy()

        return rets

    def _is_generating_new_data(self) -> bool:
        raise NotImplementedError

    def _sample_initial_transitions(
        self, transitions: List[Transition]
    ) -> List[Transition]:
        raise NotImplementedError

    def _sample_rollout_action(self, observations: np.ndarray) -> np.ndarray:
        assert self._impl, IMPL_NOT_INITIALIZED_ERROR
        return self._impl.sample_action(observations)

    def _get_rollout_horizon(self) -> int:
        raise NotImplementedError

    def _mutate_transition(
        self,
        observations: np.ndarray,
        rewards: np.ndarray,
        variances: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return observations, rewards
