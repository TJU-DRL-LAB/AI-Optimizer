from typing import Callable, List

import numpy as np

from ..dataset import Episode
from .scorer import WINDOW_SIZE, AlgoProtocol, _make_batches


def compare_continuous_action_diff(
    base_algo: AlgoProtocol,
) -> Callable[[AlgoProtocol, List[Episode]], float]:
    r"""Returns scorer function of action difference between algorithms.

    This metrics suggests how different the two algorithms are in continuous
    action-space.
    If the algorithm to compare with is near-optimal, the small action
    difference would be better.

    .. math::

        \mathbb{E}_{s_t \sim D}
            [(\pi_{\phi_1}(s_t) - \pi_{\phi_2}(s_t))^2]

    .. code-block:: python

        from tjuOfflineRL.algos import CQL
        from tjuOfflineRL.metrics.comparer import compare_continuous_action_diff

        cql1 = CQL()
        cql2 = CQL()

        scorer = compare_continuous_action_diff(cql1)

        squared_action_diff = scorer(cql2, ...)

    Args:
        base_algo: algorithm to comapre with.

    Returns:
        scorer function.

    """

    def scorer(algo: AlgoProtocol, episodes: List[Episode]) -> float:
        total_diffs = []
        for episode in episodes:
            # TODO: handle different n_frames
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                base_actions = base_algo.predict(batch.observations)
                actions = algo.predict(batch.observations)
                diff = ((actions - base_actions) ** 2).sum(axis=1).tolist()
                total_diffs += diff
        # smaller is better, sometimes?
        return -float(np.mean(total_diffs))

    return scorer


def compare_discrete_action_match(
    base_algo: AlgoProtocol,
) -> Callable[[AlgoProtocol, List[Episode]], float]:
    r"""Returns scorer function of action matches between algorithms.

    This metrics suggests how different the two algorithms are in discrete
    action-space.
    If the algorithm to compare with is near-optimal, the small action
    difference would be better.

    .. math::

        \mathbb{E}_{s_t \sim D} [\parallel
            \{\text{argmax}_a Q_{\theta_1}(s_t, a)
            = \text{argmax}_a Q_{\theta_2}(s_t, a)\}]

    .. code-block:: python

        from tjuOfflineRL.algos import DQN
        from tjuOfflineRL.metrics.comparer import compare_continuous_action_diff

        dqn1 = DQN()
        dqn2 = DQN()

        scorer = compare_continuous_action_diff(dqn1)

        percentage_of_identical_actions = scorer(dqn2, ...)

    Args:
        base_algo: algorithm to comapre with.

    Returns:
        scorer function.

    """

    def scorer(algo: AlgoProtocol, episodes: List[Episode]) -> float:
        total_matches = []
        for episode in episodes:
            # TODO: handle different n_frames
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                base_actions = base_algo.predict(batch.observations)
                actions = algo.predict(batch.observations)
                match = (base_actions == actions).tolist()
                total_matches += match
        return float(np.mean(total_matches))

    return scorer
