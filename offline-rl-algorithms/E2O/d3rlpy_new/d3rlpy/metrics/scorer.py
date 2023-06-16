from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast

import gym
import numpy as np
from typing_extensions import Protocol
import re
import torch
import csv

from d3rlpy.dataset import Episode, TransitionMiniBatch
from ..preprocessing.reward_scalers import RewardScaler
from ..preprocessing.stack import StackedObservation

WINDOW_SIZE = 1024


class AlgoProtocol(Protocol):
    def predict(self, x: Union[np.ndarray, List[Any]]) -> np.ndarray:
        ...

    def predict_value(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_std: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        ...

    @property
    def n_frames(self) -> int:
        ...

    @property
    def gamma(self) -> float:
        ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        ...


class DynamicsProtocol(Protocol):
    def predict(
        self,
        x: Union[np.ndarray, List[Any]],
        action: Union[np.ndarray, List[Any]],
        with_variance: bool = False,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        ...

    @property
    def n_frames(self) -> int:
        ...

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        ...


def _make_batches(
    episode: Episode, window_size: int, n_frames: int
) -> Iterator[TransitionMiniBatch]:
    n_batches = len(episode) // window_size
    if len(episode) % window_size != 0:
        n_batches += 1
    for i in range(n_batches):
        head_index = i * window_size
        last_index = min(head_index + window_size, len(episode))
        transitions = episode.transitions[head_index:last_index]
        batch = TransitionMiniBatch(transitions, n_frames)
        yield batch


def td_error_scorer(algo: AlgoProtocol, episodes: List[Episode]) -> float:
    r"""Returns average TD error.

    This metics suggests how Q functions overfit to training sets.
    If the TD error is large, the Q functions are overfitting.

    .. math::

        \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D}
            [(Q_\theta (s_t, a_t)
             - r_{t+1} - \gamma \max_a Q_\theta (s_{t+1}, a))^2]

    Args:
        algo: algorithm.
        episodes: list of episodes.

    Returns:
        average TD error.

    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            # estimate values for current observations
            values = algo.predict_value(batch.observations, batch.actions)

            # estimate values for next observations
            next_actions = algo.predict(batch.next_observations)
            next_values = algo.predict_value(
                batch.next_observations, next_actions
            )

            # calculate td errors
            mask = (1.0 - np.asarray(batch.terminals)).reshape(-1)
            rewards = np.asarray(batch.rewards).reshape(-1)
            if algo.reward_scaler:
                rewards = algo.reward_scaler.transform_numpy(rewards)
            y = rewards + algo.gamma * cast(np.ndarray, next_values) * mask
            total_errors += ((values - y) ** 2).tolist()

    return float(np.mean(total_errors))


def discounted_sum_of_advantage_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns average of discounted sum of advantage.

    This metrics suggests how the greedy-policy selects different actions in
    action-value space.
    If the sum of advantage is small, the policy selects actions with larger
    estimated action-values.

    .. math::

        \mathbb{E}_{s_t, a_t \sim D}
            [\sum_{t' = t} \gamma^{t' - t} A(s_{t'}, a_{t'})]

    where :math:`A(s_t, a_t) = Q_\theta (s_t, a_t)
    - \mathbb{E}_{a \sim \pi} [Q_\theta (s_t, a)]`.

    References:
        * `Murphy., A generalization error for Q-Learning.
          <http://www.jmlr.org/papers/volume6/murphy05a/murphy05a.pdf>`_

    Args:
        algo: algorithm.
        episodes: list of episodes.

    Returns:
        average of discounted sum of advantage.

    """
    total_sums = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            # estimate values for dataset actions
            dataset_values = algo.predict_value(
                batch.observations, batch.actions
            )
            dataset_values = cast(np.ndarray, dataset_values)

            # estimate values for the current policy
            actions = algo.predict(batch.observations)
            on_policy_values = algo.predict_value(batch.observations, actions)

            # calculate advantages
            advantages = (dataset_values - on_policy_values).tolist()

            # calculate discounted sum of advantages
            A = advantages[-1]
            sum_advantages = [A]
            for advantage in reversed(advantages[:-1]):
                A = advantage + algo.gamma * A
                sum_advantages.append(A)

            total_sums += sum_advantages

    # smaller is better
    return float(np.mean(total_sums))


def average_value_estimation_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns average value estimation.

    This metrics suggests the scale for estimation of Q functions.
    If average value estimation is too large, the Q functions overestimate
    action-values, which possibly makes training failed.

    .. math::

        \mathbb{E}_{s_t \sim D} [ \max_a Q_\theta (s_t, a)]

    Args:
        algo: algorithm.
        episodes: list of episodes.

    Returns:
        average value estimation.

    """
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions = algo.predict(batch.observations)
            values = algo.predict_value(batch.observations, actions)
            total_values += cast(np.ndarray, values).tolist()
    return float(np.mean(total_values))


def value_estimation_std_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns standard deviation of value estimation.

    This metrics suggests how confident Q functions are for the given
    episodes.
    This metrics will be more accurate with `boostrap` enabled and the larger
    `n_critics` at algorithm.
    If standard deviation of value estimation is large, the Q functions are
    overfitting to the training set.

    .. math::

        \mathbb{E}_{s_t \sim D, a \sim \text{argmax}_a Q_\theta(s_t, a)}
            [Q_{\text{std}}(s_t, a)]

    where :math:`Q_{\text{std}}(s, a)` is a standard deviation of action-value
    estimation over ensemble functions.

    Args:
        algo: algorithm.
        episodes: list of episodes.

    Returns:
        standard deviation.

    """
    total_stds = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions = algo.predict(batch.observations)
            _, stds = algo.predict_value(batch.observations, actions, True)
            total_stds += stds.tolist()
    return float(np.mean(total_stds))


def initial_state_value_estimation_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns mean estimated action-values at the initial states.

    This metrics suggests how much return the trained policy would get from
    the initial states by deploying the policy to the states.
    If the estimated value is large, the trained policy is expected to get
    higher returns.

    .. math::

        \mathbb{E}_{s_0 \sim D} [Q(s_0, \pi(s_0))]

    References:
        * `Paine et al., Hyperparameter Selection for Offline Reinforcement
          Learning <https://arxiv.org/abs/2007.09055>`_

    Args:
        algo: algorithm.
        episodes: list of episodes.

    Returns:
        mean action-value estimation at the initial states.

    """
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            # estimate action-value in initial states
            actions = algo.predict([batch.observations[0]])
            values = algo.predict_value([batch.observations[0]], actions)
            total_values.append(values[0])
    return float(np.mean(total_values))


def soft_opc_scorer(
    return_threshold: float,
) -> Callable[[AlgoProtocol, List[Episode]], float]:
    r"""Returns Soft Off-Policy Classification metrics.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer funciton is evaluating gaps of action-value
    estimation between the success episodes and the all episodes.
    If the learned Q-function is optimal, action-values in success episodes
    are expected to be higher than the others.
    The success episode is defined as an episode with a return above the given
    threshold.

    .. math::

        \mathbb{E}_{s, a \sim D_{success}} [Q(s, a)]
            - \mathbb{E}_{s, a \sim D} [Q(s, a)]

    .. code-block:: python

        from d3rlpy.datasets import get_cartpole
        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import soft_opc_scorer
        from sklearn.model_selection import train_test_split

        dataset, _ = get_cartpole()
        train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

        scorer = soft_opc_scorer(return_threshold=180)

        dqn = DQN()
        dqn.fit(train_episodes,
                eval_episodes=test_episodes,
                scorers={'soft_opc': scorer})

    References:
        * `Irpan et al., Off-Policy Evaluation via Off-Policy Classification.
          <https://arxiv.org/abs/1906.01624>`_

    Args:
        return_threshold: threshold of success episodes.

    Returns:
        scorer function.

    """

    def scorer(algo: AlgoProtocol, episodes: List[Episode]) -> float:
        success_values = []
        all_values = []
        for episode in episodes:
            is_success = episode.compute_return() >= return_threshold
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                values = algo.predict_value(batch.observations, batch.actions)
                values = cast(np.ndarray, values)
                all_values += values.reshape(-1).tolist()
                if is_success:
                    success_values += values.reshape(-1).tolist()
        return float(np.mean(success_values) - np.mean(all_values))

    return scorer


def continuous_action_diff_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns squared difference of actions between algorithm and dataset.

    This metrics suggests how different the greedy-policy is from the given
    episodes in continuous action-space.
    If the given episodes are near-optimal, the small action difference would
    be better.

    .. math::

        \mathbb{E}_{s_t, a_t \sim D} [(a_t - \pi_\phi (s_t))^2]

    Args:
        algo: algorithm.
        episodes: list of episodes.

    Returns:
        squared action difference.

    """
    total_diffs = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions = algo.predict(batch.observations)
            diff = ((batch.actions - actions) ** 2).sum(axis=1).tolist()
            total_diffs += diff
    return float(np.mean(total_diffs))


def discrete_action_match_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns percentage of identical actions between algorithm and dataset.

    This metrics suggests how different the greedy-policy is from the given
    episodes in discrete action-space.
    If the given episdoes are near-optimal, the large percentage would be
    better.

    .. math::

        \frac{1}{N} \sum^N \parallel
            \{a_t = \text{argmax}_a Q_\theta (s_t, a)\}

    Args:
        algo: algorithm.
        episodes: list of episodes.

    Returns:
        percentage of identical actions.

    """
    total_matches = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            actions = algo.predict(batch.observations)
            match = (batch.actions.reshape(-1) == actions).tolist()
            total_matches += match
    return float(np.mean(total_matches))


def evaluate_on_environment(
    env: gym.Env, n_trials: int = 10, epsilon: float = 0.0, render: bool = False
) -> Callable[..., float]:
    """Returns scorer function of evaluation on environment.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.

    .. code-block:: python

        import gym

        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment


        env = gym.make('CartPole-v0')

        scorer = evaluate_on_environment(env)

        cql = CQL()

        mean_episode_return = scorer(cql)


    Args:
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.
        render: flag to render environment.

    Returns:
        scoerer function.


    """

    # for image observation
    if re.match(r"Widow250", env.spec.id):
        observation_shape = env.observation_space.spaces['state'].shape
        is_image = observation_shape[0] == 3
    else:
        observation_shape = env.observation_space.shape
        is_image = len(observation_shape) == 3

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        if is_image:
            stacked_observation = StackedObservation(
                observation_shape, algo.n_frames
            )

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0

            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation['state'])

            if re.match(r"Widow250PickPlace", env.spec.id):
                episode_length = 40
            elif re.match(r"Widow250Drawer", env.spec.id):
                episode_length = 80
            else:
                episode_length = 1000

            for _ in range(episode_length):
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if is_image:
                        action = algo.predict([stacked_observation.eval()])[0]
                    else:
                        if re.match(r"Widow250", env.spec.id):
                            action = algo.predict([observation['state']])[0]
                        else:
                            action = algo.predict([observation])[0]

                observation, reward, done, _ = env.step(action)
                episode_reward += reward

                if is_image:
                    stacked_observation.append(observation)

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(episode_reward)

        return float(np.mean(episode_rewards))

    return scorer


def dynamics_observation_prediction_error_scorer(
    dynamics: DynamicsProtocol, episodes: List[Episode]
) -> float:
    r"""Returns MSE of observation prediction.

    This metrics suggests how dynamics model is generalized to test sets.
    If the MSE is large, the dynamics model are overfitting.

    .. math::

        \mathbb{E}_{s_t, a_t, s_{t+1} \sim D} [(s_{t+1} - s')^2]

    where :math:`s' \sim T(s_t, a_t)`.

    Args:
        dynamics: dynamics model.
        episodes: list of episodes.

    Returns:
        mean squared error.

    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, dynamics.n_frames):
            pred = dynamics.predict(batch.observations, batch.actions)
            errors = ((batch.next_observations - pred[0]) ** 2).sum(axis=1)
            total_errors += errors.tolist()
    return float(np.mean(total_errors))


def dynamics_reward_prediction_error_scorer(
    dynamics: DynamicsProtocol, episodes: List[Episode]
) -> float:
    r"""Returns MSE of reward prediction.

    This metrics suggests how dynamics model is generalized to test sets.
    If the MSE is large, the dynamics model are overfitting.

    .. math::

        \mathbb{E}_{s_t, a_t, r_{t+1} \sim D} [(r_{t+1} - r')^2]

    where :math:`r' \sim T(s_t, a_t)`.

    Args:
        dynamics: dynamics model.
        episodes: list of episodes.

    Returns:
        mean squared error.

    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, dynamics.n_frames):
            pred = dynamics.predict(batch.observations, batch.actions)
            rewards = batch.rewards
            if dynamics.reward_scaler:
                rewards = dynamics.reward_scaler.transform_numpy(rewards)
            errors = ((rewards - pred[1]) ** 2).reshape(-1)
            total_errors += errors.tolist()
    return float(np.mean(total_errors))


def dynamics_prediction_variance_scorer(
    dynamics: DynamicsProtocol, episodes: List[Episode]
) -> float:
    """Returns prediction variance of ensemble dynamics.

    This metrics suggests how dynamics model is confident of test sets.
    If the variance is large, the dynamics model has large uncertainty.

    Args:
        dynamics: dynamics model.
        episodes: list of episodes.

    Returns:
        variance.

    """
    total_variances = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, dynamics.n_frames):
            pred = dynamics.predict(batch.observations, batch.actions, True)
            pred = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], pred)
            total_variances += pred[2].tolist()
    return float(np.mean(total_variances))
