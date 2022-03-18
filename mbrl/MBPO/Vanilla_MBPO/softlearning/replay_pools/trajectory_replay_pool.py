from collections import deque
import gzip
import pickle
from itertools import islice

import numpy as np

from softlearning.utils.numpy import softmax
from .replay_pool import ReplayPool


def random_int_with_variable_range(mins, maxs):
    result = np.floor(np.random.uniform(mins, maxs)).astype(int)
    return result


class TrajectoryReplayPool(ReplayPool):
    def __init__(self,
                 observation_space,
                 action_space,
                 max_size):
        super(TrajectoryReplayPool, self).__init__()

        max_size = int(max_size)
        self._max_size = max_size

        self._trajectories = deque(maxlen=max_size)
        self._trajectory_lengths = deque(maxlen=max_size)
        self._num_samples = 0
        self._trajectories_since_save = 0

    @property
    def num_trajectories(self):
        return len(self._trajectories)

    @property
    def size(self):
        return sum(self._trajectory_lengths)

    @property
    def num_samples(self):
        return self._num_samples

    def add_paths(self, trajectories):
        self._trajectories += trajectories
        self._trajectory_lengths += [
            trajectory[next(iter(trajectory.keys()))].shape[0]
            for trajectory in trajectories
        ]
        self._trajectories_since_save += len(trajectories)

    def add_path(self, trajectory):
        self.add_paths([trajectory])

    def add_sample(self, sample):
        raise NotImplementedError(
            f"{self.__class__.__name__} only supports adding full paths at"
            " once.")

    def add_samples(self, samples):
        raise NotImplementedError(
            f"{self.__class__.__name__} only supports adding full paths at"
            " once.")

    def batch_by_indices(self,
                         episode_indices,
                         step_indices,
                         field_name_filter=None):
        assert len(episode_indices) == len(step_indices)

        batch_size = len(episode_indices)
        trajectories = [self._trajectories[i] for i in episode_indices]

        batch = {
            field_name: np.empty(
                (batch_size, *values.shape[1:]), dtype=values.dtype)
            for field_name, values in trajectories[0].items()
        }

        for i, episode in enumerate(trajectories):
            for field_name, episode_values in episode.items():
                batch[field_name][i] = episode_values[step_indices[i]]

        return batch

    def random_batch(self, batch_size, *args, **kwargs):
        num_trajectories = len(self._trajectories)
        if num_trajectories < 1:
            return {}

        trajectory_lengths = np.array(self._trajectory_lengths)
        trajectory_weights = trajectory_lengths / np.sum(trajectory_lengths)
        trajectory_probabilities = softmax(trajectory_weights)

        trajectory_indices = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=trajectory_probabilities)
        first_key = next(iter(
            self._trajectories[trajectory_indices[0]].keys()))
        trajectory_lengths = np.array([
            self._trajectories[trajectory_index][first_key].shape[0]
            for trajectory_index in trajectory_indices
        ])

        step_indices = random_int_with_variable_range(
            np.zeros_like(trajectory_lengths, dtype=np.int64),
            trajectory_lengths)

        batch = self.batch_by_indices(trajectory_indices, step_indices)

        return batch

    def last_n_batch(self, last_n, field_name_filter=None, **kwargs):
        num_trajectories = len(self._trajectories)
        if num_trajectories < 1:
            return {}

        trajectory_indices = []
        step_indices = []

        trajectory_lengths = 0
        for trajectory_index in range(num_trajectories-1, -1, -1):
            trajectory = self._trajectories[trajectory_index]
            trajectory_length = trajectory[list(trajectory.keys())[0]].shape[0]

            steps_from_this_episode = min(trajectory_length, last_n - trajectory_lengths)
            step_indices += list(range(
                trajectory_length-1,
                trajectory_length - steps_from_this_episode - 1,
                -1))
            trajectory_indices += [trajectory_index] * steps_from_this_episode

            trajectory_lengths += trajectory_length

            if trajectory_lengths >= last_n:
                break

        trajectory_indices = trajectory_indices[::-1]
        step_indices = step_indices[::-1]

        batch = self.batch_by_indices(trajectory_indices, step_indices)

        return batch

    def save_latest_experience(self, pickle_path):
        # deque doesn't support direct slicing, thus need to use islice
        num_trajectories = self.num_trajectories
        start_index = max(num_trajectories - self._trajectories_since_save, 0)
        end_index = num_trajectories

        latest_trajectories = tuple(islice(
            self._trajectories, start_index, end_index))

        with gzip.open(pickle_path, 'wb') as f:
            pickle.dump(latest_trajectories, f)

        self._trajectories_since_save = 0

    def load_experience(self, experience_path):
        with gzip.open(experience_path, 'rb') as f:
            latest_trajectories = pickle.load(f)

        self.add_paths(latest_trajectories)
        self._trajectories_since_save = 0
