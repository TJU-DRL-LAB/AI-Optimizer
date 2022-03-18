import gzip
import pickle

import numpy as np

from .replay_pool import ReplayPool


class FlexibleReplayPool(ReplayPool):
    def __init__(self, max_size, fields_attrs):
        super(FlexibleReplayPool, self).__init__()

        max_size = int(max_size)
        self._max_size = max_size

        self.fields = {}
        self.fields_attrs = {}

        self.add_fields(fields_attrs)

        self._pointer = 0
        self._size = 0
        self._samples_since_save = 0

    @property
    def size(self):
        return self._size

    @property
    def field_names(self):
        return list(self.fields.keys())

    def add_fields(self, fields_attrs):
        self.fields_attrs.update(fields_attrs)

        for field_name, field_attrs in fields_attrs.items():
            field_shape = (self._max_size, *field_attrs['shape'])
            initializer = field_attrs.get('initializer', np.zeros)
            self.fields[field_name] = initializer(
                field_shape, dtype=field_attrs['dtype'])

    def _advance(self, count=1):
        self._pointer = (self._pointer + count) % self._max_size
        self._size = min(self._size + count, self._max_size)
        self._samples_since_save += count

    def add_sample(self, sample):
        samples = {
            key: value[None, ...]
            for key, value in sample.items()
        }
        self.add_samples(samples)

    def add_samples(self, samples):
        field_names = list(samples.keys())
        num_samples = samples[field_names[0]].shape[0]

        index = np.arange(
            self._pointer, self._pointer + num_samples) % self._max_size

        for field_name in self.field_names:
            default_value = (
                self.fields_attrs[field_name].get('default_value', 0.0))
            values = samples.get(field_name, default_value)
            assert values.shape[0] == num_samples
            self.fields[field_name][index] = values

        self._advance(num_samples)

    def random_indices(self, batch_size):
        if self._size == 0: return np.arange(0, 0)
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size, field_name_filter=None, **kwargs):
        random_indices = self.random_indices(batch_size)
        return self.batch_by_indices(
            random_indices, field_name_filter=field_name_filter, **kwargs)

    def last_n_batch(self, last_n, field_name_filter=None, **kwargs):
        last_n_indices = np.arange(
            self._pointer - min(self.size, last_n), self._pointer
        ) % self._max_size
        return self.batch_by_indices(
            last_n_indices, field_name_filter=field_name_filter, **kwargs)

    def filter_fields(self, field_names, field_name_filter):
        if isinstance(field_name_filter, str):
            field_name_filter = [field_name_filter]

        if isinstance(field_name_filter, (list, tuple)):
            field_name_list = field_name_filter

            def filter_fn(field_name):
                return field_name in field_name_list

        else:
            filter_fn = field_name_filter

        filtered_field_names = [
            field_name for field_name in field_names
            if filter_fn(field_name)
        ]

        return filtered_field_names

    def batch_by_indices(self, indices, field_name_filter=None):
        if np.any(indices % self._max_size > self.size):
            raise ValueError(
                "Tried to retrieve batch with indices greater than current"
                " size")

        field_names = self.field_names
        if field_name_filter is not None:
            field_names = self.filter_fields(
                field_names, field_name_filter)

        return {
            field_name: self.fields[field_name][indices]
            for field_name in field_names
        }

    def save_latest_experience(self, pickle_path):
        latest_samples = self.last_n_batch(self._samples_since_save)

        with gzip.open(pickle_path, 'wb') as f:
            pickle.dump(latest_samples, f)

        self._samples_since_save = 0

    def load_experience(self, experience_path):
        with gzip.open(experience_path, 'rb') as f:
            latest_samples = pickle.load(f)

        key = list(latest_samples.keys())[0]
        num_samples = latest_samples[key].shape[0]
        for field_name, data in latest_samples.items():
            assert data.shape[0] == num_samples, data.shape

        self.add_samples(latest_samples)
        self._samples_since_save = 0

    def return_all_samples(self):
        return {
            field_name: self.fields[field_name][:self.size]
            for field_name in self.field_names
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state['fields'] = {
            field_name: self.fields[field_name][:self.size]
            for field_name in self.field_names
        }

        return state

    def __setstate__(self, state):
        if state['_size'] < state['_max_size']:
            pad_size = state['_max_size'] - state['_size']
            for field_name in state['fields'].keys():
                field_shape = state['fields_attrs'][field_name]['shape']
                state['fields'][field_name] = np.concatenate((
                    state['fields'][field_name],
                    np.zeros((pad_size, *field_shape))
                ), axis=0)

        self.__dict__ = state
