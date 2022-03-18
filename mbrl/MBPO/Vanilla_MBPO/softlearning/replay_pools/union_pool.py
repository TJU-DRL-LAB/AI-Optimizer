import numpy as np

from .replay_pool import ReplayPool


class UnionPool(ReplayPool):
    def __init__(self, pools):
        pool_sizes = np.array([b.size for b in pools])
        self._total_size = sum(pool_sizes)
        self._normalized_pool_sizes = pool_sizes / self._total_size

        self.pools = pools

    def add_sample(self, *args, **kwargs):
        raise NotImplementedError

    def terminate_episode(self):
        raise NotImplementedError

    @property
    def size(self):
        return self._total_size

    def add_path(self, **kwargs):
        raise NotImplementedError

    def random_batch(self, batch_size):

        # TODO: Hack
        partial_batch_sizes = self._normalized_pool_sizes * batch_size
        partial_batch_sizes = partial_batch_sizes.astype(int)
        partial_batch_sizes[0] = batch_size - sum(partial_batch_sizes[1:])

        partial_batches = [
            pool.random_batch(partial_batch_size) for pool,
            partial_batch_size in zip(self.pools, partial_batch_sizes)
        ]

        def all_values(key):
            return [partial_batch[key] for partial_batch in partial_batches]

        keys = partial_batches[0].keys()

        return {key: np.concatenate(all_values(key), axis=0) for key in keys}
