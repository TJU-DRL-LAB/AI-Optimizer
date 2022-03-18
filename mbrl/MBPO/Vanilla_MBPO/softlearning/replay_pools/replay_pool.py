import abc


class ReplayPool(object):
    """A class used to save and replay data."""

    @abc.abstractmethod
    def add_sample(self, sample):
        """Add a transition tuple."""
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """Clean up pool after episode termination."""
        pass

    @property
    @abc.abstractmethod
    def size(self, **kwargs):
        pass

    def add_path(self, path):
        """Add a rollout to the replay pool.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by railrl.samplers.util.rollout
        """
        self.add_samples(path)
        self.terminate_episode()

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """Return a random batch of size `batch_size`."""
        pass
