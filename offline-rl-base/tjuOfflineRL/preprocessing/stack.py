from typing import Sequence

import numpy as np


class StackedObservation:
    """StackedObservation class.

    This class is used to stack images to handle temporal features.

    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_

    Args:
        observation_shape (tuple): image observation shape.
        n_frames (int): the number of frames to stack.
        dtype (int): numpy data type.

    """

    _image_channels: int
    _n_frames: int
    _dtype: np.dtype
    _stack: np.ndarray

    def __init__(
        self,
        observation_shape: Sequence[int],
        n_frames: int,
        dtype: np.dtype = np.uint8,
    ):
        self._image_channels = observation_shape[0]
        image_size = observation_shape[1:]
        self._n_frames = n_frames
        self._dtype = dtype
        stacked_shape = (self._image_channels * n_frames, *image_size)
        self._stack = np.zeros(stacked_shape, dtype=self._dtype)

    def append(self, image: np.ndarray) -> np.ndarray:
        """Stack new image.

        Args:
            image (numpy.ndarray): image observation.

        """
        assert image.dtype == self._dtype
        self._stack = np.roll(self._stack, -self._image_channels, axis=0)
        head_channel = self._image_channels * (self._n_frames - 1)
        self._stack[head_channel:] = image.copy()

    def eval(self) -> np.ndarray:
        """Returns stacked observation.

        Returns:
            numpy.ndarray: stacked observation.

        """
        return self._stack

    def clear(self) -> None:
        """Clear stacked observation by filling 0."""
        self._stack.fill(0)


class BatchStackedObservation:
    """Batch version of StackedObservation class.

    This class is used to stack images to handle temporal features.

    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_

    Args:
        observation_shape (tuple): image observation shape.
        n_frames (int): the number of frames to stack.
        dtype (int): numpy data type.

    """

    _image_channels: int
    _n_frames: int
    _n_envs: int
    _dtype: np.dtype
    _stack: np.ndarray

    def __init__(
        self,
        observation_shape: Sequence[int],
        n_frames: int,
        n_envs: int,
        dtype: np.dtype = np.uint8,
    ):
        self._image_channels = observation_shape[0]
        image_size = observation_shape[1:]
        self._n_frames = n_frames
        self._n_envs = n_envs
        self._dtype = dtype
        stacked_shape = (n_envs, self._image_channels * n_frames, *image_size)
        self._stack = np.zeros(stacked_shape, dtype=self._dtype)

    def append(self, image: np.ndarray) -> np.ndarray:
        """Stack new image.

        Args:
            image (numpy.ndarray): image observation.

        """
        assert image.dtype == self._dtype
        self._stack = np.roll(self._stack, -self._image_channels, axis=1)
        head_channel = self._image_channels * (self._n_frames - 1)
        self._stack[:, head_channel:] = image.copy()

    def eval(self) -> np.ndarray:
        """Returns stacked observation.

        Returns:
            numpy.ndarray: stacked observation.

        """
        return self._stack

    def clear(self) -> None:
        """Clear stacked observation by filling 0."""
        self._stack.fill(0)

    def clear_by_index(self, index: int) -> None:
        """Clear stacked observation in the specific index by filling 0."""
        self._stack[index].fill(0)
