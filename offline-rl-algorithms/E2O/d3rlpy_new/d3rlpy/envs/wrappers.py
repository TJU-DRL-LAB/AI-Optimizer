import json
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gym
import numpy as np

try:
    import cv2  # this is used in AtariPreprocessing
except ImportError:
    cv2 = None

from gym.spaces import Box
from gym.wrappers import TransformReward


class ChannelFirst(gym.Wrapper):  # type: ignore
    """Channel-first wrapper for image observation environments.

    d3rlpy expects channel-first images since it's built with PyTorch.
    You can transform the observation shape with ``ChannelFirst`` wrapper.

    Args:
        env (gym.Env): gym environment.

    """

    observation_space: Box

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = self.observation_space.shape
        low = self.observation_space.low
        high = self.observation_space.high
        dtype = self.observation_space.dtype

        if len(shape) == 3:
            self.observation_space = Box(
                low=np.transpose(low, [2, 0, 1]),
                high=np.transpose(high, [2, 0, 1]),
                shape=(shape[2], shape[0], shape[1]),
                dtype=dtype,
            )
        elif len(shape) == 2:
            self.observation_space = Box(
                low=np.reshape(low, (1, *shape)),
                high=np.reshape(high, (1, *shape)),
                shape=(1, *shape),
                dtype=dtype,
            )
        else:
            raise ValueError("image observation is only allowed.")

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        observation, reward, terminal, info = self.env.step(action)
        # make channel first observation
        if observation.ndim == 3:
            observation_T = np.transpose(observation, [2, 0, 1])
        else:
            observation_T = np.reshape(observation, (1, *observation.shape))
        assert observation_T.shape == self.observation_space.shape
        return observation_T, reward, terminal, info

    def reset(self, **kwargs: Any) -> np.ndarray:
        observation = self.env.reset(**kwargs)
        # make channel first observation
        if observation.ndim == 3:
            observation_T = np.transpose(observation, [2, 0, 1])
        else:
            observation_T = np.reshape(observation, (1, *observation.shape))
        assert observation_T.shape == self.observation_space.shape
        return observation_T


# https://github.com/openai/gym/blob/0.17.3/gym/wrappers/atari_preprocessing.py
class AtariPreprocessing(gym.Wrapper):  # type: ignore
    r"""Atari 2600 preprocessings.
    This class follows the guidelines in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".
    Specifically:

    * NoopReset: obtain initial state by taking random number of no-ops on
        reset.
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not
        recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: optional
    * Scale observation: optional

    Args:
        env (Env): environment
        noop_max (int): max number of no-ops
        frame_skip (int): the frequency at which the agent experiences the game.
        screen_size (int): resize Atari frame
        terminal_on_life_loss (bool): if True, then step() returns done=True
            whenever a life is lost.
        grayscale_obs (bool): if True, then gray scale observation is returned,
            otherwise, RGB observation is returned.
        grayscale_newaxis (bool): if True and grayscale_obs=True, then a
            channel axis is added to grayscale observations to make them
            3-dimensional.
        scale_obs (bool): if True, then observation normalized in range [0,1]
            is returned. It also limits memory optimization benefits of
            FrameStack Wrapper.

    """

    def __init__(
        self,
        env: gym.Env,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = 84,
        terminal_on_life_loss: bool = False,
        grayscale_obs: bool = True,
        grayscale_newaxis: bool = False,
        scale_obs: bool = False,
    ):
        super().__init__(env)
        assert cv2 is not None, (
            "opencv-python package not installed! Try"
            " running pip install gym[atari] to get dependencies for atari"
        )
        assert frame_skip > 0
        assert screen_size > 0
        assert noop_max >= 0
        if frame_skip > 1:
            assert "NoFrameskip" in env.spec.id, (
                "disable frame-skipping in"
                " the original env. for more than one frame-skip as it will"
                " be done by the wrapper"
            )
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.terminal_on_life_loss = terminal_on_life_loss
        self.grayscale_obs = grayscale_obs
        self.grayscale_newaxis = grayscale_newaxis
        self.scale_obs = scale_obs

        # buffer of most recent two observations for max pooling
        if grayscale_obs:
            self.obs_buffer = [
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                np.empty(env.observation_space.shape[:2], dtype=np.uint8),
            ]
        else:
            self.obs_buffer = [
                np.empty(env.observation_space.shape, dtype=np.uint8),
                np.empty(env.observation_space.shape, dtype=np.uint8),
            ]

        self.ale = env.unwrapped.ale
        self.lives = 0
        self.game_over = True

        _low, _high, _obs_dtype = (
            (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        )
        _shape = (screen_size, screen_size, 1 if grayscale_obs else 3)
        if grayscale_obs and not grayscale_newaxis:
            _shape = _shape[:-1]  # type: ignore
        self.observation_space = Box(
            low=_low, high=_high, shape=_shape, dtype=_obs_dtype
        )

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        R = 0.0

        for t in range(self.frame_skip):
            _, reward, done, info = self.env.step(action)
            R += reward
            self.game_over = done

            if self.terminal_on_life_loss:
                new_lives = self.ale.lives()
                done = done or new_lives < self.lives
                self.lives = new_lives

            if done:
                break
            if t == self.frame_skip - 2:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[1])
                else:
                    self.ale.getScreenRGB2(self.obs_buffer[1])
            elif t == self.frame_skip - 1:
                if self.grayscale_obs:
                    self.ale.getScreenGrayscale(self.obs_buffer[0])
                else:
                    self.ale.getScreenRGB2(self.obs_buffer[0])

        return self._get_obs(), R, done, info

    def reset(self, **kwargs: Any) -> np.ndarray:
        # this condition is not included in the original code
        if self.game_over:
            self.env.reset(**kwargs)
        else:
            # NoopReset
            self.env.step(0)

        noops = (
            self.env.unwrapped.np_random.randint(1, self.noop_max + 1)
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset(**kwargs)

        self.lives = self.ale.lives()
        if self.grayscale_obs:
            self.ale.getScreenGrayscale(self.obs_buffer[0])
        else:
            self.ale.getScreenRGB2(self.obs_buffer[0])
        self.obs_buffer[1].fill(0)
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        if self.frame_skip > 1:  # more efficient in-place pooling
            np.maximum(
                self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0]
            )
        obs = cv2.resize(
            self.obs_buffer[0],
            (self.screen_size, self.screen_size),
            interpolation=cv2.INTER_AREA,
        )

        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        if self.grayscale_obs and self.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=-1)  # Add a channel axis
        return obs


class Atari(gym.Wrapper):  # type: ignore
    """Atari 2600 wrapper for experiments.

    Args:
        env (gym.Env): gym environment.
        is_eval (bool): flag to enter evaluation mode.

    """

    def __init__(self, env: gym.Env, is_eval: bool = False):
        env = AtariPreprocessing(env, terminal_on_life_loss=not is_eval)
        if not is_eval:
            env = TransformReward(env, lambda r: np.clip(r, -1.0, 1.0))
        super().__init__(ChannelFirst(env))


class Monitor(gym.Wrapper):  # type: ignore
    """gym.wrappers.Monitor-style Monitor wrapper.

    Args:
        env (gym.Env): gym environment.
        directory (str): directory to save.
        video_callable (callable): callable function that takes episode counter
            to control record frequency.
        force (bool): flag to allow existing directory.
        frame_rate (float): video frame rate.
        record_rate (int): images are record every ``record_rate`` frames.

    """

    _directory: str
    _video_callable: Callable[[int], bool]
    _frame_rate: float
    _record_rate: int
    _episode: int
    _episode_return: float
    _episode_step: int
    _buffer: np.ndarray

    def __init__(
        self,
        env: gym.Env,
        directory: str,
        video_callable: Optional[Callable[[int], bool]] = None,
        force: bool = False,
        frame_rate: float = 30.0,
        record_rate: int = 1,
    ):
        super().__init__(env)
        # prepare directory
        if os.path.exists(directory) and not force:
            raise ValueError(f"{directory} already exists.")
        os.makedirs(directory, exist_ok=True)
        self._directory = directory

        if video_callable:
            self._video_callable = video_callable  # type: ignore
        else:
            self._video_callable = lambda ep: ep % 10 == 0  # type: ignore

        self._frame_rate = frame_rate
        self._record_rate = record_rate

        self._episode = 0
        self._episode_return = 0.0
        self._episode_step = 0
        self._buffer = []

    def step(
        self, action: Union[np.ndarray, int]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = super().step(action)

        if self._video_callable(self._episode):  # type: ignore
            # store rendering
            frame = cv2.cvtColor(super().render("rgb_array"), cv2.COLOR_BGR2RGB)
            self._buffer.append(frame)
            self._episode_step += 1
            self._episode_return += reward
            if done:
                self._save_video()
                self._save_stats()

        return obs, reward, done, info

    def reset(self, **kwargs: Any) -> np.ndarray:
        self._episode += 1
        self._episode_return = 0.0
        self._episode_step = 0
        self._buffer = []
        return super().reset(**kwargs)

    def _save_video(self) -> None:
        height, width = self._buffer[0].shape[:2]
        path = os.path.join(self._directory, f"video{self._episode}.avi")
        fmt = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(path, fmt, self._frame_rate, (width, height))
        print(f"Saving a recorded video to {path}...")
        for i, frame in enumerate(self._buffer):
            if i % self._record_rate == 0:
                writer.write(frame)
        writer.release()

    def _save_stats(self) -> None:
        path = os.path.join(self._directory, f"stats{self._episode}.json")
        stats = {
            "episode_step": self._episode_step,
            "return": self._episode_return,
        }
        with open(path, "w") as f:
            json_str = json.dumps(stats, indent=2)
            f.write(json_str)
