# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environment wrappers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import datetime
import io
import os
import sys
import traceback
import uuid

import gym
import gym.spaces
import numpy as np
import skimage.transform
import tensorflow as tf

from planet.tools import nested


class ObservationDict(object):

  def __init__(self, env, key='observ'):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = {self._key: self._env.observation_space}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = {self._key: np.array(obs)}
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = {self._key: np.array(obs)}
    return obs


class ConcatObservation(object):
  """Select observations from a dict space and concatenate them."""

  def __init__(self, env, keys):
    self._env = env
    self._keys = keys

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    spaces = [spaces[key] for key in self._keys]
    low = np.concatenate([space.low for space in spaces], 0)
    high = np.concatenate([space.high for space in spaces], 0)
    dtypes = [space.dtype for space in spaces]
    if not all(dtype == dtypes[0] for dtype in dtypes):
      message = 'Spaces must have the same data type; are {}.'
      raise KeyError(message.format(', '.join(str(x) for x in dtypes)))
    return gym.spaces.Box(low, high, dtype=dtypes[0])

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = self._select_keys(obs)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = self._select_keys(obs)
    return obs

  def _select_keys(self, obs):
    return np.concatenate([obs[key] for key in self._keys], 0)


class SelectObservations(object):

  def __init__(self, env, keys):
    self._env = env
    self._keys = keys

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    return gym.spaces.Dict({key: spaces[key] for key in self._keys})

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action, *args, **kwargs):
    obs, reward, done, info = self._env.step(action, *args, **kwargs)
    obs = {key: obs[key] for key in self._keys}
    return obs, reward, done, info

  def reset(self, *args, **kwargs):
    obs = self._env.reset(*args, **kwargs)
    obs = {key: obs[key] for key in self._keys}
    return obs


class PixelObservations(object):

  def __init__(self, env, size=(64, 64), dtype=np.uint8, key='image'):
    assert isinstance(env.observation_space, gym.spaces.Dict)
    self._env = env
    self._size = size
    self._dtype = dtype
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    high = {np.uint8: 255, np.float: 1.0}[self._dtype]
    image = gym.spaces.Box(0, high, self._size + (3,), dtype=self._dtype)
    spaces = self._env.observation_space.spaces.copy()
    assert self._key not in spaces
    spaces[self._key] = image
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    return self._env.action_space

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs[self._key] = self._render_image()
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs[self._key] = self._render_image()
    return obs

  def _render_image(self):
    image = self._env.render('rgb_array')
    if image.shape[:2] != self._size:
      kwargs = dict(
          output_shape=self._size, mode='edge', order=1, preserve_range=True)
      image = skimage.transform.resize(image, **kwargs).astype(image.dtype)
    if self._dtype and image.dtype != self._dtype:
      if image.dtype in (np.float32, np.float64) and self._dtype == np.uint8:
        image = (image * 255).astype(self._dtype)
      elif image.dtype == np.uint8 and self._dtype in (np.float32, np.float64):
        image = image.astype(self._dtype) / 255
      else:
        message = 'Cannot convert observations from {} to {}.'
        raise NotImplementedError(message.format(image.dtype, self._dtype))
    return image


class ObservationToRender(object):

  def __init__(self, env, key='image'):
    self._env = env
    self._key = key
    self._image = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    return gym.spaces.Dict({})

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    self._image = obs.pop(self._key)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    self._image = obs.pop(self._key)
    return obs

  def render(self, *args, **kwargs):
    return self._image


class OverwriteRender(object):

  def __init__(self, env, render_fn):
    self._env = env
    self._render_fn = render_fn
    self._env.render('rgb_array')  # Set up viewer.

  def __getattr__(self, name):
    return getattr(self._env, name)

  def render(self, *args, **kwargs):
    return self._render_fn(self._env, *args, **kwargs)


class ActionRepeat(object):
  """Repeat the agent action multiple steps."""

  def __init__(self, env, amount):
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      observ, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return observ, total_reward, done, info


class NormalizeActions(object):

  def __init__(self, env):
    self._env = env
    low, high = env.action_space.low, env.action_space.high
    self._enabled = np.logical_and(np.isfinite(low), np.isfinite(high))
    self._low = np.where(self._enabled, low, -np.ones_like(low))
    self._high = np.where(self._enabled, high, np.ones_like(low))

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    space = self._env.action_space
    low = np.where(self._enabled, -np.ones_like(space.low), space.low)
    high = np.where(self._enabled, np.ones_like(space.high), space.high)
    return gym.spaces.Box(low, high, dtype=space.dtype)

  def step(self, action):
    action = (action + 1) / 2 * (self._high - self._low) + self._low
    return self._env.step(action)


class DeepMindWrapper(object):
  """Wraps a DM Control environment into a Gym interface."""

  metadata = {'render.modes': ['rgb_array']}
  reward_range = (-np.inf, np.inf)

  def __init__(self, env, render_size=(64, 64), camera_id=0):
    self._env = env
    self._render_size = render_size
    self._camera_id = camera_id

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    components = {}
    for key, value in self._env.observation_spec().items():
      components[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    return gym.spaces.Dict(components)

  @property
  def action_space(self):
    action_spec = self._env.action_spec()
    return gym.spaces.Box(
        action_spec.minimum, action_spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': time_step.discount}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    return dict(time_step.observation)

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    del args  # Unused
    del kwargs  # Unused
    return self._env.physics.render(
        *self._render_size, camera_id=self._camera_id)


class MaximumDuration(object):
  """Limits the episode to a given upper number of decision points."""

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    if self._step is None:
      raise RuntimeError('Must reset environment.')
    observ, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      self._step = None
    return observ, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class MinimumDuration(object):
  """Extends the episode to a given lower number of decision points."""

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    observ, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step < self._duration:
      done = False
    return observ, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()


class ProcessObservation(object):

  def __init__(self, env, process_fn):
    self._env = env
    self._process_fn = process_fn

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    return nested.map(
        lambda box: gym.spaces.Box(
            self._process_fn(box.low),
            self._process_fn(box.high),
            dtype=self._process_fn(box.low).dtype),
        self._env.observation_space)

  def step(self, action):
    observ, reward, done, info = self._env.step(action)
    observ = self._process_fn(observ)
    return observ, reward, done, info

  def reset(self):
    observ = self._env.reset()
    observ = self._process_fn(observ)
    return observ


class PadActions(object):
  """Pad action space to the largest action space."""

  def __init__(self, env, spaces):
    self._env = env
    self._action_space = self._pad_box_space(spaces)

  @property
  def observation_space(self):
    return self._env.observation_space

  @property
  def action_space(self):
    return self._action_space

  def step(self, action, *args, **kwargs):
    action = action[:len(self._env.action_space.low)]
    return self._env.step(action, *args, **kwargs)

  def reset(self, *args, **kwargs):
    return self._env.reset(*args, **kwargs)

  def _pad_box_space(self, spaces):
    assert all(len(space.low.shape) == 1 for space in spaces)
    length = max(len(space.low) for space in spaces)
    low, high = np.inf * np.ones(length), -np.inf * np.ones(length)
    for space in spaces:
      low[:len(space.low)] = np.minimum(space.low, low[:len(space.low)])
      high[:len(space.high)] = np.maximum(space.high, high[:len(space.high)])
    return gym.spaces.Box(low, high, dtype=np.float32)


class CollectGymDataset(object):
  """Collect transition tuples and store episodes as Numpy files.

  The time indices of the collected epiosde use the convention that at each
  time step, the agent first decides on an action, and the environment then
  returns the reward and observation.

  This means the action causes the environment state and thus observation and
  rewards at the same time step. A dynamics model can thus predict the sequence
  of observations and rewards from the sequence of actions.

  The first transition tuple contains the observation returned from resetting
  the environment, together with zeros for the action and reward. Thus, the
  episode length is one more than the number of decision points.
  """

  def __init__(self, env, outdir):
    self._env = env
    self._outdir = outdir and os.path.expanduser(outdir)
    self._episode = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action, *args, **kwargs):
    if kwargs.get('blocking', True):
      transition = self._env.step(action, *args, **kwargs)
      return self._process_step(action, *transition)
    else:
      future = self._env.step(action, *args, **kwargs)
      return lambda: self._process_step(action, *future())

  def reset(self, *args, **kwargs):
    if kwargs.get('blocking', True):
      observ = self._env.reset(*args, **kwargs)
      return self._process_reset(observ)
    else:
      future = self._env.reset(*args, **kwargs)
      return lambda: self._process_reset(future())

  def _process_step(self, action, observ, reward, done, info):
    transition = self._process_observ(observ).copy()
    transition['action'] = action
    transition['reward'] = reward
    self._episode.append(transition)
    if done:
      episode = self._get_episode()
      # info['episode'] = episode
      if self._outdir:
        filename = self._get_filename()
        self._write(episode, filename)
    return observ, reward, done, info

  def _process_reset(self, observ):
    # Resetting the environment provides the observation for time step zero.
    # The action and reward are not known for this time step, so we zero them.
    transition = self._process_observ(observ).copy()
    transition['action'] = np.zeros_like(self.action_space.low)
    transition['reward'] = 0.0
    self._episode = [transition]
    return observ

  def _process_observ(self, observ):
    if not isinstance(observ, dict):
      observ = {'observ': observ}
    return observ

  def _get_filename(self):
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4()).replace('-', '')
    filename = '{}-{}.npz'.format(timestamp, identifier)
    filename = os.path.join(self._outdir, filename)
    return filename

  def _get_episode(self):
    episode = {k: [t[k] for t in self._episode] for k in self._episode[0]}
    episode = {k: np.array(v) for k, v in episode.items()}
    for key, sequence in episode.items():
      if sequence.dtype == 'object':
        message = "Sequence '{}' is not numeric:\n{}"
        raise RuntimeError(message.format(key, sequence))
    return episode

  def _write(self, episode, filename):
    if not tf.gfile.Exists(self._outdir):
      tf.gfile.MakeDirs(self._outdir)
    with io.BytesIO() as file_:
      np.savez_compressed(file_, **episode)
      file_.seek(0)
      with tf.gfile.Open(filename, 'w') as ff:
        ff.write(file_.read())
    folder = os.path.basename(self._outdir)
    name = os.path.splitext(os.path.basename(filename))[0]
    print('Recorded episode {} to {}.'.format(name, folder))


class ConvertTo32Bit(object):
  """Convert data types of an OpenAI Gym environment to 32 bit."""

  def __init__(self, env):
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    observ, reward, done, info = self._env.step(action)
    observ = nested.map(self._convert_observ, observ)
    reward = self._convert_reward(reward)
    return observ, reward, done, info

  def reset(self):
    observ = self._env.reset()
    observ = nested.map(self._convert_observ, observ)
    return observ

  def _convert_observ(self, observ):
    if not np.isfinite(observ).all():
      raise ValueError('Infinite observation encountered.')
    if observ.dtype == np.float64:
      return observ.astype(np.float32)
    if observ.dtype == np.int64:
      return observ.astype(np.int32)
    return observ

  def _convert_reward(self, reward):
    if not np.isfinite(reward).all():
      raise ValueError('Infinite reward encountered.')
    return np.array(reward, dtype=np.float32)


class Async(object):
  """Step environment in a separate process for lock free paralellism."""

  # Message types for communication via the pipe.
  _ACCESS = 1
  _CALL = 2
  _RESULT = 3
  _EXCEPTION = 4
  _CLOSE = 5

  def __init__(self, constructor, strategy='thread'):
    """Step environment in a separate process for lock free parallelism.

    The environment will be created in the external process by calling the
    specified callable. This can be an environment class, or a function
    creating the environment and potentially wrapping it. The returned
    environment should not access global variables.

    Args:
      constructor: Callable that creates and returns an OpenAI gym environment.

    Attributes:
      observation_space: The cached observation space of the environment.
      action_space: The cached action space of the environment.
    """
    if strategy == 'thread':
      import multiprocessing.dummy as mp
    elif strategy == 'process':
      import multiprocessing as mp
    else:
      raise NotImplementedError(strategy)
    self._conn, conn = mp.Pipe()
    self._process = mp.Process(target=self._worker, args=(constructor, conn))
    atexit.register(self.close)
    self._process.start()
    self._observ_space = None
    self._action_space = None

  @property
  def observation_space(self):
    if not self._observ_space:
      self._observ_space = self.__getattr__('observation_space')
    return self._observ_space

  @property
  def action_space(self):
    if not self._action_space:
      self._action_space = self.__getattr__('action_space')
    return self._action_space

  def __getattr__(self, name):
    """Request an attribute from the environment.

    Note that this involves communication with the external process, so it can
    be slow.

    Args:
      name: Attribute to access.

    Returns:
      Value of the attribute.
    """
    self._conn.send((self._ACCESS, name))
    return self._receive()

  def call(self, name, *args, **kwargs):
    """Asynchronously call a method of the external environment.

    Args:
      name: Name of the method to call.
      *args: Positional arguments to forward to the method.
      **kwargs: Keyword arguments to forward to the method.

    Returns:
      Promise object that blocks and provides the return value when called.
    """
    payload = name, args, kwargs
    self._conn.send((self._CALL, payload))
    return self._receive

  def close(self):
    """Send a close message to the external process and join it."""
    try:
      self._conn.send((self._CLOSE, None))
      self._conn.close()
    except IOError:
      # The connection was already closed.
      pass
    self._process.join()

  def step(self, action, blocking=True):
    """Step the environment.

    Args:
      action: The action to apply to the environment.
      blocking: Whether to wait for the result.

    Returns:
      Transition tuple when blocking, otherwise callable that returns the
      transition tuple.
    """
    promise = self.call('step', action)
    if blocking:
      return promise()
    else:
      return promise

  def reset(self, blocking=True):
    """Reset the environment.

    Args:
      blocking: Whether to wait for the result.

    Returns:
      New observation when blocking, otherwise callable that returns the new
      observation.
    """
    promise = self.call('reset')
    if blocking:
      return promise()
    else:
      return promise

  def _receive(self):
    """Wait for a message from the worker process and return its payload.

    Raises:
      Exception: An exception was raised inside the worker process.
      KeyError: The received message is of an unknown type.

    Returns:
      Payload object of the message.
    """
    try:
      message, payload = self._conn.recv()
    except OSError:
      raise RuntimeError('Environment worker crashed.')
    # Re-raise exceptions in the main process.
    if message == self._EXCEPTION:
      stacktrace = payload
      raise Exception(stacktrace)
    if message == self._RESULT:
      return payload
    raise KeyError('Received message of unexpected type {}'.format(message))

  def _worker(self, constructor, conn):
    """The process waits for actions and sends back environment results.

    Args:
      constructor: Constructor for the OpenAI Gym environment.
      conn: Connection for communication to the main process.

    Raises:
      KeyError: When receiving a message of unknown type.
    """
    try:
      env = constructor()
      while True:
        try:
          # Only block for short times to have keyboard exceptions be raised.
          if not conn.poll(0.1):
            continue
          message, payload = conn.recv()
        except (EOFError, KeyboardInterrupt):
          break
        if message == self._ACCESS:
          name = payload
          result = getattr(env, name)
          conn.send((self._RESULT, result))
          continue
        if message == self._CALL:
          name, args, kwargs = payload
          result = getattr(env, name)(*args, **kwargs)
          conn.send((self._RESULT, result))
          continue
        if message == self._CLOSE:
          assert payload is None
          break
        raise KeyError('Received message of unknown type {}'.format(message))
    except Exception:
      stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
      print('Error in environment process: {}'.format(stacktrace))
      try:
        conn.send((self._EXCEPTION, stacktrace))
      except Exception:
        print('Failed to send exception back to main process.')
    try:
      conn.close()
    except Exception:
      print('Failed to properly close connection.')
