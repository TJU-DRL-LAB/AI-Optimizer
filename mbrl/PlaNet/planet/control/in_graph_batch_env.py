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

"""Batch of environments inside the TensorFlow graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf


class InGraphBatchEnv(object):
  """Batch of environments inside the TensorFlow graph.

  The batch of environments will be stepped and reset inside of the graph using
  a tf.py_func(). The current batch of observations, actions, rewards, and done
  flags are held in according variables.
  """

  def __init__(self, batch_env):
    """Batch of environments inside the TensorFlow graph.

    Args:
      batch_env: Batch environment.
    """
    self._batch_env = batch_env
    batch_dims = (len(self._batch_env),)
    observ_shape = self._parse_shape(self._batch_env.observation_space)
    observ_dtype = self._parse_dtype(self._batch_env.observation_space)
    action_shape = self._parse_shape(self._batch_env.action_space)
    action_dtype = self._parse_dtype(self._batch_env.action_space)
    with tf.variable_scope('env_temporary'):
      self._observ = tf.get_variable(
          'observ', batch_dims + observ_shape, observ_dtype,
          tf.constant_initializer(0), trainable=False)
      self._action = tf.get_variable(
          'action', batch_dims + action_shape, action_dtype,
          tf.constant_initializer(0), trainable=False)
      self._reward = tf.get_variable(
          'reward', batch_dims, tf.float32,
          tf.constant_initializer(0), trainable=False)
      # This variable should be boolean, but tf.scatter_update() does not
      # support boolean resource variables yet.
      self._done = tf.get_variable(
          'done', batch_dims, tf.int32,
          tf.constant_initializer(False), trainable=False)

  def __getattr__(self, name):
    """Forward unimplemented attributes to one of the original environments.

    Args:
      name: Attribute that was accessed.

    Returns:
      Value behind the attribute name in one of the original environments.
    """
    return getattr(self._batch_env, name)

  def __len__(self):
    """Number of combined environments."""
    return len(self._batch_env)

  def __getitem__(self, index):
    """Access an underlying environment by index."""
    return self._batch_env[index]

  def step(self, action):
    """Step the batch of environments.

    The results of the step can be accessed from the variables defined below.

    Args:
      action: Tensor holding the batch of actions to apply.

    Returns:
      Operation.
    """
    with tf.name_scope('environment/simulate'):
      observ_dtype = self._parse_dtype(self._batch_env.observation_space)
      observ, reward, done = tf.py_func(
          lambda a: self._batch_env.step(a)[:3], [action],
          [observ_dtype, tf.float32, tf.bool], name='step')
      # reward = tf.cast(reward, tf.float32)
      return tf.group(
          self._observ.assign(observ),
          self._action.assign(action),
          self._reward.assign(reward),
          self._done.assign(tf.to_int32(done)))

  def reset(self, indices=None):
    """Reset the batch of environments.

    Args:
      indices: The batch indices of the environments to reset; defaults to all.

    Returns:
      Batch tensor of the new observations.
    """
    if indices is None:
      indices = tf.range(len(self._batch_env))
    observ_dtype = self._parse_dtype(self._batch_env.observation_space)
    observ = tf.py_func(
        self._batch_env.reset, [indices], observ_dtype, name='reset')
    reward = tf.zeros_like(indices, tf.float32)
    done = tf.zeros_like(indices, tf.int32)
    with tf.control_dependencies([
        tf.scatter_update(self._observ, indices, observ),
        tf.scatter_update(self._reward, indices, reward),
        tf.scatter_update(self._done, indices, tf.to_int32(done))]):
      return tf.identity(observ)

  @property
  def observ(self):
    """Access the variable holding the current observation."""
    return self._observ + 0

  @property
  def action(self):
    """Access the variable holding the last received action."""
    return self._action + 0

  @property
  def reward(self):
    """Access the variable holding the current reward."""
    return self._reward + 0

  @property
  def done(self):
    """Access the variable indicating whether the episode is done."""
    return tf.cast(self._done, tf.bool)

  def close(self):
    """Send close messages to the external process and join them."""
    self._batch_env.close()

  def _parse_shape(self, space):
    """Get a tensor shape from a OpenAI Gym space.

    Args:
      space: Gym space.

    Raises:
      NotImplementedError: For spaces other than Box and Discrete.

    Returns:
      Shape tuple.
    """
    if isinstance(space, gym.spaces.Discrete):
      return ()
    if isinstance(space, gym.spaces.Box):
      return space.shape
    raise NotImplementedError("Unsupported space '{}.'".format(space))

  def _parse_dtype(self, space):
    """Get a tensor dtype from a OpenAI Gym space.

    Args:
      space: Gym space.

    Raises:
      NotImplementedError: For spaces other than Box and Discrete.

    Returns:
      TensorFlow data type.
    """
    if isinstance(space, gym.spaces.Discrete):
      return tf.int32
    if isinstance(space, gym.spaces.Box):
      if space.low.dtype == np.uint8:
        return tf.uint8
      else:
        return tf.float32
    raise NotImplementedError()
