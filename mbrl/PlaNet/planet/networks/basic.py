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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from planet import tools


def feed_forward(
    state, data_shape, num_layers=2, activation=tf.nn.relu,
    mean_activation=None, stop_gradient=False, trainable=True, units=100,
    std=1.0, low=-1.0, high=1.0, dist='normal'):
  """Create a model returning unnormalized MSE distribution."""
  hidden = state
  if stop_gradient:
    hidden = tf.stop_gradient(hidden)
  for _ in range(num_layers):
    hidden = tf.layers.dense(hidden, units, activation)
  mean = tf.layers.dense(
      hidden, int(np.prod(data_shape)), mean_activation, trainable=trainable)
  mean = tf.reshape(mean, tools.shape(state)[:-1] + data_shape)
  if std == 'learned':
    std = tf.layers.dense(
        hidden, int(np.prod(data_shape)), None, trainable=trainable)
    std = tf.nn.softplus(std + 0.55) + 0.01
    std = tf.reshape(std, tools.shape(state)[:-1] + data_shape)
  if dist == 'normal':
    dist = tfd.Normal(mean, std)
  elif dist == 'truncated_normal':
    # https://www.desmos.com/calculator/3o96eyqxib
    dist = tfd.TruncatedNormal(mean, std, low, high)
  elif dist == 'tanh_normal':
    # https://www.desmos.com/calculator/sxpp7ectjv
    dist = tfd.Normal(mean, std)
    dist = tfd.TransformedDistribution(dist, tfp.bijectors.Tanh())
  elif dist == 'deterministic':
    dist = tfd.Deterministic(mean)
  else:
    raise NotImplementedError(dist)
  dist = tfd.Independent(dist, len(data_shape))
  return dist
