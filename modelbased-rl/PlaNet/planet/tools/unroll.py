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

import tensorflow as tf

from planet.tools import nested
from planet.tools import shape


def closed_loop(cell, embedded, prev_action, debug=False):
  use_obs = tf.ones(tf.shape(embedded[:, :, :1])[:3], tf.bool)
  (prior, posterior), _ = tf.nn.dynamic_rnn(
      cell, (embedded, prev_action, use_obs), dtype=tf.float32)
  if debug:
    with tf.control_dependencies([tf.assert_equal(
        tf.shape(nested.flatten(posterior)[0])[1], tf.shape(embedded)[1])]):
      prior = nested.map(tf.identity, prior)
      posterior = nested.map(tf.identity, posterior)
  return prior, posterior


def open_loop(cell, embedded, prev_action, context=1, debug=False):
  use_obs = tf.ones(tf.shape(embedded[:, :context, :1])[:3], tf.bool)
  (_, closed_state), last_state = tf.nn.dynamic_rnn(
      cell, (embedded[:, :context], prev_action[:, :context], use_obs),
      dtype=tf.float32)
  use_obs = tf.zeros(tf.shape(embedded[:, context:, :1])[:3], tf.bool)
  (_, open_state), _ = tf.nn.dynamic_rnn(
      cell, (0 * embedded[:, context:], prev_action[:, context:], use_obs),
      initial_state=last_state)
  state = nested.map(
      lambda x, y: tf.concat([x, y], 1),
      closed_state, open_state)
  if debug:
    with tf.control_dependencies([tf.assert_equal(
        tf.shape(nested.flatten(state)[0])[1], tf.shape(embedded)[1])]):
      state = nested.map(tf.identity, state)
  return state


def planned(
    cell, objective_fn, embedded, prev_action, planner, context=1, length=20,
    amount=1000, debug=False):
  use_obs = tf.ones(tf.shape(embedded[:, :context, :1])[:3], tf.bool)
  (_, closed_state), last_state = tf.nn.dynamic_rnn(
      cell, (embedded[:, :context], prev_action[:, :context], use_obs),
      dtype=tf.float32)
  _, plan_state, return_ = planner(
      cell, objective_fn, last_state,
      obs_shape=shape.shape(embedded)[2:],
      action_shape=shape.shape(prev_action)[2:],
      horizon=length, amount=amount)
  state = nested.map(
      lambda x, y: tf.concat([x, y], 1),
      closed_state, plan_state)
  if debug:
    with tf.control_dependencies([tf.assert_equal(
        tf.shape(nested.flatten(state)[0])[1], context + length)]):
      state = nested.map(tf.identity, state)
      return_ = tf.identity(return_)
  return state, return_
