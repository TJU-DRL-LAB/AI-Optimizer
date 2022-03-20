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

from planet import models
from planet.tools import overshooting


class _MockCell(models.Base):
  """Mock state space model.

  The transition function is to add the action to the observation. The
  posterior function is to return the ground truth observation. If actions or
  observations are collections, only their first element is used.
  """

  def __init__(self, obs_size):
    self._obs_size = obs_size
    super(_MockCell, self).__init__(
        tf.make_template('transition', self._transition),
        tf.make_template('posterior', self._posterior))

  @property
  def state_size(self):
    return {'obs': self._obs_size}

  def _transition(self, prev_state, prev_action, zero_obs):
    if isinstance(prev_action, (tuple, list)):
      prev_action = prev_action[0]
    return {'obs': prev_state['obs'] + prev_action}

  def _posterior(self, prev_state, prev_action, obs):
    if isinstance(obs, (tuple, list)):
      obs = obs[0]
    return {'obs': obs}


class OvershootingTest(tf.test.TestCase):

  def test_example(self):
    obs = tf.constant([
        [10, 20, 30, 40, 50, 60],
        [70, 80, 0, 0, 0, 0],
    ], dtype=tf.float32)[:, :, None]
    prev_action = tf.constant([
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [9.0, 0.7, 0, 0, 0, 0],
    ], dtype=tf.float32)[:, :, None]
    length = tf.constant([6, 2], dtype=tf.int32)
    cell = _MockCell(1)
    _, prior, posterior, mask = overshooting(
        cell, obs, obs, prev_action, length, 3)
    prior = tf.squeeze(prior['obs'], 3)
    posterior = tf.squeeze(posterior['obs'], 3)
    mask = tf.to_int32(mask)
    with self.test_session():
      # Each column corresponds to a different state step, and each row
      # corresponds to a different overshooting distance from there.
      self.assertAllEqual([
          [1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 0, 0],
          [1, 1, 1, 0, 0, 0],
      ], mask.eval()[0].T)
      self.assertAllEqual([
          [1, 1, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
      ], mask.eval()[1].T)
      self.assertAllClose([
          [0.0, 10.1, 20.2, 30.3, 40.4, 50.5],
          [0.1, 10.3, 20.5, 30.7, 40.9, 0],
          [0.3, 10.6, 20.9, 31.2, 0, 0],
          [0.6, 11.0, 21.4, 0, 0, 0],
      ], prior.eval()[0].T)
      self.assertAllClose([
          [10, 20, 30, 40, 50, 60],
          [20, 30, 40, 50, 60, 0],
          [30, 40, 50, 60, 0, 0],
          [40, 50, 60, 0, 0, 0],
      ], posterior.eval()[0].T)
      self.assertAllClose([
          [9.0, 70.7, 0, 0, 0, 0],
          [9.7, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
      ], prior.eval()[1].T)
      self.assertAllClose([
          [70, 80, 0, 0, 0, 0],
          [80, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0],
      ], posterior.eval()[1].T)

  def test_nested(self):
    obs = (tf.ones((3, 50, 1)), tf.ones((3, 50, 2)), tf.ones((3, 50, 3)))
    prev_action = (tf.ones((3, 50, 1)), tf.ones((3, 50, 2)))
    length = tf.constant([49, 50, 3], dtype=tf.int32)
    cell = _MockCell(1)
    overshooting(cell, obs, obs, prev_action, length, 3)


if __name__ == '__main__':
  tf.test.main()
