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

from planet import tools


class Base(tf.nn.rnn_cell.RNNCell):

  def __init__(self, transition_tpl, posterior_tpl, reuse=None):
    super(Base, self).__init__(_reuse=reuse)
    self._posterior_tpl = posterior_tpl
    self._transition_tpl = transition_tpl
    self._debug = False

  @property
  def state_size(self):
    raise NotImplementedError

  @property
  def updates(self):
    return []

  @property
  def losses(self):
    return []

  @property
  def output_size(self):
    return (self.state_size, self.state_size)

  def zero_state(self, batch_size, dtype):
    return tools.nested.map(
        lambda size: tf.zeros([batch_size, size], dtype),
        self.state_size)

  def call(self, inputs, prev_state):
    obs, prev_action, use_obs = inputs
    if self._debug:
      with tf.control_dependencies([tf.assert_equal(use_obs, use_obs[0, 0])]):
        use_obs = tf.identity(use_obs)
    use_obs = use_obs[0, 0]
    zero_obs = tools.nested.map(tf.zeros_like, obs)
    prior = self._transition_tpl(prev_state, prev_action, zero_obs)
    posterior = tf.cond(
        use_obs,
        lambda: self._posterior_tpl(prev_state, prev_action, obs),
        lambda: prior)
    return (prior, posterior), posterior
