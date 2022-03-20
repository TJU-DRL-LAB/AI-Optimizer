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
from tensorflow_probability import distributions as tfd

from planet import tools
from planet.models import base


class DRNN(base.Base):
  r"""Doubly recurrent state-space model.

  Prior:         Posterior:

  (a)    (a)     (a,o)  (a,o)
   |      |       : :    : :
   v      v       v v    v v
  [e]--->[e]      [e]...>[e]
   |      |        :      :
   v      v        v      v
  (s)--->(s)      (s)--->(s)
   |      |        |      |
   v      v        v      v
  [d]--->[d]      [d]--->[d]
   |      |        |      |
   v      v        v      v
  (o)    (o)      (o)    (o)
  """

  def __init__(
      self, state_size, belief_size, embed_size,
      mean_only=False, min_stddev=1e-1, activation=tf.nn.elu,
      encoder_to_decoder=False, sample_to_sample=True,
      sample_to_encoder=True, decoder_to_encoder=False,
      decoder_to_sample=True, action_to_decoder=False):
    self._state_size = state_size
    self._belief_size = belief_size
    self._embed_size = embed_size
    self._encoder_cell = tf.contrib.rnn.GRUBlockCell(self._belief_size)
    self._decoder_cell = tf.contrib.rnn.GRUBlockCell(self._belief_size)
    self._kwargs = dict(units=self._embed_size, activation=tf.nn.relu)
    self._mean_only = mean_only
    self._min_stddev = min_stddev
    self._encoder_to_decoder = encoder_to_decoder
    self._sample_to_sample = sample_to_sample
    self._sample_to_encoder = sample_to_encoder
    self._decoder_to_encoder = decoder_to_encoder
    self._decoder_to_sample = decoder_to_sample
    self._action_to_decoder = action_to_decoder
    posterior_tpl = tf.make_template('posterior', self._posterior)
    super(DRNN, self).__init__(posterior_tpl, posterior_tpl)

  @property
  def state_size(self):
    return {
        'encoder_state': self._encoder_cell.state_size,
        'decoder_state': self._decoder_cell.state_size,
        'mean': self._state_size,
        'stddev': self._state_size,
        'sample': self._state_size,
    }

  def dist_from_state(self, state, mask=None):
    """Extract the latent distribution from a prior or posterior state."""
    if mask is not None:
      stddev = tools.mask(state['stddev'], mask, value=1)
    else:
      stddev = state['stddev']
    dist = tfd.MultivariateNormalDiag(state['mean'], stddev)
    return dist

  def features_from_state(self, state):
    """Extract features for the decoder network from a prior or posterior."""
    return state['decoder_state']

  def divergence_from_states(self, lhs, rhs, mask=None):
    """Compute the divergence measure between two states."""
    lhs = self.dist_from_state(lhs, mask)
    rhs = self.dist_from_state(rhs, mask)
    divergence = tfd.kl_divergence(lhs, rhs)
    if mask is not None:
      divergence = tools.mask(divergence, mask)
    return divergence

  def _posterior(self, prev_state, prev_action, obs):
    """Compute posterior state from previous state and current observation."""

    # Recurrent encoder.
    encoder_inputs = [obs, prev_action]
    if self._sample_to_encoder:
      encoder_inputs.append(prev_state['sample'])
    if self._decoder_to_encoder:
      encoder_inputs.append(prev_state['decoder_state'])
    encoded, encoder_state = self._encoder_cell(
        tf.concat(encoder_inputs, -1), prev_state['encoder_state'])

    # Sample sequence.
    sample_inputs = [encoded]
    if self._sample_to_sample:
      sample_inputs.append(prev_state['sample'])
    if self._decoder_to_sample:
      sample_inputs.append(prev_state['decoder_state'])
    hidden = tf.layers.dense(
        tf.concat(sample_inputs, -1), **self._kwargs)
    mean = tf.layers.dense(hidden, self._state_size, None)
    stddev = tf.layers.dense(hidden, self._state_size, tf.nn.softplus)
    stddev += self._min_stddev
    if self._mean_only:
      sample = mean
    else:
      sample = tfd.MultivariateNormalDiag(mean, stddev).sample()

    # Recurrent decoder.
    decoder_inputs = [sample]
    if self._encoder_to_decoder:
      decoder_inputs.append(prev_state['encoder_state'])
    if self._action_to_decoder:
      decoder_inputs.append(prev_action)
    decoded, decoder_state = self._decoder_cell(
        tf.concat(decoder_inputs, -1), prev_state['decoder_state'])

    return {
        'encoder_state': encoder_state,
        'decoder_state': decoder_state,
        'mean': mean,
        'stddev': stddev,
        'sample': sample,
    }
