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

import functools

import numpy as np
import tensorflow as tf

from planet.tools import nested
from planet.tools import shape


def overshooting(
    cell, target, embedded, prev_action, length, amount, posterior=None,
    ignore_input=False):
  """Perform open loop rollouts from the posteriors at every step.

  First, we apply the encoder to embed raw inputs and apply the model to obtain
  posterior states for every time step. Then, we perform `amount` long open
  loop rollouts from these posteriors.

  Note that the actions should be those leading to the current time step. So
  under common convention, it contains the last actions while observations are
  the current ones.

  Input:

    target, embedded:
      [A B C D E F] [A B C D E  ]

    prev_action:
      [0 A B C D E] [0 A B C D  ]

    length:
      [6 5]

    amount:
      3

  Output:

    prior, posterior, target:
      [A B C D E F] [A B C D E  ]
      [B C D E F  ] [B C D E    ]
      [C D E F    ] [C D E      ]
      [D E F      ] [D E        ]

    mask:
      [1 1 1 1 1 1] [1 1 1 1 1 0]
      [1 1 1 1 1 0] [1 1 1 1 0 0]
      [1 1 1 1 0 0] [1 1 1 0 0 0]
      [1 1 1 0 0 0] [1 1 0 0 0 0]

  """
  # Closed loop unroll to get posterior states, which are the starting points
  # for open loop unrolls. We don't need the last time step, since we have no
  # targets for unrolls from it.
  if posterior is None:
    use_obs = tf.ones(tf.shape(
        nested.flatten(embedded)[0][:, :, :1])[:3], tf.bool)
    use_obs = tf.cond(
        tf.convert_to_tensor(ignore_input),
        lambda: tf.zeros_like(use_obs, tf.bool),
        lambda: use_obs)
    (_, posterior), _ = tf.nn.dynamic_rnn(
        cell, (embedded, prev_action, use_obs), length, dtype=tf.float32,
        swap_memory=True)

  # Arrange inputs for every iteration in the open loop unroll. Every loop
  # iteration below corresponds to one row in the docstring illustration.
  max_length = shape.shape(nested.flatten(embedded)[0])[1]
  first_output = {
      # 'observ': embedded,
      'prev_action': prev_action,
      'posterior': posterior,
      'target': target,
      'mask': tf.sequence_mask(length, max_length, tf.int32),
  }

  progress_fn = lambda tensor: tf.concat([tensor[:, 1:], 0 * tensor[:, :1]], 1)
  other_outputs = tf.scan(
      lambda past_output, _: nested.map(progress_fn, past_output),
      tf.range(amount), first_output)
  sequences = nested.map(
      lambda lhs, rhs: tf.concat([lhs[None], rhs], 0),
      first_output, other_outputs)

  # Merge batch and time dimensions of steps to compute unrolls from every
  # time step as one batch. The time dimension becomes the number of
  # overshooting distances.
  sequences = nested.map(
      lambda tensor: _merge_dims(tensor, [1, 2]),
      sequences)
  sequences = nested.map(
      lambda tensor: tf.transpose(
          tensor, [1, 0] + list(range(2, tensor.shape.ndims))),
      sequences)
  merged_length = tf.reduce_sum(sequences['mask'], 1)

  # Mask out padding frames; unnecessary if the input is already masked.
  sequences = nested.map(
      lambda tensor: tensor * tf.cast(
          _pad_dims(sequences['mask'], tensor.shape.ndims),
          tensor.dtype),
      sequences)

  # Compute open loop rollouts.
  use_obs = tf.zeros(tf.shape(sequences['mask']), tf.bool)[..., None]
  embed_size = nested.flatten(embedded)[0].shape[2].value
  obs = tf.zeros(shape.shape(sequences['mask']) + [embed_size])
  prev_state = nested.map(
      lambda tensor: tf.concat([0 * tensor[:, :1], tensor[:, :-1]], 1),
      posterior)
  prev_state = nested.map(
      lambda tensor: _merge_dims(tensor, [0, 1]), prev_state)
  (priors, _), _ = tf.nn.dynamic_rnn(
      cell, (obs, sequences['prev_action'], use_obs),
      merged_length,
      prev_state)

  # Restore batch dimension.
  target, prior, posterior, mask = nested.map(
      functools.partial(_restore_batch_dim, batch_size=shape.shape(length)[0]),
      (sequences['target'], priors, sequences['posterior'], sequences['mask']))

  mask = tf.cast(mask, tf.bool)
  return target, prior, posterior, mask


def _merge_dims(tensor, dims):
  """Flatten consecutive axes of a tensor trying to preserve static shapes."""
  if isinstance(tensor, (list, tuple, dict)):
    return nested.map(tensor, lambda x: _merge_dims(x, dims))
  tensor = tf.convert_to_tensor(tensor)
  if (np.array(dims) - min(dims) != np.arange(len(dims))).all():
    raise ValueError('Dimensions to merge must all follow each other.')
  start, end = dims[0], dims[-1]
  output = tf.reshape(tensor, tf.concat([
      tf.shape(tensor)[:start],
      [tf.reduce_prod(tf.shape(tensor)[start: end + 1])],
      tf.shape(tensor)[end + 1:]], axis=0))
  merged = tensor.shape[start: end + 1].as_list()
  output.set_shape(
      tensor.shape[:start].as_list() +
      [None if None in merged else np.prod(merged)] +
      tensor.shape[end + 1:].as_list())
  return output


def _pad_dims(tensor, rank):
  """Append empty dimensions to the tensor until it is of the given rank."""
  for _ in range(rank - tensor.shape.ndims):
    tensor = tensor[..., None]
  return tensor


def _restore_batch_dim(tensor, batch_size):
  """Split batch dimension out of the first dimension of a tensor."""
  initial = shape.shape(tensor)
  desired = [batch_size, initial[0] // batch_size] + initial[1:]
  return tf.reshape(tensor, desired)
