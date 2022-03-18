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

"""Chunk sequences into fixed lengths."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from planet.tools import nested


def chunk_sequence(sequence, chunk_length, randomize=True, num_chunks=None):
  """Split a nested dict of sequence tensors into a batch of chunks.

  This function does not expect a batch of sequences, but a single sequence. A
  `length` key is added if it did not exist already. When `randomize` is set,
  up to `chunk_length - 1` initial frames will be discarded. Final frames that
  do not fit into a chunk are always discarded.

  Args:
    sequence: Nested dict of tensors with time dimension.
    chunk_length: Size of chunks the sequence will be split into.
    randomize: Start chunking from a random offset in the sequence,
        enforcing that at least one chunk is generated.
    num_chunks: Optionally specify the exact number of chunks to be extracted
        from the sequence. Requires input to be long enough.

  Returns:
    Nested dict of sequence tensors with chunk dimension.
  """
  with tf.device('/cpu:0'):
    if 'length' in sequence:
      length = sequence.pop('length')
    else:
      length = tf.shape(nested.flatten(sequence)[0])[0]
    if randomize:
      if num_chunks is None:
        num_chunks = tf.maximum(1, length // chunk_length - 1)
      else:
        num_chunks = num_chunks + 0 * length
      used_length = num_chunks * chunk_length
      max_offset = length - used_length
      offset = tf.random_uniform((), 0, max_offset + 1, dtype=tf.int32)
    else:
      if num_chunks is None:
        num_chunks = length // chunk_length
      else:
        num_chunks = num_chunks + 0 * length
      used_length = num_chunks * chunk_length
      max_offset = 0
      offset = 0
    clipped = nested.map(
        lambda tensor: tensor[offset: offset + used_length],
        sequence)
    chunks = nested.map(
        lambda tensor: tf.reshape(
            tensor, [num_chunks, chunk_length] + tensor.shape[1:].as_list()),
        clipped)
    chunks['length'] = chunk_length * tf.ones((num_chunks,), dtype=tf.int32)
    return chunks


def _pad_tensor(tensor, length, value):
  tiling = [length] + ([1] * (tensor.shape.ndims - 1))
  padding = tf.tile(0 * tensor[:1] + value, tiling)
  padded = tf.concat([tensor, padding], 0)
  return padded
