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


class StreamingMean(object):
  """Compute a streaming estimation of the mean of submitted tensors."""

  def __init__(self, shape, dtype, name):
    """Specify the shape and dtype of the mean to be estimated.

    Note that a float mean to zero submitted elements is NaN, while computing
    the integer mean of zero elements raises a division by zero error.

    Args:
      shape: Shape of the mean to compute.
      dtype: Data type of the mean to compute.
    """
    self._dtype = dtype
    with tf.variable_scope(name):
      self._sum = tf.get_variable(
          'sum', shape, dtype,
          tf.constant_initializer(0),
          trainable=False)
      self._count = tf.get_variable(
          'count', (), tf.int32,
          tf.constant_initializer(0),
          trainable=False)

  @property
  def value(self):
    """The current value of the mean."""
    return self._sum / tf.cast(self._count, self._dtype)

  @property
  def count(self):
    """The number of submitted samples."""
    return self._count

  def submit(self, value):
    """Submit a single or batch tensor to refine the streaming mean."""
    value = tf.convert_to_tensor(value)
    # Add a batch dimension if necessary.
    if value.shape.ndims == self._sum.shape.ndims:
      value = value[None, ...]
    if str(value.shape[1:]) != str(self._sum.shape):
      message = 'Value shape ({}) does not fit tracked tensor ({}).'
      raise ValueError(message.format(value.shape[1:], self._sum.shape))
    def assign():
      return tf.group(
        self._sum.assign_add(tf.reduce_sum(value, 0)),
        self._count.assign_add(tf.shape(value)[0]))
    not_empty = tf.cast(tf.reduce_prod(tf.shape(value)), tf.bool)
    return tf.cond(not_empty, assign, tf.no_op)

  def clear(self):
    """Return the mean estimate and reset the streaming statistics."""
    value = self._sum / tf.cast(self._count, self._dtype)
    with tf.control_dependencies([value]):
      reset_value = self._sum.assign(tf.zeros_like(self._sum))
      reset_count = self._count.assign(0)
    with tf.control_dependencies([reset_value, reset_count]):
      return tf.identity(value)
