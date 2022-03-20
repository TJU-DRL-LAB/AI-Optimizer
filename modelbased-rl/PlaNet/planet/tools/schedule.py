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


def binary(step, batch_size, after, every, until):
  # https://www.desmos.com/calculator/csbhr5cjzz
  offset_step = step - after
  phase = tf.less(offset_step % every, batch_size)
  active = tf.greater_equal(step, after)
  if until > 0:
      active = tf.logical_and(active, tf.less(step, until))
  result = tf.logical_and(phase, active)
  result.set_shape(tf.TensorShape([]))
  return result


def linear(step, ramp, min=None, max=None):
  # https://www.desmos.com/calculator/nrumhgvxql
  if ramp == 0:
    result = tf.constant(1, tf.float32)
  if ramp > 0:
    result = tf.minimum(tf.to_float(step) / tf.to_float(ramp), 1)
  if ramp < 0:
    result = 1 - linear(step, abs(ramp))
  if min is not None and max is not None:
    assert min <= max
  if min is not None:
    assert 0 <= min <= 1
    result = tf.maximum(min, result)
  if max is not None:
    assert 0 <= min <= 1
    result = tf.minimum(result, max)
  result.set_shape(tf.TensorShape([]))
  return result


def linear_reset(step, ramp, after, every):
  # https://www.desmos.com/calculator/motbnqhacw
  assert every > ramp, (every, ramp)  # Would never reach max value.
  assert not (every != np.inf and after == np.inf), (every, after)
  step, ramp, after, every = [
      tf.to_float(x) for x in (step, ramp, after, every)]
  before = tf.to_float(tf.less(step, after)) * step
  after = tf.to_float(tf.greater_equal(step, after)) * ((step - after) % every)
  result = tf.minimum((before + after) / ramp, 1)
  result.set_shape(tf.TensorShape([]))
  return result
