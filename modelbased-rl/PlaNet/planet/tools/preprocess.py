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


def preprocess(image, bits):
  bins = 2 ** bits
  image = tf.to_float(image)
  if bits < 8:
    image = tf.floor(image / 2 ** (8 - bits))
  image = image / bins
  image = image + tf.random_uniform(tf.shape(image), 0, 1.0 / bins)
  image = image - 0.5
  return image


def postprocess(image, bits, dtype=tf.float32):
  bins = 2 ** bits
  if dtype == tf.float32:
    image = tf.floor(bins * (image + 0.5)) / bins
  elif dtype == tf.uint8:
    image = image + 0.5
    image = tf.floor(bins * image)
    image = image * (256.0 / bins)
    image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
  else:
    raise NotImplementedError(dtype)
  return image
