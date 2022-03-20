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


def image_strip_summary(name, images, max_length=100, max_batch=10):
  """Create an image summary that places frames of a video tensor side by side.

  Args:
    name: Name tag of the summary.
    images: Tensor with the dimensions batch, time, height, width, channels.
    max_length: Maximum number of frames per sequence to include.
    max_batch: Maximum number of sequences to include.

  Returns:
    Summary string tensor.
  """
  if max_batch:
    images = images[:max_batch]
  if max_length:
    images = images[:, :max_length]
  if images.dtype == tf.uint8:
    images = tf.to_float(images) / 255.0
  length, width = tf.shape(images)[1], tf.shape(images)[3]
  images = tf.transpose(images, [0, 2, 1, 3, 4])
  images = tf.reshape(images, [1, -1, length * width, 3])
  images = tf.clip_by_value(images, 0., 1.)
  return tf.summary.image(name, images)
