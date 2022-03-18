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

import os

import numpy as np
import tensorflow as tf


def count_dataset(directory, key='reward'):
  directory = os.path.expanduser(directory)
  if not tf.gfile.Exists(directory):
    message = "Data set directory '{}' does not exist."
    raise ValueError(message.format(directory))
  pattern = os.path.join(directory, '*.npz')
  def func():
    filenames = tf.gfile.Glob(pattern)
    episodes = len(filenames)
    episodes = np.array(episodes, dtype=np.int32)
    return episodes
  return tf.py_func(func, [], tf.int32)
