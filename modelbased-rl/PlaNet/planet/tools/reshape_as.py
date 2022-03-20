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

from planet.tools import nested


def reshape_as(tensor, reference):
  if isinstance(tensor, (list, tuple, dict)):
    return nested.map(tensor, lambda x: reshape_as(x, reference))
  tensor = tf.convert_to_tensor(tensor)
  reference = tf.convert_to_tensor(reference)
  statics = reference.shape.as_list()
  dynamics = tf.shape(reference)
  shape = [
      static if static is not None else dynamics[index]
      for index, static in enumerate(statics)]
  return tf.reshape(tensor, shape)
