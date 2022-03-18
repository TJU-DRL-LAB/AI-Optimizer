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

from planet.tools import filter_variables_lib


def soft_copy_weights(source_pattern, target_pattern, amount):
  assert 0 < amount <= 1
  source_vars = filter_variables_lib.filter_variables(include=source_pattern)
  target_vars = filter_variables_lib.filter_variables(include=target_pattern)
  source_vars = sorted(source_vars, key=lambda x: x.name)
  target_vars = sorted(target_vars, key=lambda x: x.name)
  assert len(source_vars) == len(target_vars)
  updates = []
  for source, target in zip(source_vars, target_vars):
    assert source.name != target.name
    if amount == 1.0:
      updates.append(target.assign(source))
    else:
      updates.append(target.assign((1 - amount) * target + amount * source))
  return tf.group(*updates)
