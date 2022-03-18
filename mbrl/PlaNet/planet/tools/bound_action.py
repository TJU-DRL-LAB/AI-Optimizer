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

import tensorflow as tf


def bound_action(action, strategy):
  if strategy == 'none':
    pass
  elif strategy == 'clip':
    forward = tf.stop_gradient(tf.clip_by_value(action, -1.0, 1.0))
    action = action - tf.stop_gradient(action) + forward
  elif strategy == 'tanh':
    action = tf.tanh(action)
  else:
    raise NotImplementedError(strategy)
  return action
