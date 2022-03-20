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

from planet.tools import schedule as schedule_lib
from planet.tools import copy_weights


def track_network(
    trainer, batch_size, source_pattern, target_pattern, every, amount):
  init_op = tf.cond(
      tf.equal(trainer.global_step, 0),
      lambda: copy_weights.soft_copy_weights(
          source_pattern, target_pattern, 1.0),
      tf.no_op)
  schedule = schedule_lib.binary(trainer.step, batch_size, 0, every, -1)
  with tf.control_dependencies([init_op]):
    return tf.cond(
        tf.logical_and(tf.equal(trainer.phase, 'train'), schedule),
        lambda: copy_weights.soft_copy_weights(
            source_pattern, target_pattern, amount),
        tf.no_op)
