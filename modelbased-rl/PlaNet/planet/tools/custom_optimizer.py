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


class CustomOptimizer(object):

  def __init__(
      self, optimizer_cls, step, log, learning_rate,
      include=None, exclude=None, clipping=None, schedule=None,
      debug=False, name='custom_optimizer'):
    if schedule:
      learning_rate *= schedule(step)
    self._step = step
    self._log = log
    self._learning_rate = learning_rate
    self._variables = filter_variables_lib.filter_variables(include, exclude)
    self._clipping = float(clipping)
    self._debug = debug
    self._name = name
    self._optimizer = optimizer_cls(learning_rate, name=name)

  def maybe_minimize(self, condition, loss):
    # loss = tf.cond(condition, lambda: loss, float)
    update_op, grad_norm = tf.cond(
        condition,
        lambda: self.minimize(loss),
        lambda: (tf.no_op(), 0.0))
    with tf.control_dependencies([update_op]):
      summary = tf.cond(
          tf.logical_and(condition, self._log),
          lambda: self.summarize(grad_norm), str)
    if self._debug:
      # print_op = tf.print('{}_grad_norm='.format(self._name), grad_norm)
      message = 'Zero gradient norm in {} optimizer.'.format(self._name)
      assertion = lambda: tf.assert_greater(grad_norm, 0.0, message=message)
      assert_op = tf.cond(condition, assertion, tf.no_op)
      with tf.control_dependencies([assert_op]):
        summary = tf.identity(summary)
    return summary, grad_norm

  def minimize(self, loss):
    with tf.name_scope('optimizer_{}'.format(self._name)):
      if self._debug:
        loss = tf.check_numerics(loss, '{}_loss'.format(self._name))
      gradients, variables = zip(*self._optimizer.compute_gradients(
          loss, self._variables, colocate_gradients_with_ops=True))
      grad_norm = tf.global_norm(gradients)
      if self._clipping:
        gradients, _ = tf.clip_by_global_norm(
            gradients, self._clipping, grad_norm)
      optimize = self._optimizer.apply_gradients(zip(gradients, variables))
    return optimize, grad_norm

  def summarize(self, grad_norm):
    summaries = []
    with tf.name_scope('optimizer_{}'.format(self._name)):
      summaries.append(tf.summary.scalar('learning_rate', self._learning_rate))
      summaries.append(tf.summary.scalar('grad_norm', grad_norm))
      if self._clipping:
        clipped = tf.minimum(grad_norm, self._clipping)
        summaries.append(tf.summary.scalar('clipped_gradient_norm', clipped))
      summary = tf.summary.merge(summaries)
    return summary
