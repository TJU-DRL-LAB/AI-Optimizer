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
from tensorflow_probability import distributions as tfd

from planet import tools
from planet.training import utility
from planet.tools import summary


def define_summaries(graph, config, cleanups):
  summaries = []
  plot_summaries = []  # Control dependencies for non thread-safe matplot.
  length = graph.data['length']
  mask = tf.range(graph.embedded.shape[1].value)[None, :] < length[:, None]
  heads = graph.heads.copy()
  last_time = tf.Variable(lambda: tf.timestamp(), trainable=False)
  last_step = tf.Variable(lambda: 0.0, trainable=False, dtype=tf.float64)

  def transform(dist):
    mean = config.postprocess_fn(dist.mean())
    mean = tf.clip_by_value(mean, 0.0, 1.0)
    return tfd.Independent(tfd.Normal(mean, 1.0), len(dist.event_shape))
  heads.unlock()
  heads['image'] = lambda features: transform(graph.heads['image'](features))
  heads.lock()

  with tf.variable_scope('general'):
    summaries += summary.data_summaries(graph.data, config.postprocess_fn)
    summaries += summary.dataset_summaries(config.train_dir)
    summaries += summary.objective_summaries(graph.objectives)
    summaries.append(tf.summary.scalar('step', graph.step))
    new_time, new_step = tf.timestamp(), tf.cast(graph.global_step, tf.float64)
    delta_time, delta_step = new_time - last_time, new_step - last_step
    with tf.control_dependencies([delta_time, delta_step]):
      assign_ops = [last_time.assign(new_time), last_step.assign(new_step)]
      with tf.control_dependencies(assign_ops):
        summaries.append(tf.summary.scalar(
            'steps_per_second', delta_step / delta_time))
        summaries.append(tf.summary.scalar(
            'seconds_per_step', delta_time / delta_step))

  with tf.variable_scope('closedloop'):
    prior, posterior = tools.unroll.closed_loop(
        graph.cell, graph.embedded, graph.data['action'], config.debug)
    summaries += summary.state_summaries(graph.cell, prior, posterior, mask)
    with tf.variable_scope('prior'):
      prior_features = graph.cell.features_from_state(prior)
      prior_dists = {
          name: head(prior_features)
          for name, head in heads.items()}
      summaries += summary.dist_summaries(prior_dists, graph.data, mask)
      summaries += summary.image_summaries(
          prior_dists['image'], config.postprocess_fn(graph.data['image']))
    with tf.variable_scope('posterior'):
      posterior_features = graph.cell.features_from_state(posterior)
      posterior_dists = {
          name: head(posterior_features)
          for name, head in heads.items()}
      summaries += summary.dist_summaries(
          posterior_dists, graph.data, mask)
      summaries += summary.image_summaries(
          posterior_dists['image'],
          config.postprocess_fn(graph.data['image']))

  with tf.variable_scope('openloop'):
    state = tools.unroll.open_loop(
        graph.cell, graph.embedded, graph.data['action'],
        config.open_loop_context, config.debug)
    state_features = graph.cell.features_from_state(state)
    state_dists = {name: head(state_features) for name, head in heads.items()}
    summaries += summary.dist_summaries(state_dists, graph.data, mask)
    summaries += summary.image_summaries(
        state_dists['image'], config.postprocess_fn(graph.data['image']))
    summaries += summary.state_summaries(graph.cell, state, posterior, mask)
    with tf.control_dependencies(plot_summaries):
      plot_summary = summary.prediction_summaries(
          state_dists, graph.data, state)
      plot_summaries += plot_summary
      summaries += plot_summary

  with tf.variable_scope('simulation'):
    sim_returns = []
    for name, params in config.test_collects.items():
      # These are expensive and equivalent for train and test phases, so only
      # do one of them.
      sim_summary, sim_return = tf.cond(
          tf.equal(graph.phase, 'test'),
          lambda: utility.simulate_episodes(
              config, params, graph, cleanups,
              expensive_summaries=False,
              gif_summary=True,
              name=name),
          lambda: ('', 0.0),
          name='should_simulate_' + params.task.name)
      summaries.append(sim_summary)
      sim_returns.append(sim_return)

  summaries = tf.summary.merge(summaries)
  score = tf.reduce_mean(sim_returns)[None]
  return summaries, score
