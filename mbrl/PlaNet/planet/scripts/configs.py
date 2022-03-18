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

import tensorflow as tf

from planet import control
from planet import models
from planet import networks
from planet import tools
from planet.scripts import tasks as tasks_lib
from planet.scripts import objectives as objectives_lib


ACTIVATIONS = {
    'relu': tf.nn.relu,
    'elu': tf.nn.elu,
    'tanh': tf.tanh,
    'swish': lambda x: x * tf.sigmoid(x),
    'softplus': tf.nn.softplus,
    'none': None,
}


def default(config, params):
  config.debug = False
  config.loss_scales = tools.AttrDict(_unlocked=True)
  config = _data_processing(config, params)
  config = _model_components(config, params)
  config = _tasks(config, params)
  config = _loss_functions(config, params)
  config = _training_schedule(config, params)
  return config


def debug(config, params):
  defaults = tools.AttrDict(_unlocked=True)
  defaults.action_repeat = 50
  defaults.num_seed_episodes = 1
  defaults.train_steps = 10
  defaults.test_steps = 10
  defaults.max_steps = 500
  defaults.train_collects = [dict(after=10, every=10)]
  defaults.test_collects = [dict(after=10, every=10)]
  defaults.model_size = 10
  defaults.state_size = 5
  defaults.num_layers = 1
  defaults.num_units = 10
  defaults.batch_shape = [5, 10]
  defaults.loader_every = 5
  defaults.loader_window = 2
  defaults.planner_amount = 5
  defaults.planner_topk = 2
  defaults.planner_iterations = 2
  with params.unlocked:
    for key, value in defaults.items():
      if key not in params:
        params[key] = value
  config = default(config, params)
  config.debug = True
  return config


def _data_processing(config, params):
  config.batch_shape = params.get('batch_shape', (50, 50))
  config.num_chunks = params.get('num_chunks', 1)
  image_bits = params.get('image_bits', 5)
  config.preprocess_fn = tools.bind(
      tools.preprocess.preprocess, bits=image_bits)
  config.postprocess_fn = tools.bind(
      tools.preprocess.postprocess, bits=image_bits)
  config.open_loop_context = 5
  config.data_reader = tools.numpy_episodes.episode_reader
  config.data_loader = {
      'cache': tools.bind(
          tools.numpy_episodes.cache_loader,
          every=params.get('loader_every', 1000)),
      'recent': tools.bind(
          tools.numpy_episodes.recent_loader,
          every=params.get('loader_every', 1000)),
      'reload': tools.numpy_episodes.reload_loader,
      'dummy': tools.numpy_episodes.dummy_loader,
  }[params.get('loader', 'recent')]
  config.bound_action = tools.bind(
      tools.bound_action,
      strategy=params.get('bound_action', 'clip'))
  return config


def _model_components(config, params):
  config.gradient_heads = params.get('gradient_heads', ['image', 'reward'])
  network = getattr(networks, params.get('network', 'conv_ha'))
  config.activation = ACTIVATIONS[params.get('activation', 'relu')]
  config.num_layers = params.get('num_layers', 3)
  config.num_units = params.get('num_units', 300)
  config.head_network = tools.bind(
      networks.feed_forward,
      num_layers=config.num_layers,
      units=config.num_units,
      activation=config.activation)
  config.encoder = network.encoder
  config.decoder = network.decoder
  config.heads = tools.AttrDict(_unlocked=True)
  config.heads.image = config.decoder
  size = params.get('model_size', 200)
  state_size = params.get('state_size', 30)
  model = params.get('model', 'rssm')
  if model == 'ssm':
    config.cell = tools.bind(
        models.SSM, state_size, size,
        params.get('mean_only', False),
        config.activation,
        params.get('min_stddev', 1e-1))
  elif model == 'rssm':
    config.cell = tools.bind(
        models.RSSM, state_size, size, size,
        params.get('future_rnn', True),
        params.get('mean_only', False),
        params.get('min_stddev', 1e-1),
        config.activation,
        params.get('model_layers', 1))
  elif params.model == 'drnn':
    config.cell = tools.bind(
        models.DRNN, state_size, size, size,
        params.get('mean_only', False),
        params.get('min_stddev', 1e-1), config.activation,
        params.get('drnn_encoder_to_decoder', False),
        params.get('drnn_sample_to_sample', True),
        params.get('drnn_sample_to_encoder', True),
        params.get('drnn_decoder_to_encoder', False),
        params.get('drnn_decoder_to_sample', True),
        params.get('drnn_action_to_decoder', False))
  else:
    raise NotImplementedError("Unknown model '{}.".format(params.model))
  return config


def _tasks(config, params):
  tasks = params.get('tasks', ['cheetah_run'])
  tasks = [getattr(tasks_lib, name)(config, params) for name in tasks]
  config.isolate_envs = params.get('isolate_envs', 'thread')
  def common_spaces_ctor(task, action_spaces):
    env = task.env_ctor()
    env = control.wrappers.SelectObservations(env, ['image'])
    env = control.wrappers.PadActions(env, action_spaces)
    return env
  if len(tasks) > 1:
    action_spaces = [task.env_ctor().action_space for task in tasks]
    for index, task in enumerate(tasks):
      env_ctor = tools.bind(common_spaces_ctor, task, action_spaces)
      tasks[index] = tasks_lib.Task(
          task.name, env_ctor, task.max_length, ['reward'])
  for name in tasks[0].state_components:
    if name == 'reward' or params.get('state_diagnostics', False):
      config.heads[name] = tools.bind(
          config.head_network,
          stop_gradient=name not in config.gradient_heads)
      config.loss_scales[name] = 1.0
  config.tasks = tasks
  return config


def _loss_functions(config, params):
  for head in config.gradient_heads:
    assert head in config.heads, head
  config.loss_scales.divergence = params.get('divergence_scale', 1.0)
  config.loss_scales.global_divergence = params.get('global_div_scale', 0.0)
  config.loss_scales.overshooting = params.get('overshooting_scale', 0.0)
  for head in config.heads:
    defaults = {'reward': 10.0}
    scale = defaults[head] if head in defaults else 1.0
    config.loss_scales[head] = params.get(head + '_loss_scale', scale)
  config.free_nats = params.get('free_nats', 3.0)
  config.overshooting_distance = params.get('overshooting_distance', 0)
  config.os_stop_posterior_grad = params.get('os_stop_posterior_grad', True)
  config.optimizers = tools.AttrDict(_unlocked=True)
  config.optimizers.main = tools.bind(
      tools.CustomOptimizer,
      optimizer_cls=tools.bind(tf.train.AdamOptimizer, epsilon=1e-4),
      # schedule=tools.bind(tools.schedule.linear, ramp=0),
      learning_rate=params.get('main_learning_rate', 1e-3),
      clipping=params.get('main_gradient_clipping', 1000.0))
  return config


def _training_schedule(config, params):
  config.train_steps = int(params.get('train_steps', 50000))
  config.test_steps = int(params.get('test_steps', 50))
  config.max_steps = int(params.get('max_steps', 5e7))
  config.train_log_every = config.train_steps
  config.train_checkpoint_every = None
  config.test_checkpoint_every = int(
      params.get('checkpoint_every', 10 * config.test_steps))
  config.checkpoint_to_load = None
  config.savers = [tools.AttrDict(exclude=(r'.*_temporary.*',))]
  config.print_metrics_every = config.train_steps // 10
  config.train_dir = os.path.join(params.logdir, 'train_episodes')
  config.test_dir = os.path.join(params.logdir, 'test_episodes')
  config.random_collects = _initial_collection(config, params)
  config.train_collects = _active_collection(
      params.get('train_collects', [{}]), dict(
          prefix='train',
          save_episode_dir=config.train_dir,
          action_noise=params.get('train_action_noise', 0.3),
      ), config, params)
  config.test_collects = _active_collection(
      params.get('test_collects', [{}]), dict(
          prefix='test',
          save_episode_dir=config.test_dir,
          action_noise=0.0,
      ), config, params)
  return config


def _initial_collection(config, params):
  num_seed_episodes = params.get('num_seed_episodes', 5)
  sims = tools.AttrDict(_unlocked=True)
  for task in config.tasks:
    sims['train-' + task.name] = tools.AttrDict(
        task=task,
        save_episode_dir=config.train_dir,
        num_episodes=num_seed_episodes)
    sims['test-' + task.name] = tools.AttrDict(
        task=task,
        save_episode_dir=config.test_dir,
        num_episodes=num_seed_episodes)
  return sims


def _active_collection(collects, defaults, config, params):
  defs = dict(
      name='main',
      batch_size=1,
      horizon=params.get('planner_horizon', 12),
      objective=params.get('collect_objective', 'reward'),
      after=params.get('collect_every', 5000),
      every=params.get('collect_every', 5000),
      until=-1,
      action_noise=0.0,
      action_noise_ramp=params.get('action_noise_ramp', 0),
      action_noise_min=params.get('action_noise_min', 0.0),
  )
  defs.update(defaults)
  sims = tools.AttrDict(_unlocked=True)
  for task in config.tasks:
    for collect in collects:
      collect = tools.AttrDict(collect, _defaults=defs)
      sim = _define_simulation(
          task, config, params, collect.horizon, collect.batch_size,
          collect.objective)
      sim.unlock()
      sim.save_episode_dir = collect.save_episode_dir
      sim.steps_after = int(collect.after)
      sim.steps_every = int(collect.every)
      sim.steps_until = int(collect.until)
      sim.exploration = tools.AttrDict(
          scale=collect.action_noise,
          schedule=tools.bind(
              tools.schedule.linear,
              ramp=collect.action_noise_ramp,
              min=collect.action_noise_min,
          ))
      name = '{}_{}_{}'.format(collect.prefix, collect.name, task.name)
      assert name not in sims, (set(sims.keys()), name)
      sims[name] = sim
      assert not collect.untouched, collect.untouched
  return sims


def _define_simulation(
    task, config, params, horizon, batch_size, objective='reward',
    rewards=False):
  planner = params.get('planner', 'cem')
  if planner == 'cem':
    planner_fn = tools.bind(
        control.planning.cross_entropy_method,
        amount=params.get('planner_amount', 1000),
        iterations=params.get('planner_iterations', 10),
        topk=params.get('planner_topk', 100),
        horizon=horizon)
  else:
    raise NotImplementedError(planner)
  return tools.AttrDict(
      task=task,
      num_agents=batch_size,
      planner=planner_fn,
      objective=tools.bind(getattr(objectives_lib, objective), params=params))
