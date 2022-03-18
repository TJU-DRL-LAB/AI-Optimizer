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

r"""Train a Deep Planning Network agent.

Full training run:

python3 -m planet.scripts.train \
    --logdir /path/to/logdir \
    --config default \
    --params '{tasks: [cheetah_run]}'

For debugging:

python3 -m planet.scripts.train \
    --logdir /path/to/logdir \
    --resume_runs False \
    --num_runs 1000 \
    --config debug \
    --params '{tasks: [cheetah_run]}'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# Need offline backend to render summaries from within tf.py_func.
import matplotlib
matplotlib.use('Agg')

import ruamel.yaml as yaml
import tensorflow as tf

from planet import tools
from planet import training
from planet.scripts import configs


def process(logdir, args):
  with args.params.unlocked:
    args.params.logdir = logdir
  config = tools.AttrDict()
  with config.unlocked:
    config = getattr(configs, args.config)(config, args.params)
  training.utility.collect_initial_episodes(config)
  tf.reset_default_graph()
  dataset = tools.numpy_episodes.numpy_episodes(
      config.train_dir, config.test_dir, config.batch_shape,
      reader=config.data_reader,
      loader=config.data_loader,
      num_chunks=config.num_chunks,
      preprocess_fn=config.preprocess_fn)
  for score in training.utility.train(
      training.define_model, dataset, logdir, config):
    yield score


def main(args):
  training.utility.set_up_logging()
  experiment = training.Experiment(
      args.logdir,
      process_fn=functools.partial(process, args=args),
      num_runs=args.num_runs,
      ping_every=args.ping_every,
      resume_runs=args.resume_runs)
  for run in experiment:
    for unused_score in run:
      pass


if __name__ == '__main__':
  boolean = lambda x: bool(['False', 'True'].index(x))
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--logdir', required=True)
  parser.add_argument(
      '--num_runs', type=int, default=1)
  parser.add_argument(
      '--config', default='default',
      help='Select a configuration function from scripts/configs.py.')
  parser.add_argument(
      '--params', default='{}',
      help='YAML formatted dictionary to be used by the config.')
  parser.add_argument(
      '--ping_every', type=int, default=0,
      help='Used to prevent conflicts between multiple workers; 0 to disable.')
  parser.add_argument(
      '--resume_runs', type=boolean, default=True,
      help='Whether to resume unfinished runs in the log directory.')
  args_, remaining = parser.parse_known_args()
  args_.params = tools.AttrDict(yaml.safe_load(args_.params.replace('#', ',')))
  args_.logdir = args_.logdir and os.path.expanduser(args_.logdir)
  remaining.insert(0, sys.argv[0])
  tf.app.run(lambda _: main(args_), remaining)
