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

import argparse
import csv
import fnmatch
import functools
import multiprocessing.dummy as multiprocessing
import os
import re
import sys
import traceback

# import imageio
import numpy as np
import skimage.io
import tensorflow as tf
from tensorboard.backend.event_processing import (
    plugin_event_multiplexer as event_multiplexer)


lock = multiprocessing.Lock()


def safe_print(*args, **kwargs):
  with lock:
    print(*args, **kwargs)


def create_reader(logdir):
  reader = event_multiplexer.EventMultiplexer()
  reader.AddRun(logdir, 'run')
  reader.Reload()
  return reader


def extract_values(reader, tag):
  events = reader.Tensors('run', tag)
  steps = [event.step for event in events]
  times = [event.wall_time for event in events]
  values = [tf.make_ndarray(event.tensor_proto) for event in events]
  return steps, times, values


def export_scalar(basename, steps, times, values):
  safe_print('Writing', basename + '.csv')
  values = [value.item() for value in values]
  with tf.gfile.Open(basename + '.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(('wall_time', 'step', 'value'))
    for row in zip(times, steps, values):
      writer.writerow(row)


def export_image(basename, steps, times, values):
  tf.reset_default_graph()
  tf_string = tf.placeholder(tf.string)
  tf_tensor = tf.image.decode_image(tf_string)
  with tf.Session() as sess:
    for step, time_, value in zip(steps, times, values):
      filename = '{}-{}-{}.png'.format(basename, step, time_)
      width, height, string = value[0], value[1], value[2]
      del width
      del height
      tensor = sess.run(tf_tensor, {tf_string: string})
      # imageio.imsave(filename, tensor)
      skimage.io.imsave(filename, tensor)
      filename = '{}-{}-{}.npy'.format(basename, step, time_)
      np.save(filename, tensor)


def process_logdir(logdir, args):
  clean = lambda text: re.sub('[^A-Za-z0-9_]', '_', text)
  basename = os.path.join(args.outdir, clean(logdir))
  if len(tf.gfile.Glob(basename + '*')) > 0 and not args.force:
    safe_print('Exists', logdir)
    return
  try:
    safe_print('Start', logdir)
    reader = create_reader(logdir)
    for tag in reader.Runs()['run']['tensors']:  # tensors -> scalars
      if fnmatch.fnmatch(tag, args.tags):
        steps, times, values = extract_values(reader, tag)
        filename = '{}___{}'.format(basename, clean(tag))
        export_scalar(filename, steps, times, values)
    # for tag in tags['images']:
    #   if fnmatch.fnmatch(tag, args.tags):
    #     steps, times, values = extract_values(reader, tag)
    #     filename = '{}___{}'.format(basename, clean(tag))
    #     export_image(filename, steps, times, values)
    del reader
    safe_print('Done', logdir)
  except Exception:
    safe_print('Exception', logdir)
    safe_print(traceback.print_exc())


def main(args):
  logdirs = tf.gfile.Glob(args.logdirs)
  print(len(logdirs), 'logdirs.')
  assert logdirs
  tf.gfile.MakeDirs(args.outdir)
  np.random.shuffle(logdirs)
  pool = multiprocessing.Pool(args.workers)
  worker_fn = functools.partial(process_logdir, args=args)
  pool.map(worker_fn, logdirs)


if __name__ == '__main__':
  boolean = lambda x: ['False', 'True'].index(x)
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--logdirs', required=True,
      help='glob for log directories to fetch')
  parser.add_argument(
      '--tags', default='trainer*score',
      help='glob for tags to save')
  parser.add_argument(
      '--outdir', required=True,
      help='output directory to store values')
  parser.add_argument(
      '--force', type=boolean, default=False,
      help='overwrite existing files')
  parser.add_argument(
      '--workers', type=int, default=10,
      help='number of worker threads')
  args_, remaining = parser.parse_known_args()
  args_.logdirs = os.path.expanduser(args_.logdirs)
  args_.outdir = os.path.expanduser(args_.outdir)
  remaining.insert(0, sys.argv[0])
  tf.app.run(lambda _: main(args_), remaining)
