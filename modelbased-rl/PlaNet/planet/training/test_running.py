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
import pickle
import threading
import time

import numpy as np
import tensorflow as tf

from planet.training import running


class TestExperiment(tf.test.TestCase):

  def test_no_kills(self):
    tf.logging.set_verbosity(tf.logging.INFO)
    basedir = os.path.join(tf.test.get_temp_dir(), 'test_no_kills')
    processes = []
    for worker_name in range(20):
      processes.append(threading.Thread(
          target=_worker_normal, args=(basedir, str(worker_name))))
      processes[-1].start()
    for process in processes:
      process.join()
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/DONE'))
    self.assertEqual(100, len(filepaths))
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/PING'))
    self.assertEqual(100, len(filepaths))
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/started'))
    self.assertEqual(100, len(filepaths))
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/resumed'))
    self.assertEqual(0, len(filepaths))
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/failed'))
    self.assertEqual(0, len(filepaths))
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/numbers'))
    self.assertEqual(100, len(filepaths))
    for filepath in filepaths:
      with tf.gfile.GFile(filepath, 'rb') as file_:
        self.assertEqual(10, len(pickle.load(file_)))

  def test_dying_workers(self):
    tf.logging.set_verbosity(tf.logging.INFO)
    basedir = os.path.join(tf.test.get_temp_dir(), 'test_dying_workers')
    processes = []
    for worker_name in range(20):
      processes.append(threading.Thread(
          target=_worker_dying, args=(basedir, 15, str(worker_name))))
      processes[-1].start()
    for process in processes:
      process.join()
    processes = []
    for worker_name in range(20):
      processes.append(threading.Thread(
          target=_worker_normal, args=(basedir, str(worker_name))))
      processes[-1].start()
    for process in processes:
      process.join()
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/DONE'))
    self.assertEqual(100, len(filepaths))
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/PING'))
    self.assertEqual(100, len(filepaths))
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/FAIL'))
    self.assertEqual(0, len(filepaths))
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/started'))
    self.assertEqual(100, len(filepaths))
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/resumed'))
    self.assertEqual(20, len(filepaths))
    filepaths = tf.gfile.Glob(os.path.join(basedir, '*/numbers'))
    self.assertEqual(100, len(filepaths))
    for filepath in filepaths:
      with tf.gfile.GFile(filepath, 'rb') as file_:
        self.assertEqual(10, len(pickle.load(file_)))


def _worker_normal(basedir, worker_name):
  experiment = running.Experiment(
      basedir, _process_fn, _start_fn, _resume_fn,
      num_runs=100, worker_name=worker_name, ping_every=1.0)
  for run in experiment:
    for score in run:
      pass


def _worker_dying(basedir, die_at_step, worker_name):
  experiment = running.Experiment(
      basedir, _process_fn, _start_fn, _resume_fn,
      num_runs=100, worker_name=worker_name, ping_every=1.0)
  step = 0
  for run in experiment:
    for score in run:
      step += 1
      if step >= die_at_step:
        return


def _start_fn(logdir):
  assert not tf.gfile.Exists(os.path.join(logdir, 'DONE'))
  assert not tf.gfile.Exists(os.path.join(logdir, 'started'))
  assert not tf.gfile.Exists(os.path.join(logdir, 'resumed'))
  with tf.gfile.GFile(os.path.join(logdir, 'started'), 'w') as file_:
    file_.write('\n')
  with tf.gfile.GFile(os.path.join(logdir, 'numbers'), 'wb') as file_:
    pickle.dump([], file_)
  return []


def _resume_fn(logdir):
  assert not tf.gfile.Exists(os.path.join(logdir, 'DONE'))
  assert tf.gfile.Exists(os.path.join(logdir, 'started'))
  with tf.gfile.GFile(os.path.join(logdir, 'resumed'), 'w') as file_:
    file_.write('\n')
  with tf.gfile.GFile(os.path.join(logdir, 'numbers'), 'rb') as file_:
    numbers = pickle.load(file_)
  if len(numbers) != 5:
    raise Exception('Expected to be resumed in the middle for this test.')
  return numbers


def _process_fn(logdir, numbers):
  assert tf.gfile.Exists(os.path.join(logdir, 'started'))
  while len(numbers) < 10:
    number = np.random.uniform(0, 0.1)
    time.sleep(number)
    numbers.append(number)
    with tf.gfile.GFile(os.path.join(logdir, 'numbers'), 'wb') as file_:
      pickle.dump(numbers, file_)
    yield number


if __name__ == '__main__':
  tf.test.main()
