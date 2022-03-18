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
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import tensorflow as tf

from planet import tools
from planet.scripts import train


class PlanetTest(tf.test.TestCase):

  def test_dummy_isolate_none(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        config='debug',
        params=tools.AttrDict(
            task='dummy',
            isolate_envs='none',
            max_steps=30),
        ping_every=0,
        resume_runs=False)
    try:
      tf.app.run(lambda _: train.main(args), [sys.argv[0]])
    except SystemExit:
      pass

  def test_dummy_isolate_thread(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        config='debug',
        params=tools.AttrDict(
            task='dummy',
            isolate_envs='thread',
            max_steps=30),
        ping_every=0,
        resume_runs=False)
    try:
      tf.app.run(lambda _: train.main(args), [sys.argv[0]])
    except SystemExit:
      pass

  def test_dummy_isolate_process(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        config='debug',
        params=tools.AttrDict(
            task='dummy',
            isolate_envs='process',
            max_steps=30),
        ping_every=0,
        resume_runs=False)
    try:
      tf.app.run(lambda _: train.main(args), [sys.argv[0]])
    except SystemExit:
      pass

  def test_dm_control_isolate_none(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        config='debug',
        params=tools.AttrDict(
            task='cup_catch',
            isolate_envs='none',
            max_steps=30),
        ping_every=0,
        resume_runs=False)
    try:
      tf.app.run(lambda _: train.main(args), [sys.argv[0]])
    except SystemExit:
      pass

  def test_dm_control_isolate_thread(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        config='debug',
        params=tools.AttrDict(
            task='cup_catch',
            isolate_envs='thread',
            max_steps=30),
        ping_every=0,
        resume_runs=False)
    try:
      tf.app.run(lambda _: train.main(args), [sys.argv[0]])
    except SystemExit:
      pass

  def test_dm_control_isolate_process(self):
    args = tools.AttrDict(
        logdir=self.get_temp_dir(),
        num_runs=1,
        config='debug',
        params=tools.AttrDict(
            task='cup_catch',
            isolate_envs='process',
            max_steps=30),
        ping_every=0,
        resume_runs=False)
    try:
      tf.app.run(lambda _: train.main(args), [sys.argv[0]])
    except SystemExit:
      pass


if __name__ == '__main__':
  tf.test.main()
