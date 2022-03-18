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

import gym
import numpy as np


class DummyEnv(object):

  def __init__(self):
    self._random = np.random.RandomState(seed=0)
    self._step = None

  @property
  def observation_space(self):
    low = np.zeros([64, 64, 3], dtype=np.float32)
    high = np.ones([64, 64, 3], dtype=np.float32)
    spaces = {'image': gym.spaces.Box(low, high)}
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    low = -np.ones([5], dtype=np.float32)
    high = np.ones([5], dtype=np.float32)
    return gym.spaces.Box(low, high)

  def reset(self):
    self._step = 0
    obs = self.observation_space.sample()
    return obs

  def step(self, action):
    obs = self.observation_space.sample()
    reward = self._random.uniform(0, 1)
    self._step += 1
    done = self._step >= 1000
    info = {}
    return obs, reward, done, info
