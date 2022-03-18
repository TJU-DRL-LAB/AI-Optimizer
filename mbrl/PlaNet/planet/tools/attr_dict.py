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

import collections
import contextlib
import os

import numpy as np
import ruamel.yaml as yaml


class AttrDict(dict):  # collections.OrderedDict
  """Wrap a dictionary to access keys as attributes."""

  def __init__(self, *args, **kwargs):
    unlocked = kwargs.pop('_unlocked', not (args or kwargs))
    defaults = kwargs.pop('_defaults', {})
    touched = kwargs.pop('_touched', set())
    super(AttrDict, self).__setattr__('_unlocked', True)
    super(AttrDict, self).__setattr__('_touched', set())
    super(AttrDict, self).__setattr__('_defaults', {})
    super(AttrDict, self).__init__(*args, **kwargs)
    super(AttrDict, self).__setattr__('_unlocked', unlocked)
    super(AttrDict, self).__setattr__('_defaults', defaults)
    super(AttrDict, self).__setattr__('_touched', touched)

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)

  def __setattr__(self, name, value):
    self[name] = value

  def __getitem__(self, name):
    # Do not provide None for unimplemented magic attributes.
    # if name.startswith('__'):
    #   raise AttributeError(name)
    self._touched.add(name)
    if name in self:
      return super(AttrDict, self).__getitem__(name)
    if name in self._defaults:
      return self._defaults[name]
    raise AttributeError(name)

  def __setitem__(self, name, value):
    # if name.startswith('_'):
    #   raise AttributeError('Cannot set private attribute {}'.format(name))
    if name.startswith('__'):
      raise AttributeError("Cannot set magic attribute '{}'".format(name))
    if not self._unlocked:
      message = 'Use obj.unlock() before setting {}'
      raise RuntimeError(message.format(name))
    super(AttrDict, self).__setitem__(name, value)

  def __repr__(self):
    items = []
    for key, value in self.items():
      items.append('{}: {}'.format(key, self._format_value(value)))
    return '{' + ', '.join(items) + '}'

  def get(self, key, default=None):
    self._touched.add(key)
    if key not in self:
      return default
    return self[key]

  @property
  def untouched(self):
    return sorted(set(self.keys()) - self._touched)

  @property
  @contextlib.contextmanager
  def unlocked(self):
    self.unlock()
    yield
    self.lock()

  def lock(self):
    super(AttrDict, self).__setattr__('_unlocked', False)
    for value in self.values():
      if isinstance(value, AttrDict):
        value.lock()

  def unlock(self):
    super(AttrDict, self).__setattr__('_unlocked', True)
    for value in self.values():
      if isinstance(value, AttrDict):
        value.unlock()

  def summarize(self):
    items = []
    for key, value in self.items():
      items.append('{}: {}'.format(key, self._format_value(value)))
    return '\n'.join(items)

  def update(self, mapping):
    if not self._unlocked:
      message = 'Use obj.unlock() before updating'
      raise RuntimeError(message)
    super(AttrDict, self).update(mapping)
    return self

  def copy(self, _unlocked=False):
    return type(self)(super(AttrDict, self).copy(), _unlocked=_unlocked)

  def save(self, filename):
    assert str(filename).endswith('.yaml')
    directory = os.path.dirname(str(filename))
    os.makedirs(directory, exist_ok=True)
    with open(filename, 'w') as f:
      yaml.dump(collections.OrderedDict(self), f)

  @classmethod
  def load(cls, filename):
    assert str(filename).endswith('.yaml')
    with open(filename, 'r') as f:
      return cls(yaml.load(f, Loader=yaml.Loader))

  def _format_value(self, value):
    if isinstance(value, np.ndarray):
      template = '<np.array shape={} dtype={} min={} mean={} max={}>'
      min_ = self._format_value(value.min())
      mean = self._format_value(value.mean())
      max_ = self._format_value(value.max())
      return template.format(value.shape, value.dtype, min_, mean, max_)
    if isinstance(value, float) and 1e-3 < abs(value) < 1e6:
      return '{:.3f}'.format(value)
    if isinstance(value, float):
      return '{:4.1e}'.format(value)
    if hasattr(value, '__name__'):
      return value.__name__
    return str(value)
