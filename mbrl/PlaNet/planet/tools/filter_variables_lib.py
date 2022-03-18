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

import re

import tensorflow as tf


def filter_variables(include=None, exclude=None):
  # Check arguments.
  if include is None:
    include = (r'.*',)
  if exclude is None:
    exclude = ()
  if not isinstance(include, (tuple, list)):
    include = (include,)
  if not isinstance(exclude, (tuple, list)):
    exclude = (exclude,)
  # Compile regexes.
  include = [re.compile(regex) for regex in include]
  exclude = [re.compile(regex) for regex in exclude]
  variables = tf.global_variables()
  if not variables:
    raise RuntimeError('There are no variables to filter.')
  # Check regexes.
  for regex in include:
    message = "Regex r'{}' does not match any variables in the graph.\n"
    message += 'All variables:\n'
    message += '\n'.join('- {}'.format(var.name) for var in variables)
    if not any(regex.match(variable.name) for variable in variables):
      raise RuntimeError(message.format(regex.pattern))
  # Filter variables.
  filtered = []
  for variable in variables:
    if not any(regex.match(variable.name) for regex in include):
      continue
    if any(regex.match(variable.name) for regex in exclude):
      continue
    filtered.append(variable)
  # Check result.
  if not filtered:
    message = 'No variables left after filtering.'
    message += '\nIncludes:\n' + '\n'.join(regex.pattern for regex in include)
    message += '\nExcludes:\n' + '\n'.join(regex.pattern for regex in exclude)
    raise RuntimeError(message)
  return filtered
