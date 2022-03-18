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

"""Tools for manipulating nested tuples, list, and dictionaries."""

from __future__ import absolute_import
from __future__ import division

_builtin_zip = zip
_builtin_map = map
_builtin_filter = filter


def zip_(*structures, **kwargs):
  """Combine corresponding elements in multiple nested structure to tuples.

  The nested structures can consist of any combination of lists, tuples, and
  dicts. All provided structures must have the same nesting.

  Args:
    *structures: Nested structures.
    flatten: Whether to flatten the resulting structure into a tuple. Keys of
        dictionaries will be discarded.

  Returns:
    Nested structure.
  """
  # Named keyword arguments are not allowed after *args in Python 2.
  flatten = kwargs.pop('flatten', False)
  assert not kwargs, 'zip() got unexpected keyword arguments.'
  return map(
      lambda *x: x if len(x) > 1 else x[0],
      *structures,
      flatten=flatten)


def map_(function, *structures, **kwargs):
  """Apply a function to every element in a nested structure.

  If multiple structures are provided as input, their structure must match and
  the function will be applied to corresponding groups of elements. The nested
  structure can consist of any combination of lists, tuples, and dicts.

  Args:
    function: The function to apply to the elements of the structure. Receives
        one argument for every structure that is provided.
    *structures: One of more nested structures.
    flatten: Whether to flatten the resulting structure into a tuple. Keys of
        dictionaries will be discarded.

  Returns:
    Nested structure.
  """
  # Named keyword arguments are not allowed after *args in Python 2.
  flatten = kwargs.pop('flatten', False)
  assert not kwargs, 'map() got unexpected keyword arguments.'

  def impl(function, *structures):
    if len(structures) == 0:
      return structures
    if all(isinstance(s, (tuple, list)) for s in structures):
      if len(set(len(x) for x in structures)) > 1:
        raise ValueError('Cannot merge tuples or lists of different length.')
      args = tuple((impl(function, *x) for x in _builtin_zip(*structures)))
      if hasattr(structures[0], '_fields'):  # namedtuple
        return type(structures[0])(*args)
      else:  # tuple, list
        return type(structures[0])(args)
    if all(isinstance(s, dict) for s in structures):
      if len(set(frozenset(x.keys()) for x in structures)) > 1:
        raise ValueError('Cannot merge dicts with different keys.')
      merged = {
          k: impl(function, *(s[k] for s in structures))
          for k in structures[0]}
      return type(structures[0])(merged)
    return function(*structures)

  result = impl(function, *structures)
  if flatten:
    result = flatten_(result)
  return result


def flatten_(structure):
  """Combine all leaves of a nested structure into a tuple.

  The nested structure can consist of any combination of tuples, lists, and
  dicts. Dictionary keys will be discarded but values will ordered by the
  sorting of the keys.

  Args:
    structure: Nested structure.

  Returns:
    Flat tuple.
  """
  if isinstance(structure, dict):
    result = ()
    for key in sorted(list(structure.keys())):
        result += flatten_(structure[key])
    return result
  if isinstance(structure, (tuple, list)):
    result = ()
    for element in structure:
      result += flatten_(element)
    return result
  return (structure,)


def filter_(predicate, *structures, **kwargs):
  """Select elements of a nested structure based on a predicate function.

  If multiple structures are provided as input, their structure must match and
  the function will be applied to corresponding groups of elements. The nested
  structure can consist of any combination of lists, tuples, and dicts.

  Args:
    predicate: The function to determine whether an element should be kept.
        Receives one argument for every structure that is provided.
    *structures: One of more nested structures.
    flatten: Whether to flatten the resulting structure into a tuple. Keys of
        dictionaries will be discarded.

  Returns:
    Nested structure.
  """
  # Named keyword arguments are not allowed after *args in Python 2.
  flatten = kwargs.pop('flatten', False)
  assert not kwargs, 'filter() got unexpected keyword arguments.'

  def impl(predicate, *structures):
    if len(structures) == 0:
      return structures
    if all(isinstance(s, (tuple, list)) for s in structures):
      if len(set(len(x) for x in structures)) > 1:
        raise ValueError('Cannot merge tuples or lists of different length.')
      # Only wrap in tuples if more than one structure provided.
      if len(structures) > 1:
        filtered = (impl(predicate, *x) for x in _builtin_zip(*structures))
      else:
        filtered = (impl(predicate, x) for x in structures[0])
      # Remove empty containers and construct result structure.
      if hasattr(structures[0], '_fields'):  # namedtuple
        filtered = (x if x != () else None for x in filtered)
        return type(structures[0])(*filtered)
      else:  # tuple, list
        filtered = (
            x for x in filtered if not isinstance(x, (tuple, list, dict)) or x)
        return type(structures[0])(filtered)
    if all(isinstance(s, dict) for s in structures):
      if len(set(frozenset(x.keys()) for x in structures)) > 1:
        raise ValueError('Cannot merge dicts with different keys.')
      # Only wrap in tuples if more than one structure provided.
      if len(structures) > 1:
        filtered = {
            k: impl(predicate, *(s[k] for s in structures))
            for k in structures[0]}
      else:
        filtered = {k: impl(predicate, v) for k, v in structures[0].items()}
      # Remove empty containers and construct result structure.
      filtered = {
          k: v for k, v in filtered.items()
          if not isinstance(v, (tuple, list, dict)) or v}
      return type(structures[0])(filtered)
    if len(structures) > 1:
      return structures if predicate(*structures) else ()
    else:
      return structures[0] if predicate(structures[0]) else ()

  result = impl(predicate, *structures)
  if flatten:
    result = flatten_(result)
  return result


zip = zip_
map = map_
flatten = flatten_
filter = filter_
