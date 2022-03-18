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

import tensorflow as tf

from planet.tools import nested


class ZipTest(tf.test.TestCase):

  def test_scalar(self):
    self.assertEqual(42, nested.zip(42))
    self.assertEqual((13, 42), nested.zip(13, 42))

  def test_empty(self):
    self.assertEqual({}, nested.zip({}, {}))

  def test_base_case(self):
    self.assertEqual((1, 2, 3), nested.zip(1, 2, 3))

  def test_shallow_list(self):
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    result = nested.zip(a, b, c)
    self.assertEqual([(1, 4, 7), (2, 5, 8), (3, 6, 9)], result)

  def test_shallow_tuple(self):
    a = (1, 2, 3)
    b = (4, 5, 6)
    c = (7, 8, 9)
    result = nested.zip(a, b, c)
    self.assertEqual(((1, 4, 7), (2, 5, 8), (3, 6, 9)), result)

  def test_shallow_dict(self):
    a = {'a': 1, 'b': 2, 'c': 3}
    b = {'a': 4, 'b': 5, 'c': 6}
    c = {'a': 7, 'b': 8, 'c': 9}
    result = nested.zip(a, b, c)
    self.assertEqual({'a': (1, 4, 7), 'b': (2, 5, 8), 'c': (3, 6, 9)}, result)

  def test_single(self):
    a = [[1, 2], 3]
    result = nested.zip(a)
    self.assertEqual(a, result)

  def test_mixed_structures(self):
    a = [(1, 2), 3, {'foo': [4]}]
    b = [(5, 6), 7, {'foo': [8]}]
    result = nested.zip(a, b)
    self.assertEqual([((1, 5), (2, 6)), (3, 7), {'foo': [(4, 8)]}], result)

  def test_different_types(self):
    a = [1, 2, 3]
    b = 'a b c'.split()
    result = nested.zip(a, b)
    self.assertEqual([(1, 'a'), (2, 'b'), (3, 'c')], result)

  def test_use_type_of_first(self):
    a = (1, 2, 3)
    b = [4, 5, 6]
    c = [7, 8, 9]
    result = nested.zip(a, b, c)
    self.assertEqual(((1, 4, 7), (2, 5, 8), (3, 6, 9)), result)

  def test_namedtuple(self):
    Foo = collections.namedtuple('Foo', 'value')
    foo, bar = Foo(42), Foo(13)
    self.assertEqual(Foo((42, 13)), nested.zip(foo, bar))


class MapTest(tf.test.TestCase):

  def test_scalar(self):
    self.assertEqual(42, nested.map(lambda x: x, 42))

  def test_empty(self):
    self.assertEqual({}, nested.map(lambda x: x, {}))

  def test_shallow_list(self):
    self.assertEqual([2, 4, 6], nested.map(lambda x: 2 * x, [1, 2, 3]))

  def test_shallow_dict(self):
    data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    self.assertEqual(data, nested.map(lambda x: x, data))

  def test_mixed_structure(self):
    structure = [(1, 2), 3, {'foo': [4]}]
    result = nested.map(lambda x: 2 * x, structure)
    self.assertEqual([(2, 4), 6, {'foo': [8]}], result)

  def test_mixed_types(self):
    self.assertEqual([14, 'foofoo'], nested.map(lambda x: x * 2, [7, 'foo']))

  def test_multiple_lists(self):
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    result = nested.map(lambda x, y, z: x + y + z, a, b, c)
    self.assertEqual([12, 15, 18], result)

  def test_namedtuple(self):
    Foo = collections.namedtuple('Foo', 'value')
    foo, bar = [Foo(42)], [Foo(13)]
    function = nested.map(lambda x, y: (y, x), foo, bar)
    self.assertEqual([Foo((13, 42))], function)
    function = nested.map(lambda x, y: x + y, foo, bar)
    self.assertEqual([Foo(55)], function)


class FlattenTest(tf.test.TestCase):

  def test_scalar(self):
    self.assertEqual((42,), nested.flatten(42))

  def test_empty(self):
    self.assertEqual((), nested.flatten({}))

  def test_base_case(self):
    self.assertEqual((1,), nested.flatten(1))

  def test_convert_type(self):
    self.assertEqual((1, 2, 3), nested.flatten([1, 2, 3]))

  def test_mixed_structure(self):
    self.assertEqual((1, 2, 3, 4), nested.flatten([(1, 2), 3, {'foo': [4]}]))

  def test_value_ordering(self):
    self.assertEqual((1, 2, 3), nested.flatten({'a': 1, 'b': 2, 'c': 3}))


class FilterTest(tf.test.TestCase):

  def test_empty(self):
    self.assertEqual({}, nested.filter(lambda x: True, {}))
    self.assertEqual({}, nested.filter(lambda x: False, {}))

  def test_base_case(self):
    self.assertEqual((), nested.filter(lambda x: False, 1))

  def test_single_dict(self):
    predicate = lambda x: x % 2 == 0
    data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    self.assertEqual({'b': 2, 'd': 4}, nested.filter(predicate, data))

  def test_multiple_lists(self):
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [7, 8, 9]
    predicate = lambda *args: any(x % 4 == 0 for x in args)
    result = nested.filter(predicate, a, b, c)
    self.assertEqual([(1, 4, 7), (2, 5, 8)], result)

  def test_multiple_dicts(self):
    a = {'a': 1, 'b': 2, 'c': 3}
    b = {'a': 4, 'b': 5, 'c': 6}
    c = {'a': 7, 'b': 8, 'c': 9}
    predicate = lambda *args: any(x % 4 == 0 for x in args)
    result = nested.filter(predicate, a, b, c)
    self.assertEqual({'a': (1, 4, 7), 'b': (2, 5, 8)}, result)

  def test_mixed_structure(self):
    predicate = lambda x: x % 2 == 0
    data = [(1, 2), 3, {'foo': [4]}]
    self.assertEqual([(2,), {'foo': [4]}], nested.filter(predicate, data))

  def test_remove_empty_containers(self):
    data = [(1, 2, 3), 4, {'foo': [5, 6], 'bar': 7}]
    self.assertEqual([], nested.filter(lambda x: False, data))

  def test_namedtuple(self):
    Foo = collections.namedtuple('Foo', 'value1, value2')
    self.assertEqual(Foo(1, None), nested.filter(lambda x: x == 1, Foo(1, 2)))

  def test_namedtuple_multiple(self):
    Foo = collections.namedtuple('Foo', 'value1, value2')
    foo = Foo(1, 2)
    bar = Foo(2, 3)
    result = nested.filter(lambda x, y: x + y > 3, foo, bar)
    self.assertEqual(Foo(None, (2, 3)), result)

  def test_namedtuple_nested(self):
    Foo = collections.namedtuple('Foo', 'value1, value2')
    foo = Foo(1, [1, 2, 3])
    self.assertEqual(Foo(None, [2, 3]), nested.filter(lambda x: x > 1, foo))
