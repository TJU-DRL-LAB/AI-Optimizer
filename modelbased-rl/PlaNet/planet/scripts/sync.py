#!/usr/bin/python3

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

import argparse
import glob
import os
import shutil


def find_source_files(directory):
  top_level = glob.glob(directory + '*.py')
  recursive = glob.glob(directory + '/**/*.py')
  return top_level + recursive


def copy_source_tree(source_dir, target_dir):
  for source in find_source_files(source_dir):
    target = os.path.join(target_dir, os.path.relpath(source, source_dir))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target):
      print('Override', os.path.relpath(target, target_dir))
    else:
      print('Add', os.path.relpath(target, target_dir))
    shutil.copy(source, target)
  for target in find_source_files(target_dir):
    source = os.path.join(source_dir, os.path.relpath(target, target_dir))
    if not os.path.exists(source):
      print('Remove', os.path.relpath(target, target_dir))
      os.remove(target)


def infer_headers(directory):
  try:
    filename = find_source_files(directory)[0]
  except IndexError:
    raise RuntimeError('No code files found in {}.'.format(directory))
  header = []
  with open(filename, 'r') as f:
    for index, line in enumerate(f):
      if index == 0 and not line.startswith('#'):
        break
      if not line.startswith('#') and line.strip(' \n'):
        break
      header.append(line)
  return header


def add_headers(directory, header):
  for filename in find_source_files(directory):
    with open(filename, 'r') as f:
      text = f.readlines()
    with open(filename, 'w') as f:
      f.write(''.join(header + text))


def remove_headers(directory, header):
  for filename in find_source_files(directory):
    with open(filename, 'r') as f:
      text = f.readlines()
    if text[:len(header)] == header:
      text = text[len(header):]
    with open(filename, 'w') as f:
      f.write(''.join(text))


def main(args):
  print('Inferring headers.\n')
  source_header = infer_headers(args.source)
  print('{} Source header {}\n{}{}\n'.format(
      '-' * 32, '-' * 32, ''.join(source_header), '-' * 79))
  target_header = infer_headers(args.target)
  print('{} Target header {}\n{}{}\n'.format(
      '-' * 32, '-' * 32, ''.join(target_header), '-' * 79))
  print('Synchronizing directories.')
  copy_source_tree(args.source, args.target)
  if target_header and not source_header:
    print('Adding headers.')
    add_headers(args.target, target_header)
  if source_header and not target_header:
    print('Removing headers.')
    remove_headers(args.target, source_header)
  print('Done.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--source', type=os.path.expanduser, required=True)
  parser.add_argument('--target', type=os.path.expanduser, required=True)
  main(parser.parse_args())
