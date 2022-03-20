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
import os
import shutil

from matplotlib import animation
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.misc


def create_animation(frames, size, fps=10, **kwargs):
  fig = plt.figure(figsize=size, frameon=False)
  ax = fig.add_axes([0, 0, 1, 1])
  img = ax.imshow(frames[0], **kwargs)
  ax.set_xticks([])
  ax.set_yticks([])
  callback = lambda frame: (img, img.set_data(frame))[:1]
  kwargs = dict(frames=frames, interval=1000 / fps, blit=True)
  anim = animation.FuncAnimation(fig, callback, **kwargs)
  return anim


def save_animation(filepath, anim, fps=10, overwrite=False):
  if filepath.endswith('.mp4'):
    if not shutil.which('ffmpeg'):
      raise RuntimeError("The 'ffmpeg' executable must be in PATH.")
    mpl.rcParams['animation.ffmpeg_path'] = 'ffmpeg'
    kwargs = dict(fps=fps, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])
  elif filepath.endswith('.gif'):
    kwargs = dict(fps=fps, writer='imagemagick')
  else:
    message = "Unknown video format; filename should end in '.mp4' or '.gif'."
    raise NotImplementedError(message)
  if os.path.exists(filepath) and not overwrite:
    message = 'Skip rendering animation because {} already exists.'
    print(message.format(filepath))
    return
  print('Render animation to {}.'.format(filepath))
  anim.save(filepath, **kwargs)


def unpack_image_strip(image, tile_width, tile_height):
  image = image.reshape((
      image.shape[0] // tile_height,
      tile_height,
      image.shape[1] // tile_width,
      tile_width,
      image.shape[2],
  ))
  image = image.transpose((0, 2, 1, 3, 4))
  return image


def pack_animation_frames(image):
  image = image.transpose((1, 2, 0, 3, 4))
  image = image.reshape((
      image.shape[0],
      image.shape[1],
      image.shape[2] * image.shape[3],
      image.shape[4],
  ))
  return image


def main(args):
  image = scipy.misc.imread(args.image_path)
  image = unpack_image_strip(image, args.tile_width, args.tile_height)
  frames = pack_animation_frames(image)
  size = frames.shape[2] / args.dpi, frames.shape[1] / args.dpi
  anim = create_animation(frames, size, args.fps)
  save_animation(args.animation_path, anim, args.fps, args.overwrite)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--image-path', required=True)
  parser.add_argument('-o', '--animation-path', required=True)
  parser.add_argument('-f', '--overwrite', action='store_true', default=False)
  parser.add_argument('-x', '--tile-width', type=int, default=64)
  parser.add_argument('-y', '--tile-height', type=int, default=64)
  parser.add_argument('-r', '--fps', type=int, default=10)
  parser.add_argument('-d', '--dpi', type=float, default=50)
  args = parser.parse_args()
  main(args)
