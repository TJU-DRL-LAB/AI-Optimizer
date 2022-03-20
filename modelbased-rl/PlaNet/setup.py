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

"""Setup script for PlaNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools


setuptools.setup(
    name='planetrl',
    version='1.0.0',
    description=(
        'Deep Planning Network: Control from pixels by latent planning ' +
        'with learned dynamics.'),
    license='Apache 2.0',
    url='http://github.com/google-research/planet',
    install_requires=[
        'dm_control',
        'gym',
        'matplotlib',
        'ruamel.yaml',
        'scikit-image',
        'scipy',
        'tensorflow-gpu==1.13.1',
        'tensorflow_probability==0.6.0',
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
    ],
)
