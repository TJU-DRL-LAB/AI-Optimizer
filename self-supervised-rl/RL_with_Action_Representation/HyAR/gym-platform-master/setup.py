from setuptools import setup

setup(
    name='gym_platform',
    version='0.0.1',
    description='Platform domain OpenAI Gym environment',
    author='Craig Bester',
    packages=['gym_platform'],
    include_package_data=True, # include asset files. TODO: specify specific files?
    install_requires=['gym',
                      'pygame', #'pygame>=1.9.3'
                      'numpy',  #'numpy>=1.14.0'
    ]
) 
