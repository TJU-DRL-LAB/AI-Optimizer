from distutils.core import setup
from setuptools import find_packages

setup(
    name='mbpo',
    packages=find_packages(),
    version='0.0.1',
    description='Model-based policy optimization',
    long_description=open('./README.md').read(),
    author='Michael Janner',
    author_email='janner@berkeley.edu',
    url='https://people.eecs.berkeley.edu/~janner/mbpo/',
    entry_points={
        'console_scripts': (
            'mbpo=softlearning.scripts.console_scripts:main',
            'viskit=mbpo.scripts.console_scripts:main'
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)
