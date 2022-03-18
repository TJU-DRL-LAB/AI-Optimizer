from distutils.core import setup
from setuptools import find_packages

setup(
    name='ed2-mbpo',
    packages=find_packages(),
    version='0.0.1',
    description='ED2-Model-based policy optimization',
    long_description=open('README.md').read(),
    author='XXX',
    author_email='XXX',
    url='XXX',
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
