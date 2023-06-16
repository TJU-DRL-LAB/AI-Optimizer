from setuptools import setup, find_packages

setup(
    name='pex',
    version='0.0.1',
    python_requires='>=3.7.0',
    install_requires=[
        'tqdm',
        'scipy',
        'pandas',
        'torch>=1.8.1',
        'gym==0.15.4',
        'd4rl@git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl'
    ],
    packages=find_packages(),
)
