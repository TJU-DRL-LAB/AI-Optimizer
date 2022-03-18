import os

from setuptools import setup, Extension, find_packages

os.environ['CFLAGS'] = '-std=c++11'

# get __version__ variable
here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'tjuOfflineRL', '_version.py')).read())

if __name__ == "__main__":
    from numpy import get_include
    from Cython.Build import cythonize

    # setup Cython build
    ext = Extension('tjuOfflineRL.dataset',
                    sources=['tjuOfflineRL/dataset.pyx'],
                    include_dirs=[get_include(), 'tjuOfflineRL/cpp/include'],
                    language='c++',
                    extra_compile_args=["-std=c++11", "-O3", "-ffast-math"],
                    extra_link_args=["-std=c++11"])

    ext_modules = cythonize([ext],
                            compiler_directives={
                                'linetrace': True,
                                'binding': True
                            })

    # main setup
    setup(name="tjuOfflineRL",
          version=__version__,
          description="An offline deep reinforcement learning library",
          long_description=open("README.md").read(),
          long_description_content_type="text/markdown",
          url="https://github.com/TJU-DRL-LAB/Offline-RL.git",
          author="TJU-DRL-LAB",
          author_email="jianye.hao@tju.edu.cn",
          license="MIT License",
          classifiers=["Development Status :: 5 - Stable",
                       "Intended Audience :: Developers",
                       "Intended Audience :: Education",
                       "Intended Audience :: Science/Research",
                       "Topic :: Scientific/Engineering",
                       "Topic :: Scientific/Engineering :: Artificial Intelligence",
                       "Programming Language :: Python :: 3.6",
                       "Programming Language :: Python :: 3.7",
                       "Programming Language :: Python :: 3.8",
                       "Programming Language :: Python :: Implementation :: CPython",
                       "Operating System :: POSIX :: Linux",
                       'Operating System :: Microsoft :: Windows',
                       "Operating System :: MacOS :: MacOS X"],
          install_requires=["torch",
                            "scikit-learn",
                            "tensorboardX",
                            "tqdm",
                            "GPUtil",
                            "h5py",
                            "gym",
                            "click",
                            "typing-extensions",
                            "cloudpickle",
                            "scipy",
                            "structlog",
                            "colorama"],
          packages=find_packages(exclude=["tests*"]),
          python_requires=">=3.7.0",
          zip_safe=False,
          package_data={'tjuOfflineRL': ['*.pyx',
                                         '*.pxd',
                                         '*.h',
                                         '*.pyi',
                                         'py.typed']},
          ext_modules=ext_modules,
          entry_points={'console_scripts': ['tjuOfflineRL=tjuOfflineRL.cli:cli']})
