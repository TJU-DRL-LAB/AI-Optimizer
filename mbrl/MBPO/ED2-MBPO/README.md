# ED2-MBPO
Code for combining ED2 with MBPO.

## Installation
Follow the installation of MBPO:
1. Install [MuJoCo 1.50](https://www.roboti.us/index.html) at `~/.mujoco/mjpro150` and copy your license key to `~/.mujoco/mjkey.txt`
2. Clone `mbpo`
```
git clone --recursive https://github.com/jannerm/mbpo.git
```
3. Create a conda environment and install mbpo
```
cd mbpo
conda env create -f environment/gpu-env.yml
conda activate mbpo
pip install -e viskit
pip install -e .
```

## Usage
Configuration files can be found in [`examples/config/`](examples/config).
```
python run.py
```

## Acknowledgments
Most implementation in ED2-MBPO comes from [Michael Janner](https://github.com/JannerM/mbpo) implementation.


