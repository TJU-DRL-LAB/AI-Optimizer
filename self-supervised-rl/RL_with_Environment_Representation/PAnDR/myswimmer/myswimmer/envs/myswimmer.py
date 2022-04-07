import gym
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.swimmer_v3 import SwimmerEnv as SwimmerEnvOrig

import re
import os
import random
import numpy as np

from tempfile import mkdtemp
import contextlib
from shutil import copyfile, rmtree
from pathlib import Path


@contextlib.contextmanager
def make_temp_directory(prefix=''):
    temp_dir = mkdtemp(prefix)
    try:
        yield temp_dir
    finally:
        rmtree(temp_dir)


class SwimmerEnv(SwimmerEnvOrig):
    '''
    Family of Swimmer environments with different but related dynamics.
    '''
    def __init__(self, default_ind=0, num_envs=20, radius=0.1, viscosity=0.1, 
        density=4000, swimmer_density=1000, basepath=None):

        self.num_envs = num_envs
        SwimmerEnvOrig.__init__(self)

        self.default_params = {'wind': [0, 0, 0], 'viscosity': 0.1, \
                               'density': 4000, "swimmer_density": 1000}
        self.default_ind = default_ind

        self.env_configs = []
        for i in range(num_envs):
            angle = i * (1 / num_envs) * (2*np.pi)
            wind_x = radius * np.cos(angle)
            wind_y = radius * np.sin(angle)
            self.env_configs.append(
                {'wind': [wind_x, wind_y, 0], 'viscosity': viscosity, \
                 'density': density, 'swimmer_density': swimmer_density}
            )  
        self.env_configs.append(self.default_params)

        self.angle = (self.default_ind + 1) * (1 / num_envs) * (2*np.pi) 
        self.wind_x = radius * np.cos(self.angle)
        self.wind_y = radius * np.sin(self.angle)

        self.basepath = basepath
        file = open(os.path.join(self.basepath, "swimmer.xml"))
        self.xml = file.readlines()
        file.close()

    def get_xml(self, ind=0):
        xmlfile = open(os.path.join(self.basepath, "swimmer.xml"))
        tmp  = xmlfile.read()
        xmlfile.close()
        if ind is None:
            ind = self.default_ind

        params = {}
        params.update(self.default_params)
        params.update(self.env_configs[ind])

        wind_params = params['wind']
        viscosity = params['viscosity']
        density = params['density']
        swimmer_density = params["swimmer_density"]
        wx, wy, wz = wind_params 

        tmp = re.sub("<geom density=\"1000\" fromto=\"1.5 0 0 0.5 0 0\" size=\"0.1\" type=\"capsule\"/>",
                    f"<geom density=\"{swimmer_density}\" fromto=\"1.5 0 0 0.5 0 0\" size=\"0.1\" type=\"capsule\"/>", tmp)

        tmp = re.sub("<geom density=\"1000\" fromto=\"0 0 0 -1 0 0\" size=\"0.1\" type=\"capsule\"/>",
                    f"<geom density=\"{swimmer_density}\" fromto=\"0 0 0 -1 0 0\" size=\"0.1\" type=\"capsule\"/>", tmp)

        tmp = re.sub("<option collision=\"predefined\" density=\"4000\" integrator=\"RK4\" timestep=\"0.01\" viscosity=\"0.1\"/>",
                     f"<option collision=\"predefined\" density=\"{density}\" integrator=\"RK4\" timestep=\"0.01\" wind=\"{wx} {wy} {wz}\" viscosity=\"{viscosity}\"/>", tmp)

        f = open(os.path.join(str(Path.home()), 'tmpswimmer.xml'), 'w')
        f.write(tmp)
        f.close()
        return tmp

    def reset(self, env_id=None, same=False):
        if same:
            pass
        elif env_id:
            self.ind = env_id
        else:
            self.ind = self.default_ind

        with make_temp_directory(prefix='swimmer') as path_to_temp:
            xml = self.get_xml(self.ind)
            fpath = os.path.join(path_to_temp, f"tmp.xml")

            f = open(fpath, 'w')
            f.write(xml)
            f.close()
            mujoco_env.MujocoEnv.__init__(self, fpath, 5)

        self.sim.reset()
        ob = self.reset_model()
        return ob

