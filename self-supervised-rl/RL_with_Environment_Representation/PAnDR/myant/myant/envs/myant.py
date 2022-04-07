import gym
from gym.envs.mujoco import mujoco_env
from gym.envs.mujoco.ant_v3 import AntEnv as AntEnvOrig

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


class AntEnv(AntEnvOrig):
    '''
    Family of Ant environments with different but related dynamics.
    '''
    def __init__(self, default_ind=0, num_envs=20, radius=4.0, viscosity=0.05, basepath=None):
        self.num_envs = num_envs
        AntEnvOrig.__init__(self)

        self.default_params = {'limbs': [.2, .2, .2, .2], 'wind': [0, 0, 0], 'viscosity': 0.0}
        self.default_ind = default_ind

        self.env_configs = []

        for i in range(num_envs):
            angle = i * (1/num_envs) * (2*np.pi)
            wind_x = radius * np.cos(angle)
            wind_y = radius * np.sin(angle)
            self.env_configs.append(
                {'limbs': [.2, .2, .2, .2], 'wind': [wind_x, wind_y, 0], 'viscosity': viscosity}
            )
                
        self.env_configs.append(self.default_params)

        self.angle = (self.default_ind + 1) * (1/num_envs) * (2*np.pi) 
        self.wind_x = radius * np.cos(self.angle)
        self.wind_y = radius * np.sin(self.angle)

        self.basepath = basepath
        self.basepath = '/home/st/anaconda3/envs/cuda11/lib/python3.7/site-packages/gym/envs/mujoco/assets/'
        file = open(os.path.join(self.basepath, "ant.xml"))
        self.xml = file.readlines()
        file.close()


    def get_xml(self, ind=0):
        xmlfile = open(os.path.join(self.basepath, "ant.xml"))
        tmp  = xmlfile.read()
        xmlfile.close()
        if ind is None:
            ind = self.default_ind
        
        params = {}
        params.update(self.default_params)
        params.update(self.env_configs[ind])
        limb_params = params['limbs']
        wind_params = params['wind']
        viscosity = params['viscosity']
        wx, wy, wz = wind_params
        
        tmp = re.sub("<option integrator=\"RK4\" timestep=\"0.01\"/>",
                     f"<option integrator=\"RK4\" timestep=\"0.01\" wind=\"{wx} {wy} {wz}\" viscosity=\"{viscosity}\"/>", tmp)
        
        lf = limb_params[0]
        rf = limb_params[1]
        lb = limb_params[2]
        rb = limb_params[3]
        tmp = re.sub(" 0.2 0.2", f" {lf} {lf}", tmp)
        tmp = re.sub("\"0.2 0.2", f"\"{lf} {lf}", tmp)
        tmp = re.sub(" -0.2 0.2", f" -{rf} {rf}", tmp)
        tmp = re.sub("\"-0.2 0.2", f"\"-{rf} {rf}", tmp)
        tmp = re.sub(" 0.2 -0.2", f" {lb} -{lb}", tmp)
        tmp = re.sub("\"0.2 -0.2", f"\"{lb} -{lb}", tmp)
        tmp = re.sub(" -0.2 -0.2", f" -{rb} -{rb}", tmp)
        tmp = re.sub("\"-0.2 -0.2", f"\"-{rb} -{rb}", tmp)

        f = open(os.path.join(str(Path.home()), 'tmpant.xml'), 'w')
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
        with make_temp_directory(prefix='ant') as path_to_temp:
            xml = self.get_xml(self.ind)
            fpath = os.path.join(path_to_temp, f"tmp.xml")

            f = open(fpath, 'w')
            f.write(xml)
            f.close()
            mujoco_env.MujocoEnv.__init__(self, fpath, 5)

        self.sim.reset()
        ob = self.reset_model()
        return ob
