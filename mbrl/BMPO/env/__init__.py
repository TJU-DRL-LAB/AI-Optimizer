import sys

from .pendulum import PendulumEnv
from .walker2d import Walker2dEnv
from .walker2dNT import Walker2dNTEnv
from .hopper import HopperEnv
from .hopperNT import HopperNTEnv
from .ant import AntEnv

env_overwrite = {'Pendulum': PendulumEnv,'Hopper':HopperEnv,'HopperNT':HopperNTEnv,
                 'Walker2d':Walker2dEnv,'Walker2dNT':Walker2dNTEnv,'Ant': AntEnv}

sys.modules[__name__] = env_overwrite