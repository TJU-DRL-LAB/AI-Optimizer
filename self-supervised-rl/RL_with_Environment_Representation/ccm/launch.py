"""
Launch an experiment on local machine with docker
"""

import sys

import doodad as dd
import doodad.ssh as ssh
import doodad.mount as mount


mode_local = dd.mode.LocalDocker(
        image='rakelly/rlkit:latest'
)

# Set up code and output directories
OUTPUT_DIR = 'output'  # doodad will prepend `/mounts` to this, set config in code to output to this path
mounts = [
    mount.MountLocal(local_dir='~/rlkit', pythonpath=True),  # point to your code
    mount.MountLocal(local_dir='~/.mujoco', mount_point='/root/.mujoco'),  # point to your mujoco
    mount.MountLocal(local_dir='~/rlkit/output', mount_point=OUTPUT_DIR, output=True),
]

call = sys.argv[1] + ' 1' # assume script has arg for docker mode
dd.launch_python(
    target=call,  # call target script (absolute path)
    mode=mode_local,
    mount_points=mounts,
    verbose=True,
)
