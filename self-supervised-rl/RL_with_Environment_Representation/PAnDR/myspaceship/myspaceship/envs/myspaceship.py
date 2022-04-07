import gym
from gym import spaces
import numpy as np
import random, math
from scipy import interpolate


class SpaceshipEnv(gym.Env):
    '''
    Family of Spaceship environments with different but related dynamics.

    An agent with a unit charge is initialized at the bottom of a room 
    whose dynamics are given by two other (fixed) electric charges in the room. 
    The agent needs to navigate from the bottom to the top of the room
    to obtain positive reward. The electric charges are strong enough to 
    deflect the agent's trajectory and make it leave the room through one of the other walls.
    '''
    def __init__(self, default_ind=0, num_envs=20, radius=1.5, 
        room_size=5, door_size=5, basepath=None):
        
        self.default_ind = default_ind
        self.num_envs = num_envs 

        self.radius = radius

        self.room_size = room_size
        self.door_size = door_size
        self.door_x_min = room_size / 2 - door_size / 2
        self.door_x_max = room_size / 2 + door_size / 2
        self.door_y = room_size
        self.door_x = room_size / 2

        self.q_pos_list = [(0, 0), (0, 5), (5, 0), (5, 5)]

        self.states = []
        self.reward = 0
        
        self.angle = (self.default_ind + 1) * (1/self.num_envs) * (2*np.pi) 
        self.q1 = radius * np.cos(self.angle) 
        self.q2 = radius * np.sin(self.angle) 
        self.q_list = [self.q1, self.q2]

        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), \
            high=np.array([5.0, 5.0]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-3.0, -3.0]), \
            high=np.array([3.0, 3.0]), dtype=np.float32)
        
        self.reset()

    def reset(self, env_id=None, same=False):
        if same: 
            pass
        elif env_id:
            self.ind = env_id
        elif env_id is not None:
            self.angle = env_id
        else:
            self.ind = self.default_ind

        self.angle = random.uniform(self.ind * (1/self.num_envs) * (2*np.pi), 
                                    (self.ind + 1) * (1/self.num_envs) * (2*np.pi))

        self.q1 = self.radius * np.cos(self.angle)
        self.q2 = self.radius * np.sin(self.angle)
        self.q_list = [self.q1, self.q2]

        self.done = False
        self.time_step = 0

        self.force_field_x, self.force_field_y = self._compute_force_field()

        self.pos_x = self.room_size / 2
        self.pos_y = self.room_size / 25

        self.state = (self.pos_x, self.pos_y)
        self.states.append(self.state)
        self.reward = 0
                
        return np.array(self.state).reshape(2)

    def step(self, action):
        if self.done:
            self.reset()

        self.time_step += 1

        agent_force_x, agent_force_y = action

        env_force_x = self.force_field_x(self.pos_x, self.pos_y)
        env_force_y = self.force_field_y(self.pos_x, self.pos_y)

        self.pos_x = self.pos_x + (agent_force_x + 5.0 * env_force_x) * (0.3)**2/2
        self.pos_y = self.pos_y + (agent_force_y + 5.0 * env_force_y) * (0.3)**2/2
        self.state = (self.pos_x, self.pos_y)
        self.states.append(self.state)

        if self.pos_x > self.door_x_min and self.pos_x < self.door_x_max \
            and self.pos_y > self.door_y:
            self.done = True

        elif self.pos_x < 0 or self.pos_x > self.room_size or \
            self.pos_y < 0 or self.pos_y > self.room_size or \
            self.time_step >= 50 or self.dist_to_charge() < .2:
            self.done = True
        else:
            self.reward = 0

        if self.done:
            dist_to_target = math.sqrt((self.pos_x - self.door_x)**2 + \
                                       (self.pos_y - self.door_y)**2)
            self.reward = math.exp(-3.0*dist_to_target)
        return np.array(self.state).reshape(2), self.reward, self.done, {}

    def dist_to_charge(self):
        '''
        Compute the distance from the charge to the goal location (center of the door)
        '''
        dists = []
        for i in range(len(self.q_list)):
            x_q = self.q_pos_list[i][0]
            y_q = self.q_pos_list[i][1]
            dists.append(math.sqrt((x_q-self.pos_x)**2 + (y_q - self.pos_y)**2))
        return min(dists)

    def _compute_force(self, xarr, yarr):
        '''
        Compute the electric forces for selected points in the environment (a fixed grid)
        '''
        force_x_arr = []
        force_y_arr = []

        for x, y in zip(xarr, yarr):
            force_x = 0
            force_y = 0
            for i in range(len(self.q_list)):
                q = self.q_list[i]
                x_q = self.q_pos_list[i][0]
                y_q = self.q_pos_list[i][1]

                force_x += q * (x - x_q) / (((x - x_q)**2 + (y - y_q)**2)**(3/2) + 0.1)
                force_y += q * (y - y_q) / (((x - x_q)**2 + (y - y_q)**2)**(3/2) + 0.1)

            force_x_arr.append(force_x)
            force_y_arr.append(force_y)

        return force_x_arr, force_y_arr

    def _compute_force_field(self):
        '''
        Compute the electric forces for a grid covering the environment
        '''
        x = np.arange(0, self.room_size, 0.1)
        y = np.arange(0, self.room_size, 0.1)
        xx, yy = np.meshgrid(x, y)
        zx, zy = self._compute_force(xx, yy)
        self.force_field_x = interpolate.interp2d(x, y, zx, kind='linear')
        self.force_field_y = interpolate.interp2d(x, y, zy, kind='linear')
        return self.force_field_x, self.force_field_y

