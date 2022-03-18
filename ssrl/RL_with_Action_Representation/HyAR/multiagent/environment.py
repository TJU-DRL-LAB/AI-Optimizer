import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
import math


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = False
        # self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        self.flag = 0
        self.direction = 0
        self.accel = 0
        # hybrid action space
        self.n_action = 3
        self.hybrid_action_space = True
        # self.hybrid_action_space = False
        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,),
                                            dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def get_action_motions(self, n_actions):
        shape = (2 ** n_actions, 2)
        # print("shape",shape,shape[0])
        motions = np.zeros(shape)
        for idx in range(shape[0]):
            action = binaryEncoding(idx, n_actions)
            motions[idx] = np.dot(action, self.movement)
        max_dist = np.max(np.linalg.norm(motions, ord=2, axis=-1))
        motions /= max_dist
        return motions

    def get_movements(self, n_actions):
        """
        Divides 360 degrees into n_actions and
        assigns how much it should make the agent move in both x,y directions

        usage:  delta_x, delta_y = np.dot(action, movement)
        :param n_actions:
        :return: x,y direction movements for each of the n_actions
        """
        x = np.linspace(0, 2 * np.pi, n_actions + 1)  # 创建等差数列
        # print("x",x)
        y = np.linspace(0, 2 * np.pi, n_actions + 1)
        motion_x = np.around(np.cos(x)[:-1], decimals=3)  # 返回四舍五入后的值
        # print("motion_x",motion_x)
        motion_y = np.around(np.sin(y)[:-1], decimals=3)
        movement = np.vstack((motion_x, motion_y)).T  # 按垂直方向（行顺序）堆叠数组构成一个新的数组

        return movement

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            # self._set_action1(action_n[i], agent, self.action_space[i])
            self._set_action1(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    def _set_action1(self, action, agent, action_space, time=None):

        # agent.action.u = np.zeros(self.world.dim_p + 2)  #维度加2 前四维是移动的连续动作参数，后面一个离散动作是控制移动，一个是控制具体操作（抓、踢等）
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if action[0][0] == 2:
            agent.action.u = np.zeros(self.world.dim_p + 3)
        elif action[0][0] == 4:
            agent.action.u = np.zeros(self.world.dim_p + 3)
        elif action[0][0] == 5:
            agent.action.u = np.zeros(self.world.dim_p + 5)
        elif action[0][0] == 6:
            agent.action.u = np.zeros(self.world.dim_p + 4)
        else:
            agent.action.u = np.zeros(self.world.dim_p + 2)

        if agent.movable:
            # physical action
            if self.hybrid_action_space:

                # 4维连续动作参数(移动，停止)
                if action[0][0] == 0:
                    agent.action.u[2] = action[0][5]
                    if action[0][5] == 0:
                        agent.action.u[0] += 0
                        agent.action.u[1] += 0
                    else:
                        agent.action.u[0] += (action[0][1] - action[0][2]) * 2.0
                        agent.action.u[1] += (action[0][3] - action[0][4]) * 2.0
                    agent.action.u[3] = action[0][6]

                # 4个离散动作上下左右，4维连续动作参数[]
                if action[0][0] == 5:

                    if action[0][5] == 1: agent.action.u[0] = -action[0][1] * 2.0
                    if action[0][6] == 1: agent.action.u[0] = action[0][2] * 2.0
                    if action[0][7] == 1: agent.action.u[1] = -action[0][3] * 2.0
                    if action[0][8] == 1: agent.action.u[1] = action[0][4] * 2.0

                    agent.action.u[2] = action[0][5]
                    agent.action.u[3] = action[0][6]
                    agent.action.u[4] = action[0][7]
                    agent.action.u[5] = action[0][8]


                # 4个离散动作上下左右，4维连续动作参数[]
                if action[0][0] == 6:

                    if action[0][5] == 1: agent.action.u[0] = -action[0][1] * 2.0
                    if action[0][6] == 1: agent.action.u[0] = action[0][2] * 2.0
                    if action[0][7] == 1: agent.action.u[1] = -action[0][3] * 2.0
                    if action[0][8] == 1: agent.action.u[1] = action[0][4] * 2.0

                    agent.action.u[2] = action[0][5]
                    agent.action.u[3] = action[0][6]
                    agent.action.u[4] = action[0][7]
                    agent.action.u[5] = action[0][8]

                #  1是连续动作参数 2是离散动作，3是动作的维度
                if action[0][0] == 7:
                    # print("action[0][3]",action[0][3])
                    action_dim = int(action[0][3])
                    # print("action_dim",action_dim)
                    self.movement = self.get_movements(action_dim)
                    self.motions = self.get_action_motions(action_dim)
                    # print("action",int(action[0][2]))
                    # print("self.motions",self.motions,self.motions[int(action[0][2])])
                    action_true = self.motions[int(action[0][2])]
                    # print("action_true", action_true)
                    agent.action.u[0] += action_true[0] * action[0][1] * 2.0
                    agent.action.u[1] += action_true[1] * action[0][1] * 2.0


                #  1是离散动作，2是动作的维度  3是连续动作参数
                if action[0][0] == 8:
                    action_dim = int(action[0][2])
                    self.movement = self.get_movements(action_dim)
                    self.motions = self.get_action_motions(action_dim)
                    action_true = self.motions[int(action[0][1])]
                    accel=action[0][3][int(action[0][1])]
                    agent.action.u[0] += action_true[0] * accel * 2.0
                    agent.action.u[1] += action_true[1] * accel * 2.0


                # direction
                # (空，角度，移动，抓取)
                if action[0][0] == 1:
                    if action[0][2] == 0:
                        agent.action.u[0] += 0
                        agent.action.u[1] += 0
                    else:
                        agent.action.u[0] += np.sin(action[0][1]) * 2.0
                        agent.action.u[1] += np.cos(action[0][1]) * 2.0
                    agent.action.u[2] = action[0][2]
                    agent.action.u[3] = action[0][3]

                ##move
                # （空，角度，力度，调整角度，调整力度，停止）
                if action[0][0] == 2:
                    if action[0][5] == 1:
                        agent.action.u[0] += 0
                        agent.action.u[1] += 0
                    elif action[0][3] == 1:
                        self.direction = action[0][1]
                        agent.action.u[0] += np.sin(self.direction) * self.accel * 2.0
                        agent.action.u[1] += np.cos(self.direction) * self.accel * 2.0
                    elif action[0][4] == 1:
                        self.accel = action[0][2]
                        agent.action.u[0] += np.sin(self.direction) * self.accel * 2.0
                        agent.action.u[1] += np.cos(self.direction) * self.accel * 2.0

                    agent.action.u[2] = action[0][3]
                    agent.action.u[3] = action[0][4]
                    agent.action.u[4] = action[0][5]

                # direction
                # (空，角度，移动，抓取) 对direction进行了修改
                if action[0][0] == 3:
                    if action[0][2] == 0:
                        agent.action.u[0] += 0
                        agent.action.u[1] += 0
                    else:
                        agent.action.u[0] += np.sin(action[0][1]) * 0.5
                        agent.action.u[1] += np.cos(action[0][1]) * 0.5
                    agent.action.u[2] = action[0][2]
                    agent.action.u[3] = action[0][3]

                ##move_hard
                # （空，角度，力度，调整角度，调整力度，停止）
                if action[0][0] == 4:
                    if action[0][5] == 1:
                        agent.action.u[0] += 0
                        agent.action.u[1] += 0
                    elif action[0][3] == 1:
                        self.direction = action[0][1]
                        # agent.action.u[0] += np.sin(self.direction) * self.accel* 2.0
                        # agent.action.u[1] += np.cos(self.direction) * self.accel* 2.0
                    elif action[0][4] == 1:
                        self.accel = action[0][2]
                        agent.action.u[0] += np.sin(self.direction) * self.accel * 2.0
                        agent.action.u[1] += np.cos(self.direction) * self.accel * 2.0

                    agent.action.u[2] = action[0][3]
                    agent.action.u[3] = action[0][4]
                    agent.action.u[4] = action[0][5]

            # print("agent_u",agent.action.u,self.direction)
            sensitivity = 1.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                # from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array=mode == 'rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x, y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i + env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n


def binaryEncoding(num, size):
    binary = np.zeros(size)
    i = -1
    while num > 0:
        binary[i] = num % 2
        num = num // 2
        i -= 1
    return binary
