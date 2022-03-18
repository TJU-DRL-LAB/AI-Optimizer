"""
Platform domain by Warwick Masson et al. [2016], Reinforcement Learning with Parameterized Actions
Based on code from https://github.com/WarwickMasson/aaai-platformer

Author: C. Bester
June 2018
"""

import numpy as np
import gym
import pygame
import os
from os import path
from gym import error, spaces, utils
from gym.utils import seeding
import sys
__numba = True
try:
    from numba import jit
except:
    __numba = False


class Constants:
    # platform constants
    PLATFORM_HEIGHT = 40.0
    WIDTH1 = 250
    WIDTH2 = 275
    WIDTH3 = 50
    GAP1 = 225
    GAP2 = 235
    HEIGHT1 = 0.0
    HEIGHT2 = 0.0
    HEIGHT3 = 0.0
    MAX_HEIGHT = max(1.0, HEIGHT1, HEIGHT2, HEIGHT3)
    MAX_PLATFORM_WIDTH = max(WIDTH1, WIDTH2, WIDTH3)
    TOTAL_WIDTH = WIDTH1 + WIDTH2 + WIDTH3 + GAP1 + GAP2
    MAX_GAP = max(GAP1, GAP2)
    CHECK_SCALE = True

    # enemy constants
    ENEMY_SPEED = 30.0
    ENEMY_NOISE = 0.5
    ENEMY_SIZE = np.array((20.0, 30.0))

    # player constants
    PLAYER_NOISE = ENEMY_NOISE
    PLAYER_SIZE = np.copy(ENEMY_SIZE)

    # action constants
    DT = 0.05
    MAX_DX = 100.0
    MAX_DY = 200.0
    MAX_DX_ON = 70.0
    MAX_DDX = (MAX_DX - MAX_DX_ON) / DT
    MAX_DDY = MAX_DY / DT
    LEAP_DEV = 1.0
    HOP_DEV = 1.0
    VELOCITY_DECAY = 0.99
    GRAVITY = 9.8

    # scaling constants
    SHIFT_VECTOR = np.array([PLAYER_SIZE[0], 0., 0., ENEMY_SPEED,  # basic features
                             0., 0., 0., 0., 0.])  # platform features
    SCALE_VECTOR = np.array([TOTAL_WIDTH + PLAYER_SIZE[0], MAX_DX, TOTAL_WIDTH, 2 * ENEMY_SPEED,  # basic
                             MAX_PLATFORM_WIDTH, MAX_PLATFORM_WIDTH, MAX_GAP, TOTAL_WIDTH, MAX_HEIGHT])  # platform

    # available actions: RUN, HOP, LEAP
    # parameters for actions other than the one selected are ignored
    # action bounds were set from empirical testing using the default constants
    PARAMETERS_MIN = np.array([0, 0, 0])
    PARAMETERS_MAX = np.array([
        30,  # run
        720,  # hop
        430  # leap
    ])


ASSETS_PATH = path.join(path.dirname(__file__), "assets")
ENEMY_PATH = path.join(ASSETS_PATH, "enemy.png")
PLAYER_PATH = path.join(ASSETS_PATH, "player.png")
PLATFORM_PATH = path.join(ASSETS_PATH, "platform_v3.png")
BACKGROUND_PATH = path.join(ASSETS_PATH, "background.png")

# actions
RUN = "run"
HOP = "hop"
LEAP = "leap"
JUMP = "jump"

ACTION_LOOKUP = {
    0: RUN,
    1: HOP,
    2: LEAP,
}

SCREEN_HEIGHT = 300 # 500
SCREEN_WIDTH = int(Constants.TOTAL_WIDTH)


class PlatformEnv(gym.Env):
    # metadata = {'render.modes': ['human', 'rgb_array']}
    metadata = {'render.modes': ['human']}  # cannot use rgb_array at the moment due to frame skip between actions

    def __init__(self):
        """ Setup environment """

        # Entities
        self.xpos = 0.0
        self.player = Player()
        self.platform1 = Platform(0.0, Constants.HEIGHT1, Constants.WIDTH1)
        self.platform2 = Platform(Constants.GAP1 + self.platform1.size[0], Constants.HEIGHT2, Constants.WIDTH2)
        self.platform3 = Platform(self.platform2.position[0] +
                                  Constants.GAP2 + self.platform2.size[0], Constants.HEIGHT3, Constants.WIDTH3)
        self.enemy1 = Enemy(self.platform1)
        self.enemy2 = Enemy(self.platform2)

        self.np_random = None
        self.seed()

        self.states = []
        self.render_states = []  # record internal states for playback, cleared on reset()

        num_actions = 3
        self.action_space = spaces.Tuple((
            spaces.Discrete(num_actions),  # actions
            # spaces.Box(Constants.PARAMETERS_MIN, Constants.PARAMETERS_MAX, dtype=np.float32),  # parameters
            spaces.Tuple(  # parameters
                tuple(spaces.Box(low=np.array([Constants.PARAMETERS_MIN[i]]), high=np.array([Constants.PARAMETERS_MAX[i]]), dtype=np.float32)
                      for i in range(num_actions))
            )
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0., high=1., shape=self.get_state().shape, dtype=np.float32),
            spaces.Discrete(200),  # steps (200 limit is an estimate)
        ))

        self.window = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self._update_seeds()
        return [seed]

    def _update_seeds(self):
        self.player.np_random = self.np_random
        self.enemy1.np_random = self.np_random
        self.enemy2.np_random = self.np_random

    def _draw_entity(self, entity, sprite):
        """ Draws an entity as a rectangle. """
        self._draw_sprite(sprite, entity.position, entity.size)

    def _draw_sprite(self, sprite, position, size):
        """ Draws a sprite as a rectangle. Repeats the sprite as needed to fill the given size. """
        for i in range(int(size[0] / sprite.get_width())):
            pos = position + self.centre
            pos[0] += int(sprite.get_width() * i)
            self.draw_surface.blit(sprite, (pos[0], pos[1]))

    def _draw_background(self):
        """ Draw the static elements. """
        self.draw_surface.blit(self.background_sprite, (0, 0))
        self._draw_entity(self.platform1, self.platform_sprite)
        self._draw_entity(self.platform2, self.platform_sprite)
        self._draw_entity(self.platform3, self.platform_sprite)

    def _draw_foreground(self, render_state=None):
        """ Draw the player and the enemies. """
        if render_state:
            player_pos = render_state[0]
            enemy1_pos = render_state[1]
            enemy2_pos = render_state[2]
        else:
            player_pos = self.player.position
            enemy1_pos = self.enemy1.position
            enemy2_pos = self.enemy2.position
        self._draw_sprite(self.player_sprite, player_pos, self.player.size)
        self._draw_sprite(self.enemy_sprite, enemy1_pos, self.enemy1.size)
        self._draw_sprite(self.enemy_sprite, enemy2_pos, self.enemy2.size)

    def _get_image(self, alpha=255):
        surf = pygame.transform.flip(self.draw_surface, False, True)
        surf.set_alpha(alpha)
        image_data = pygame.surfarray.array3d(pygame.transform.rotate(surf, 90))
        return image_data

    def _draw_render_state(self, render_state):
        """
        Renders an internal state and returns an image.

        :param render_state:
        :return:
        """
        self._draw_background()
        self._draw_foreground(render_state)
        alpha = 255
        self._draw_to_screen(alpha)

    def _draw_to_screen(self, alpha=255):
        """ Draw the current window to the screen. """
        surf = pygame.transform.flip(self.draw_surface, False, True)
        surf.set_alpha(alpha)
        self.window.blit(surf, (0, 0))
        pygame.display.update()

    def step(self, action):
        """
        Take a full, stabilised update.

        Parameters
        ----------
        action (ndarray) :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
            reward (float) :
            terminal (bool) :
            info (dict) :
        """
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        terminal = False
        running = True
        act_index = action[0]
        act = ACTION_LOOKUP[act_index]
        param = action[1][act_index][0]
        #print(action)
        #print(act,param)
        param = np.clip(param, Constants.PARAMETERS_MIN[act_index], Constants.PARAMETERS_MAX[act_index])

        steps = 0
        difft = 1.0
        reward = 0.
        self.xpos = self.player.position[0]

        # update until back on platform
        while running:
            reward, terminal = self._update(act, param)

            if act == RUN:
                difft -= Constants.DT
                running = difft > 0
            elif act in [JUMP, HOP, LEAP]:
                running = not self._on_platforms()

            if terminal:
                running = False
            steps += 1

        state = self.get_state()
        obs = (state, steps)
        return obs, reward, terminal, {}

    def reset(self):
        self.xpos = 0.0
        self.player.reset()
        self.enemy1.reset()
        self.enemy2.reset()
        self.states = []
        self.render_states = []
        return self.get_state(), 0

    def _on_platforms(self):
        """ Checks if the player is on any of the platforms. """
        for platform in [self.platform1, self.platform2, self.platform3]:
            if self.player.on_platform(platform):
                return True
        return False

    def _perform_action(self, act, parameters, dt=Constants.DT):
        """ Applies for selected action for the given agent. """
        if self._on_platforms():
            if act == JUMP:
                self.player.jump(parameters)
            elif act == RUN:
                self.player.run(parameters, dt)
            elif act == LEAP:
                self.player.leap_to(parameters)
            elif act == HOP:
                self.player.hop_to(parameters)
        else:
            self.player.fall()

    def _lower_bound(self):
        """ Returns the lowest height of the platforms. """
        lower = min(self.platform1.position[1], self.platform2.position[1], self.platform3.position[1])
        return lower

    def _right_bound(self):
        """ Returns the edge of the game. """
        return self.platform3.position[0] + self.platform3.size[0]

    def _terminal_check(self, reward=0.0):
        """ Determines if the episode is ended, and the reward. """
        end_episode = self.player.position[1] < self._lower_bound() + Constants.PLATFORM_HEIGHT
        right = self.player.position[0] >= self._right_bound()
        for entity in [self.enemy1, self.enemy2]:
            if self.player.colliding(entity):
                end_episode = True
        if right:
            reward = (self._right_bound() - self.xpos) / self._right_bound()
            end_episode = True
        return reward, end_episode

    def _update(self, act, param, dt=Constants.DT):
        """
        Performs a single transition with the given action.

        Returns
        -------
        reward (float) :
        terminal (bool) :
        """
        # self.xpos = self.player.position[0]
        self.states.append([self.player.position.copy(),
                            self.enemy1.position.copy(),
                            self.enemy2.position.copy()])
        self.render_states.append(self.states[-1])
        self._perform_action(act, param, dt)
        if self._on_platforms():
            self.player.ground_bound()
        if self.player.position[0] > self.platform2.position[0]:
            enemy = self.enemy2
        else:
            enemy = self.enemy1
        for entity in [self.player, enemy]:
            entity.update(dt)
        for platform in [self.platform1, self.platform2, self.platform3]:
            if self.player.colliding(platform):
                self.player.decollide(platform)
                self.player.velocity[0] = 0.0
        reward = (self.player.position[0] - self.xpos) / self._right_bound()
        return self._terminal_check(reward)

    def _draw_render_states(self, mode="human"):
        """
        Draw the internal states from the last action.
        """
        if mode == "rgb_array":
            frames = []
        length = len(self.render_states)
        for i in range(0, length):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    sys.exit()
            self._draw_render_state(self.render_states[i])
        self.render_states = []  # clear states for next render

    def _platform_features(self, basic_features):
        """
        Compute the implicit features of the platforms.

        Parameters
        ----------
        basic_features :
        """
        xpos = basic_features[0]
        if xpos < Constants.WIDTH1 + Constants.GAP1:
            pos = 0.0
            wd1 = Constants.WIDTH1
            wd2 = Constants.WIDTH2
            gap = Constants.GAP1
            diff = Constants.HEIGHT2 - Constants.HEIGHT1
        elif xpos < Constants.WIDTH1 + Constants.GAP1 + Constants.WIDTH2 + Constants.GAP2:
            pos = Constants.WIDTH1 + Constants.GAP1
            wd1 = Constants.WIDTH2
            wd2 = Constants.WIDTH3
            gap = Constants.GAP2
            diff = Constants.HEIGHT3 - Constants.HEIGHT2
        else:
            pos = Constants.WIDTH1 + Constants.GAP1 + Constants.WIDTH2 + Constants.GAP2
            wd1 = Constants.WIDTH3
            wd2 = 0.0
            gap = 0.0
            diff = 0.0
        return [wd1, wd2, gap, pos, diff]
        # return [wd1 // Constants.MAX_PLATFORM_WIDTH, wd2 // Constants.MAX_PLATFORM_WIDTH, gap // Constants.MAX_GAP, pos // Constants.TOTAL_WIDTH, diff // Constants.MAX_HEIGHT]

    def _scale_state(self, state):
        scaled = (state + Constants.SHIFT_VECTOR) / Constants.SCALE_VECTOR
        return scaled

    def get_state(self):
        """ Returns the scaled representation of the current state. """
        if self.player.position[0] > self.platform2.position[0]:
            enemy = self.enemy2
        else:
            enemy = self.enemy1
        basic_features = [
            self.player.position[0],  # 0
            self.player.velocity[0],  # 1
            enemy.position[0],  # 2
            enemy.dx]  # 3
        platform_features = self._platform_features(basic_features)
        state = np.concatenate((basic_features, platform_features))
        scaled_state = self._scale_state(state)
        return scaled_state

    def render(self, mode='human', close=False):
        if close:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            return

        self._initialse_window()

        self._draw_render_states(mode)

        img = self._get_image()
        if mode == 'rgb_array':
            return img
        # elif mode == 'human':
        #    from gym.envs.classic_control import rendering
        #    if self.viewer is None:
        #        self.viewer = rendering.SimpleImageViewer(SCREEN_WIDTH, SCREEN_HEIGHT)
        #    self.viewer.imshow(img)

    def _initialse_window(self):
        # initialise visualiser
        if self.window is None:
            pygame.init()
            self.window_size = (SCREEN_WIDTH, SCREEN_HEIGHT)
            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
            self.background_sprite = pygame.image.load(BACKGROUND_PATH).convert_alpha()
            self.platform_sprite = pygame.image.load(PLATFORM_PATH).convert_alpha()
            self.enemy_sprite = pygame.image.load(ENEMY_PATH).convert_alpha()
            self.player_sprite = pygame.image.load(PLAYER_PATH).convert_alpha()
            self.centre = np.array((0, 100)) / 2
            self.draw_surface = pygame.Surface(self.window_size)

    def save_render_states(self, dir, prefix, index=0):
        self._initialse_window()
        import os
        for s in self.render_states:
            self._draw_render_state(s)
            pygame.image.save(self.window, os.path.join(dir, prefix+"_"+str("{:04d}".format(index))+".bmp"))
            index += 1
        return index


class Platform:
    """ Represents a fixed platform. """

    def __init__(self, xpos, ypos, width):
        self.position = np.array((xpos, ypos))
        self.size = np.array((width, Constants.PLATFORM_HEIGHT))


class Enemy:
    """ Defines the enemy. """

    size = Constants.ENEMY_SIZE
    speed = Constants.ENEMY_SPEED

    def __init__(self, platform):
        """ Initializes the enemy on the platform. """
        self.dx = -self.speed
        self.platform = platform
        self.position = self.platform.size + self.platform.position
        self.position[0] -= self.size[0]
        self.np_random = np.random  # overwritten by seed()

    def reset(self):
        self.dx = -self.speed
        self.position = self.platform.size + self.platform.position
        self.position[0] -= self.size[0]

    def update(self, dt):
        """ Shift the enemy along the platform. """
        right = self.platform.position[0] + self.platform.size[0] - self.size[0]
        if not self.platform.position[0] < self.position[0] < right:
            self.dx *= -1
        self.dx += self.np_random.normal(0.0, Constants.ENEMY_NOISE * dt)
        self.dx = np.clip(self.dx, -self.speed, self.speed)
        self.position[0] += self.dx * dt
        self.position[0] = np.clip(self.position[0], self.platform.position[0], right)


class Player:
    """ Represents the player character. """

    size = Constants.ENEMY_SIZE
    speed = Constants.ENEMY_SPEED

    def __init__(self):
        """ Initialize the position to the starting platform. """
        # self.position = vector(self.np_random.rand()*0.01, PLATHEIGHT)
        # self.velocity = vector(self.np_random.rand()*0.0001, 0.0)
        self.position = np.array((0., Constants.PLATFORM_HEIGHT))
        self.velocity = np.array((0., 0.0))
        self.np_random = np.random  # overwritten by seed()

    def reset(self):
        self.position = np.array((0., Constants.PLATFORM_HEIGHT))
        self.velocity = np.array((0., 0.0))

    def update(self, dt):
        """ Update the position and velocity. """
        self.position += self.velocity * dt
        self.position[0] = np.clip(self.position[0], 0.0, Constants.TOTAL_WIDTH)
        self.velocity[0] *= Constants.VELOCITY_DECAY

    def accelerate(self, accel, dt=Constants.DT):
        """ Applies a power to the entity in direction theta. """
        accel = np.clip(accel, (-Constants.MAX_DDX, -Constants.MAX_DDY), (Constants.MAX_DDX, Constants.MAX_DDY))
        self.velocity += accel * dt
        self.velocity[0] -= abs(self.np_random.normal(0.0, Constants.PLAYER_NOISE * dt))
        self.velocity = np.clip(self.velocity, (-Constants.MAX_DX, -Constants.MAX_DY),
                                (Constants.MAX_DX, Constants.MAX_DY))
        self.velocity[0] = max(self.velocity[0], 0.0)

    def ground_bound(self):
        """ Bound dx while on the ground. """
        self.velocity[0] = np.clip(self.velocity[0], 0.0, Constants.MAX_DX_ON)

    def run(self, power, dt):
        """ Run for a given power and time. """
        if dt > 0:
            self.accelerate(np.array((power / dt, 0.0)), dt)

    def jump(self, power):
        """ Jump up for a single step. """
        self.accelerate(np.array((0.0, power / Constants.DT)))

    def jump_to(self, diffx, dy0, dev):
        """ Jump to a specific position. """
        time = 2.0 * dy0 / Constants.GRAVITY + 1.0
        dx0 = diffx / time - self.velocity[0]
        dx0 = np.clip(dx0, -Constants.MAX_DDX, Constants.MAX_DY - dy0)
        if dev > 0:
            noise = -abs(self.np_random.normal(0.0, dev, 2))
        else:
            noise = np.zeros((2,))
        acceleration = np.array((dx0, dy0)) + noise
        self.accelerate(acceleration / Constants.DT)

    def hop_to(self, diffx):
        """ Jump high to a position. """
        self.jump_to(diffx, 35.0, Constants.HOP_DEV)

    def leap_to(self, diffx):
        """ Jump over a gap. """
        self.jump_to(diffx, 25.0, Constants.LEAP_DEV)

    def fall(self):
        """ Apply gravity. """
        self.accelerate(np.array((0.0, -Constants.GRAVITY)))

    def decollide(self, other):
        """ Shift overlapping entities apart. """
        precorner = other.position - self.size
        postcorner = other.position + other.size
        newx, newy = self.position[0], self.position[1]
        if self.position[0] < other.position[0]:
            newx = precorner[0]
        elif self.position[0] > postcorner[0] - self.size[0]:
            newx = postcorner[0]
        if self.position[1] < other.position[1]:
            newy = precorner[1]
        elif self.position[1] > postcorner[1] - self.size[1]:
            newy = postcorner[1]
        if newx == self.position[0]:
            self.velocity[1] = 0.0
            self.position[1] = newy
        elif newy == self.position[1]:
            self.velocity[0] = 0.0
            self.position[0] = newx
        elif abs(self.position[0] - newx) < abs(self.position[1] - newy):
            self.velocity[0] = 0.0
            self.position[0] = newx
        else:
            self.velocity[1] = 0.0
            self.position[1] = newy

    def above_platform(self, platform):
        """ Checks the player is above the platform. """
        return -self.size[0] <= self.position[0] - platform.position[0] <= platform.size[0]

    def on_platform(self, platform):
        """ Checks the player is standing on the platform. """
        ony = self.position[1] - platform.position[1] == platform.size[1]
        return self.above_platform(platform) and ony

    def colliding(self, other):
        """ Check if two entities are overlapping. """
        return _colliding(self.size, self.position, other.size, other.position)


if __numba:
    @jit(nogil=True, nopython=True)
    def _colliding(self_size, self_position, other_size, other_position):
        precorner = other_position - self_size
        postcorner = other_position + other_size
        collide = np.all(precorner < self_position)
        collide = collide and np.all(self_position < postcorner)
        return collide
else:
    def _colliding(self_size, self_position, other_size, other_position):
        precorner = other_position - self_size
        postcorner = other_position + other_size
        collide = np.all(precorner < self_position)
        collide = collide and np.all(self_position < postcorner)
        return collide
