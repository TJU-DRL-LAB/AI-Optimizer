"""
Robot Soccer Goal domain by Warwick Masson et al. [2016], Reinforcement Learning with Parameterized Actions
Based on code from https://github.com/WarwickMasson/aaai-goal

Author: C. Bester
June 2018
"""
import numpy as np
import math
import gym
import pygame
from gym import spaces, error
from gym.utils import seeding
import sys
from .config import PLAYER_CONFIG, BALL_CONFIG, GOAL_AREA_LENGTH, GOAL_AREA_WIDTH, GOAL_WIDTH, GOAL_DEPTH, KICKABLE, \
    INERTIA_MOMENT, MINPOWER, MAXPOWER, PITCH_LENGTH, PITCH_WIDTH, CATCHABLE, CATCH_PROBABILITY, SHIFT_VECTOR, \
    SCALE_VECTOR, LOW_VECTOR, HIGH_VECTOR
from .util import bound, bound_vector, angle_position, angle_between, angle_difference, angle_close, norm_angle, \
    vector_to_tuple

# actions
KICK = "kick"
DASH = "dash"
TURN = "turn"
TO_BALL = "toball"
SHOOT_GOAL = "shootgoal"
TURN_BALL = "turnball"
DRIBBLE = "dribble"
KICK_TO = "kickto"

ACTION_LOOKUP = {
    0: KICK_TO,
    1: SHOOT_GOAL,
    2: SHOOT_GOAL,
}

# field bounds seem to be 0, PITCH_LENGTH / 2, -PITCH_WIDTH / 2, PITCH_WIDTH / 2
PARAMETERS_MIN = [
    np.array([0, -PITCH_WIDTH / 2]),  # -15
    np.array([-GOAL_WIDTH / 2]),  # -7.01
    np.array([-GOAL_WIDTH / 2]),  # -7.01
]

PARAMETERS_MAX = [
    np.array([PITCH_LENGTH, PITCH_WIDTH / 2]),  # 40, 15
    np.array([GOAL_WIDTH / 2]),  # 7.01
    np.array([GOAL_WIDTH / 2]),  # 7.01
]

def norm(vec2d):
    # from numpy.linalg import norm
    # faster to use custom norm because we know the vectors are always 2D
    assert len(vec2d) == 2
    return math.sqrt(vec2d[0]*vec2d[0] + vec2d[1]*vec2d[1])


class GoalEnv(gym.Env):
    # metadata = {'render.modes': ['human', 'rgb_array']}
    metadata = {'render.modes': ['human']}  # cannot use rgb_array at the moment due to frame skip between actions
    _VISUALISER_SCALE_FACTOR = 20
    _VISUALISER_DELAY = 120  # fps

    def __init__(self):
        """ The entities are set up and added to a space. """

        self.np_random = None
        self.entities = []

        self.player = None
        self.ball = None
        self.goalie = None

        self.states = []
        self.render_states = []
        self.window = None

        self.time = 0
        self.max_time = 100

        num_actions = len(ACTION_LOOKUP)
        self.action_space = spaces.Tuple((
            spaces.Discrete(num_actions),  # actions
            spaces.Tuple(  # parameters
                tuple(spaces.Box(PARAMETERS_MIN[i], PARAMETERS_MAX[i], dtype=np.float32) for i in range(num_actions))
            )
        ))
        self.observation_space = spaces.Tuple((
            # spaces.Box(low=0., high=1., shape=self.get_state().shape, dtype=np.float32),  # scaled states
            spaces.Box(low=LOW_VECTOR, high=HIGH_VECTOR, dtype=np.float32),  # unscaled states
            spaces.Discrete(200),  # internal time steps (200 limit is an estimate)
        ))

        self.seed()

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
        act_index = action[0]
        act = ACTION_LOOKUP[act_index]
        param = action[1][act_index]
        param = np.clip(param, PARAMETERS_MIN[act_index], PARAMETERS_MAX[act_index])

        steps = 0
        self.time += 1
        if self.time == self.max_time:
            reward = -self.ball.goal_distance()
            end_episode = True
            state = self.get_state()
            return (state, 0), reward, end_episode, {}
        end_episode = False
        run = True
        reward = 0.
        while run:
            steps += 1
            reward, end_episode = self._update(act, param)
            run = not end_episode
            if run:
                run = not self.player.can_kick(self.ball)
                if act == DRIBBLE:
                    run = not self.ball.close_to(param) or run
                elif act == KICK_TO:
                    run = norm(self.ball.velocity) > 0.1 or run
                elif act == TURN_BALL:
                    theta = angle_between(self.player.position, self.ball.position)
                    run = not angle_close(theta, param[0]) or run
                elif act == SHOOT_GOAL:
                    run = not end_episode
                else:
                    run = False
        state = self.get_state()
        return (state, steps), reward, end_episode, {}

    def _update(self, act, param):
        """
        Performs a single transition with the given action,
        returns the reward and terminal status.
        """
        self.states.append([
            self.player.position.copy(),
            self.player.orientation,
            self.goalie.position.copy(),
            self.goalie.orientation,
            self.ball.position.copy()])
        self.render_states.append(self.states[-1])
        self._perform_action(act, param, self.player)
        self.goalie.move(self.ball, self.player)
        for entity in self.entities:
            entity.update()
        self._resolve_collisions()
        return self._terminal_check()

    def reset(self):
        # TODO: implement reset for each entity to avoid creating new objects and reduce duplicate code
        initial_player = np.array((0, self.np_random.uniform(-PITCH_WIDTH / 2, PITCH_WIDTH / 2)))
        angle = angle_between(initial_player, np.array((PITCH_LENGTH / 2, 0)))
        self.player = Player(initial_player, angle)

        MACHINE_EPSILON = 1e-12  # ensure always kickable on first state
        # fixes seeded runs changing between machines due to minor precision differences,
        # specifically from angle_position due to cos and sin approximations
        initial_ball = initial_player + (KICKABLE - MACHINE_EPSILON) * angle_position(angle)
        #initial_ball = initial_player + KICKABLE * angle_position(angle)
        self.ball = Ball(initial_ball)

        initial_goalie = self._keeper_target(initial_ball)
        angle2 = angle_between(initial_goalie, initial_ball)
        self.goalie = Goalie(initial_goalie, angle2)

        self.entities = [self.player, self.goalie, self.ball]
        self._update_entity_seeds()

        self.states = []
        self.render_states = []

        self.time = 0

        self.states.append([
            self.player.position.copy(),
            self.player.orientation,
            self.goalie.position.copy(),
            self.goalie.orientation,
            self.ball.position.copy()])
        self.render_states.append(self.states[-1])

        return self.get_state(), 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.reset()
        self._update_entity_seeds()
        return [seed]

    def _update_entity_seeds(self):
        # will be empty at initialisation, call again after creating all entities
        for entity in self.entities:
            entity.np_random = self.np_random

    @staticmethod
    def _keeper_line(ball):
        """ Finds the line the keeper wants to stay to. """
        grad = -ball[1] / (PITCH_LENGTH / 2 - ball[0])
        yint = ball[1] - grad * ball[0]
        return grad, yint

    def _keeper_target(self, ball):
        """ Target the keeper wants to move towards. """
        grad, yint = self._keeper_line(ball)
        if ball[0] < PITCH_LENGTH / 2 - GOAL_AREA_LENGTH:
            xval = ball[0]
        else:
            if ball[1] < -GOAL_AREA_WIDTH / 2:
                xval = (-GOAL_AREA_WIDTH / 2 - yint) / grad
            else:
                xval = (GOAL_AREA_WIDTH / 2 - yint) / grad
        xval = bound(xval, PITCH_LENGTH / 2 - GOAL_AREA_LENGTH, PITCH_LENGTH / 2)
        yval = bound(grad * xval + yint, -GOAL_AREA_WIDTH / 2, GOAL_AREA_WIDTH / 2)
        return np.array((xval, yval))

    def get_state(self):
        """ Returns the representation of the current state. """
        state = np.concatenate((
            self.player.position,
            self.player.velocity,
            [self.player.orientation],
            self.goalie.position,
            self.goalie.velocity,
            [self.goalie.orientation],
            self.ball.position,
            self.ball.velocity))
        #return self.scale_state(state)
        return state

    def _load_from_state(self, state):
        assert len(state) == len(self.get_state())
        self.player.position[0] = state[0]
        self.player.position[1] = state[1]
        self.player.velocity[0] = state[2]
        self.player.velocity[1] = state[3]
        self.player.orientation = state[4]
        self.goalie.position[0] = state[5]
        self.goalie.position[1] = state[6]
        self.goalie.velocity[0] = state[7]
        self.goalie.velocity[1] = state[8]
        self.goalie.orientation = state[9]
        self.ball.position[0] = state[10]
        self.ball.position[1] = state[11]
        self.ball.velocity[0] = state[12]
        self.ball.velocity[1] = state[13]

    def _perform_action(self, act, parameters, agent):
        """ Applies for selected action for the given agent. """

        if act == KICK:
            agent.kick_ball(self.ball, parameters[0], parameters[1])
        elif act == DASH:
            agent.dash(parameters[0])
        elif act == TURN:
            agent.turn(parameters[0])
        elif act == TO_BALL:
            agent.to_ball(self.ball)
        elif act == SHOOT_GOAL:
            agent.shoot_goal(self.ball, parameters[0])
        elif act == TURN_BALL:
            agent.turn_ball(self.ball, parameters[0])
        elif act == DRIBBLE:
            agent.dribble(self.ball, parameters)
        elif act == KICK_TO:
            agent.kick_to(self.ball, parameters[0])
        else:
            raise error.InvalidAction("Action not recognised: ", act)

    def _resolve_collisions(self):
        """ Shift apart all colliding entities with one pass. """
        for index, entity1 in enumerate(self.entities):
            for entity2 in self.entities[index + 1:]:
                if entity1.colliding(entity2):
                    entity1.decollide(entity2)

    def _terminal_check(self):
        """ Determines if the episode is ended, and the reward. """
        if self.ball.in_net():
            end_episode = True
            reward = 50
        elif self.goalie.can_catch(self.ball) or not self.ball.in_field():
            end_episode = True
            reward = -self.ball.goal_distance()
        else:
            end_episode = False
            reward = 0
        if end_episode:
            self.states.append([
                self.player.position.copy(),
                self.player.orientation,
                self.goalie.position.copy(),
                self.goalie.orientation,
                self.ball.position.copy()])
        return reward, end_episode

    def _is_stable(self):
        """ Determines whether objects have stopped moving. """
        speeds = [norm(entity.velocity) for entity in self.entities]
        return max(speeds) < 0.1

    @staticmethod
    def scale_state(state):
        """ Scale state variables between 0 and 1. """
        scaled_state = (state + SHIFT_VECTOR) / SCALE_VECTOR
        return scaled_state

    @staticmethod
    def unscale_state(scaled_state):
        """ Unscale state variables. """
        state = (scaled_state * SCALE_VECTOR) - SHIFT_VECTOR
        return state

    def __draw_internal_state(self, internal_state, fade=False):
        """ Draw the field and players. """

        player_position = internal_state[0]
        player_orientation = internal_state[1]
        goalie_position = internal_state[2]
        goalie_orientation = internal_state[3]
        ball_position = internal_state[4]
        ball_size = BALL_CONFIG['SIZE']

        self.window.blit(self.__background, (0, 0))

        # Draw goal and penalty areas
        length = self.__visualiser_scale(PITCH_LENGTH / 2)
        width = self.__visualiser_scale(PITCH_WIDTH)

        self.__draw_vertical(length, 0, width)
        self.__draw_box(GOAL_AREA_WIDTH, GOAL_AREA_LENGTH)
        # self.draw_box(PENALTY_AREA_WIDTH, PENALTY_AREA_LENGTH)

        depth = length + self.__visualiser_scale(GOAL_DEPTH)
        self.__draw_horizontal(width / 2 - self.__visualiser_scale(GOAL_WIDTH / 2), length, depth)
        self.__draw_horizontal(width / 2 + self.__visualiser_scale(GOAL_WIDTH / 2), length, depth)

        # self.draw_radius(vector(0, 0), CENTRE_CIRCLE_RADIUS)
        # Draw Players
        self.__draw_player(player_position, player_orientation, self.__white)
        if not fade:
            self.__draw_radius(player_position, KICKABLE)
        self.__draw_player(goalie_position, goalie_orientation, self.__red)
        if not fade:
            self.__draw_radius(goalie_position, CATCHABLE)
        # Draw ball
        self.__draw_entity(ball_position, ball_size, self.__black)
        pygame.display.update()

    def __visualiser_scale(self, value):
        ''' Scale up a value. '''
        return int(self._VISUALISER_SCALE_FACTOR * value)

    def __upscale(self, position):
        ''' Maps a simulator position to a field position. '''
        pos1 = self.__visualiser_scale(position[0])
        pos2 = self.__visualiser_scale(position[1] + PITCH_WIDTH / 2)
        return np.array([pos1, pos2])

    def __draw_box(self, area_width, area_length):
        """ Draw a box at the goal line. """
        lower_corner = self.__visualiser_scale(PITCH_WIDTH / 2 - area_width / 2)
        upper_corner = lower_corner + self.__visualiser_scale(area_width)
        line = self.__visualiser_scale(PITCH_LENGTH / 2 - area_length)
        self.__draw_vertical(line, lower_corner, upper_corner)
        self.__draw_horizontal(lower_corner, line, self.__visualiser_scale(PITCH_LENGTH / 2))
        self.__draw_horizontal(upper_corner, line, self.__visualiser_scale(PITCH_LENGTH / 2))

    def __draw_player(self, position, orientation, colour):
        ''' Draw a player with given position and orientation. '''
        size = PLAYER_CONFIG['SIZE']
        self.__draw_entity(position, size, colour)
        radius_end = size * angle_position(orientation)
        pos = vector_to_tuple(self.__upscale(position))
        end = vector_to_tuple(self.__upscale(position + radius_end))
        pygame.draw.line(self.window, self.__black, pos, end)

    def __draw_radius(self, position, radius):
        """ Draw an empty circle. """
        pos = vector_to_tuple(self.__upscale(position))
        radius = self.__visualiser_scale(radius)
        pygame.draw.circle(self.window, self.__white, pos, radius, 1)

    def __draw_entity(self, position, size, colour):
        """ Draws an entity as a ball. """
        pos = vector_to_tuple(self.__upscale(position))
        radius = self.__visualiser_scale(size)
        pygame.draw.circle(self.window, colour, pos, radius)

    def __draw_horizontal(self, yline, xline1, xline2):
        """ Draw a horizontal line. """
        pos1 = (xline1, yline)
        pos2 = (xline2, yline)
        pygame.draw.line(self.window, self.__white, pos1, pos2)

    def __draw_vertical(self, xline, yline1, yline2):
        """ Draw a vertical line. """
        pos1 = (xline, yline1)
        pos2 = (xline, yline2)
        pygame.draw.line(self.window, self.__white, pos1, pos2)

    def __draw_render_states(self):
        """
        Draw the internal states from the last action.
        """
        length = len(self.render_states)
        for i in range(0, length):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    sys.exit()
            self.__draw_internal_state(self.render_states[i])
            self.__clock.tick(self._VISUALISER_DELAY)
        self.render_states = []  # clear states for next render

    def render(self, mode='human', close=False):
        if close:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            return

        self._initialse_window()

        self.__draw_render_states()

        #img = self._get_image()
        #if mode == 'rgb_array':
        #    return img
        # elif mode == 'human':
        #    from gym.envs.classic_control import rendering
        #    if self.viewer is None:
        #        self.viewer = rendering.SimpleImageViewer(SCREEN_WIDTH, SCREEN_HEIGHT)
        #    self.viewer.imshow(img)

    def _initialse_window(self):
        # initialise visualiser
        if self.window is None:
            pygame.init()
            width = self.__visualiser_scale(PITCH_LENGTH / 2 + GOAL_DEPTH)
            height = self.__visualiser_scale(PITCH_WIDTH)
            self.window = pygame.display.set_mode((width, height))
            self.__clock = pygame.time.Clock()
            size = (width, height)
            self.__background = pygame.Surface(size)
            self.__white = pygame.Color(255, 255, 255, 0)
            self.__black = pygame.Color(0, 0, 0, 0)
            self.__red = pygame.Color(255, 0, 0, 0)
            self.__background.fill(pygame.Color(0, 125, 0, 0))

    def save_render_states(self, dir, prefix, index=0):
        self._initialse_window()
        import os
        for s in self.render_states:
            self.__draw_internal_state(s)
            pygame.image.save(self.window, os.path.join(dir, prefix+"_"+str("{:04d}".format(index))+".jpeg"))
            index += 1
        return index


class Entity:
    """ This is a base entity class, representing moving objects. """

    def __init__(self, config):
        self.rand = config['RAND']
        self.accel_max = config['ACCEL_MAX']
        self.speed_max = config['SPEED_MAX']
        self.power_rate = config['POWER_RATE']
        self.decay = config['DECAY']
        self.size = config['SIZE']
        self.position = np.array([0., 0.])
        self.velocity = np.array([0., 0.])
        self.np_random = None  # overwritten by seed()

    def update(self):
        """ Update the position and velocity. """
        self.position += self.velocity
        self.velocity *= self.decay

    def accelerate(self, power, theta):
        """ Applies a power to the entity in direction theta. """
        rrand = self.np_random.uniform(-self.rand, self.rand)
        theta = (1 + rrand) * theta
        rmax = self.rand * norm(self.velocity)
        noise = self.np_random.uniform(-rmax, rmax, size=2)
        rate = float(power) * self.power_rate
        acceleration = rate * angle_position(theta) + noise
        acceleration = bound_vector(acceleration, self.accel_max)
        self.velocity += acceleration
        self.velocity = bound_vector(self.velocity, self.speed_max)

    def decollide(self, other):
        """ Shift overlapping entities apart. """
        overlap = (self.size + other.size - self.distance(other)) / 2
        theta1 = angle_between(self.position, other.position)
        theta2 = angle_between(other.position, self.position)
        self.position += overlap * angle_position(theta2)
        other.position += overlap * angle_position(theta1)
        self.velocity *= -1
        other.velocity *= -1

    def colliding(self, other):
        """ Check if two entities are overlapping. """
        dist = self.distance(other)
        return dist < self.size + other.size

    def distance(self, other):
        """ Computes the euclidean distance to another entity. """
        return norm(self.position - other.position)

    def in_area(self, left, right, bot, top):
        """ Checks if the entity is in the area. """
        xval, yval = self.position
        in_length = left <= xval <= right
        in_width = bot <= yval <= top
        return in_length and in_width


class Player(Entity):
    """ This represents a player with a position,
        velocity and an orientation. """

    def __init__(self, position, orientation):
        """ The values for this class are defined by the player constants. """
        Entity.__init__(self, PLAYER_CONFIG)
        self.position = position
        self.orientation = orientation

    def homothetic_centre(self, ball):
        """ Computes the homothetic centre between the player and the ball. """
        ratio = 1. / (self.size + ball.size)
        position = (ball.position * self.size + self.position * ball.size)
        return ratio * position

    def tangent_points(self, htc):
        """ Finds the tangent points on the player wrt to homothetic centre. """
        diff = htc - self.position
        square = sum(diff ** 2)
        if square <= self.size ** 2:
            delta = 0.0
        else:
            delta = np.sqrt(square - self.size ** 2)
        xt1 = (diff[0] * self.size ** 2 + self.size * diff[1] * delta) / square
        xt2 = (diff[0] * self.size ** 2 - self.size * diff[1] * delta) / square
        yt1 = (diff[1] * self.size ** 2 + self.size * diff[0] * delta) / square
        yt2 = (diff[1] * self.size ** 2 - self.size * diff[0] * delta) / square
        tangent1 = np.array((xt1, yt1)) + self.position
        tangent2 = np.array((xt1, yt2)) + self.position
        tangent3 = np.array((xt2, yt1)) + self.position
        tangent4 = np.array((xt2, yt2)) + self.position
        if norm(tangent1 - self.position) == self.size:
            return tangent1, tangent4
        else:
            return tangent2, tangent3

    def ball_angles(self, ball, angle):
        """ Determines which angle to kick the ball along. """
        htc = self.homothetic_centre(ball)
        tangent1, tangent2 = self.tangent_points(htc)
        target = self.position + self.size * angle_position(angle)
        if norm(tangent1 - target) < norm(tangent2 - target):
            return angle_between(htc, tangent1)
        else:
            return angle_between(htc, tangent2)

    def kick_power(self, ball):
        """ Determines the kick power weighting given ball position. """
        angle = angle_between(self.position, ball.position)
        dir_diff = abs(angle_difference(angle, self.orientation))
        dist = self.distance(ball)
        return 1 - 0.25 * dir_diff / np.pi - 0.25 * dist / KICKABLE

    def facing_ball(self, ball):
        """ Determines whether the player is facing the ball. """
        angle = angle_between(self.position, ball.position)
        return self.facing_angle(angle)

    def facing_angle(self, angle):
        """ Determines whether the player is facing an angle. """
        return angle_close(self.orientation, angle)

    def turn(self, angle):
        """ Turns the player. """
        moment = norm_angle(angle)
        speed = norm(self.velocity)
        angle = moment / (1 + INERTIA_MOMENT * speed)
        self.orientation = self.orientation + angle

    def dash(self, power):
        """ Dash forward. """
        power = bound(power, MINPOWER, MAXPOWER)
        self.accelerate(power, self.orientation)

    def can_kick(self, ball):
        """ Determines whether the player can kick the ball. """
        return self.distance(ball) <= KICKABLE

    def kick_ball(self, ball, power, direction):
        """ Kicks the ball. """
        if self.can_kick(ball):
            power = bound(power, MINPOWER, MAXPOWER)
            power *= self.kick_power(ball)
            ball.accelerate(power, self.orientation + direction)

    def kick_towards(self, ball, power, direction):
        """ Kick the ball directly to a direction. """
        self.kick_ball(ball, power, direction - self.orientation)

    def shoot_goal(self, ball, ypos):
        """ Shoot the goal at a targeted position on the goal line. """
        ypos = bound(ypos, -GOAL_WIDTH / 2, GOAL_WIDTH / 2)
        target = np.array((PITCH_LENGTH / 2 + ball.size, ypos))
        self.kick_to(ball, target)

    def face_ball(self, ball):
        """ Turn the player towards the ball. """
        theta = angle_between(self.position, ball.position)
        self.face_angle(theta)

    def face_angle(self, angle):
        """ Turn the player towards and angle. """
        self.turn(angle - self.orientation)

    def to_ball(self, ball):
        """ Move towards the ball. """
        if not self.facing_ball(ball):
            self.face_ball(ball)
        elif not self.can_kick(ball):
            self.dash(10)

    def kick_to(self, ball, target):
        """ Kick the ball to a target position. """
        if not self.can_kick(ball):
            self.to_ball(ball)
        else:
            accel = (1 - ball.decay) * (target - self.position) - ball.velocity
            power = norm(accel) / (self.kick_power(ball) * ball.power_rate)
            theta = np.arctan2(accel[1], accel[0])
            self.kick_towards(ball, power, theta)

    def turn_ball(self, ball, angle):
        """ Turn the ball around the player. """
        if not self.can_kick(ball):
            self.to_ball(ball)
        elif not self.facing_angle(angle):
            self.face_angle(angle)
        elif self.size < self.distance(ball):
            theta = self.ball_angles(ball, angle)
            power = 0.1 / self.kick_power(ball)
            self.kick_towards(ball, power, theta)

    def dribble(self, ball, target):
        """ Dribble the ball to a position. """
        angle = angle_between(self.position, ball.position)
        theta = angle_between(self.position, target)
        if not self.can_kick(ball):
            self.to_ball(ball)
        elif ball.close_to(target):
            pass
        elif not angle_close(angle, theta):
            self.turn_ball(ball, theta)
        elif not self.facing_angle(theta):
            self.face_angle(theta)
        elif self.distance(ball) < (KICKABLE + self.size + ball.size) / 2:
            self.kick_towards(ball, 1.5, theta)
        else:
            self.dash(10)


class Goalie(Player):
    """ This class defines a special goalie player. """

    def move(self, ball, player):
        """ This moves the goalie. """
        ball_end = ball.position + ball.velocity / (1 - ball.decay)
        diff = ball_end - ball.position
        grad = diff[1] / diff[0] if diff[0] != 0. else 0  # avoid division by 0
        yint = ball.position[1] - grad * ball.position[0]
        goal_y = grad * PITCH_LENGTH / 2 + yint
        if ball_end[0] > PITCH_LENGTH / 2 and -GOAL_WIDTH / 2 - CATCHABLE <= goal_y <= GOAL_WIDTH / 2 + CATCHABLE \
                and grad != 0:
            grad2 = -1 / grad
            yint2 = self.position[1] - grad2 * self.position[0]
            ballx = (yint2 - yint) / (grad - grad2)
            bally = grad * ballx + yint
            target = np.array((ballx, bally))
            self.move_towards(20, target)
            self.orientation = angle_between(self.position, target)
        else:
            self.orientation = angle_between(self.position, ball_end)
            self.move_towards(8, ball_end)

    def move_towards(self, power, target):
        """ Move towards target position. """
        theta = angle_between(self.position, target)
        self.accelerate(power, theta)

    def can_catch(self, ball):
        """ Determines whether the goalie can catch the ball. """
        can_catch = self.distance(ball) < CATCHABLE
        return self.np_random.random_sample() <= CATCH_PROBABILITY and can_catch


class Ball(Entity):
    """ This class represents the ball, which has no orientation. """

    def __init__(self, position):
        """ The values for this class are defined by the ball constants. """
        Entity.__init__(self, BALL_CONFIG)
        self.position = position

    def close_to(self, position):
        """ Determines whether the ball is close to a postion. """
        return norm(self.position - position) <= 1.5

    def goal_distance(self):
        """ Returns the distance from the goal box. """
        if self.position[0] < PITCH_LENGTH / 2:
            if self.position[1] < -GOAL_WIDTH / 2:
                bot_corner = np.array((PITCH_LENGTH / 2, -GOAL_WIDTH / 2))
                return norm(self.position - bot_corner)
            elif self.position[1] > GOAL_WIDTH / 2:
                top_corner = np.array((PITCH_LENGTH / 2, GOAL_WIDTH / 2))
                return norm(self.position - top_corner)
            else:
                return PITCH_LENGTH / 2 - self.position[0]
        else:
            if self.position[1] < -GOAL_WIDTH / 2:
                return GOAL_WIDTH / 2 - self.position[1]
            elif self.position[1] > GOAL_WIDTH / 2:
                return self.position[1] - GOAL_WIDTH / 2
            else:
                return 0

    def in_field(self):
        """ Checks if the ball has left the field. """
        return self.in_area(0, PITCH_LENGTH / 2, -PITCH_WIDTH / 2, PITCH_WIDTH / 2)

    def in_net(self):
        """ Checks if the ball is in the net. """
        return self.in_area(PITCH_LENGTH / 2, PITCH_LENGTH / 2 + GOAL_DEPTH, -GOAL_WIDTH / 2, GOAL_WIDTH / 2)

    def in_goalbox(self):
        """ Checks if the ball is in the goal box. """
        return self.in_area(PITCH_LENGTH / 2 - GOAL_AREA_LENGTH, PITCH_LENGTH / 2, -GOAL_AREA_WIDTH / 2,
                            GOAL_AREA_WIDTH)
