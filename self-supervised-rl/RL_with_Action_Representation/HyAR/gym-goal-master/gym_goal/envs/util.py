import math
import numpy as np


def bound(value, lower, upper):
    """ Clips off a value which exceeds the lower or upper bounds. """
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value


def bound_vector(vect, maximum):
    """ Bounds a vector between a negative and positive maximum range. """
    xval = bound(vect[0], -maximum, maximum)
    yval = bound(vect[1], -maximum, maximum)
    return np.array([xval, yval])


def angle_difference(angle1, angle2):
    """ Computes the real difference between angles. """
    return norm_angle(angle1 - angle2)


def norm_angle(angle):
    """ Normalize the angle between -pi and pi. """
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle


def angle_close(angle1, angle2):
    """ Determines whether an angle1 is close to angle2. """
    return abs(angle_difference(angle1, angle2)) < np.pi/8


def angle_between(pos1, pos2):
    """ Computes the angle between two positions. """
    diff = pos2 - pos1
    # return np.arctan2(diff[1], diff[0])
    return math.atan2(diff[1], diff[0])  # faster than numpy


def angle_position(theta):
    """ Computes the position on a unit circle at angle theta. """
    return vector(np.cos(theta), np.sin(theta))


def vector(xvalue, yvalue):
    """ Returns a 2D numpy vector. """
    # return np.array([float(xvalue), float(yvalue)])
    return np.array([xvalue, yvalue], dtype=np.float64)


def vector_to_tuple(vect):
    """ Converts a numpy array to a tuple. """
    assert len(vect) == 2
    return (vect[0], vect[1])
    # return tuple(map(tuple, vect))
