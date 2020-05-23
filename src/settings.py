import numpy as np
from utils import quaternion2yaw

# columns used as input for the model
input_cols = ['pos_x', 'pos_y', 'theta', 'camera']

# columns used as target for the model
# note: this is used for the real thymio
# target_cols = ['target_sensors']

# note: this is used for the simulated thymio
target_cols = ['target_left_sensor',
               'target_center_left_sensor',
               'target_center_sensor',
               'target_center_right_sensor',
               'target_right_sensor']

# a list of coordinates in a regular grid, shape = (n, 2)
coords = np.stack(np.meshgrid(
    np.linspace(0, .3, int(.3 / .02) + 1),
    np.linspace(-.15, .15, int(.3 / .02) + 1)
)).reshape([2, -1]).T


def mktr(x, y):
    """Returns a translation matrix given x and y."""
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])


def mkrot(theta):
    """returns a rotation matrix given theta."""
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos, -sin, 0],
                     [sin, cos, 0],
                     [0, 0, 1]])


# thymio's sensors' location w.r.t. robot frame
left_rot = mkrot(quaternion2yaw([0.0, 0.0, 0.3256, 0.9455]))
left_trs = mktr(0.063, 0.0493)

center_left_rot = mkrot(quaternion2yaw([0.0, 0.0, 0.1650, 0.9863]))
center_left_trs = mktr(0.0756, 0.0261)

center_rot = mkrot(quaternion2yaw([0.0, 0.0, 0.0, 1.0]))
center_trs = mktr(0.08, 0)

center_right_rot = mkrot(quaternion2yaw([0.0, 0.0, -0.1650, 0.9863]))
center_right_trs = mktr(0.0756, -0.0261)

right_rot = mkrot(quaternion2yaw([0.0, 0.0, -0.3256, 0.9455]))
right_trs = mktr(0.063, -0.0493)

# transformation matrices of sensors' frames w.r.t. robot frame
robot_geometry = [
    np.matmul(left_rot, left_trs),
    np.matmul(center_left_rot, center_left_trs),
    np.matmul(center_rot, center_trs),
    np.matmul(center_right_rot, center_right_trs),
    np.matmul(right_rot, right_trs)
]
