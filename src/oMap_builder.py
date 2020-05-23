import warnings
import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion


def quaternion2yaw(q):
    '''Converts a quaternion into the respective z euler angle.

    Args:
            q: a quaternion, composed of X, Y, Z, W.

    Returns:
             The euler angle part for the z axis a.k.a. yaw.
    '''
    if hasattr(q, 'x'):
        q = [q.x, q.y, q.z, q.w]

    return euler_from_quaternion(q)[2]


def get_maps(relative_pose, sensor_readings, robot_geometry, coords):
    coords_homo = np.array([[xy[0], xy[1], 1.] for xy in coords])

    rel_transform = np.matmul(mktr(*relative_pose[:2]),
                              mkrot(relative_pose[2]))

    rel_coords = np.matmul(rel_transform,
                           coords_homo.T)[:-1].T

    rel_robot_geometry = np.matmul(rel_transform,
                                   robot_geometry)

    rel_sensor_readings = np.array([[r, 0, 1.] for r in sensor_readings])

    # batch matrix multiplication
    rel_object_poses = np.einsum('ijk,ik->ij',
                                 rel_robot_geometry,
                                 rel_sensor_readings)[:, :-1]

    occupancy_map = np.full((coords.shape[0],), -1, dtype=np.float)

    distances = np.linalg.norm(
        rel_coords[:, None, :] - rel_object_poses[None, :, :],
        axis=-1)

    indices = distances.argmin(axis=0)
    mask = distances[indices,
                     np.arange(distances.shape[1])] <= 0.02
    indices = indices[mask]
    occupancy_map[indices] = sensor_readings[mask] < 0.12

    visualize_occ_map_coord(coords, occupancy_map)

    return occupancy_map


def mktr(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])


def mkrot(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos, -sin, 0],
                     [sin, cos, 0],
                     [0, 0, 1]])


def visualize_occ_map_coord(occ_map_coord, occ_map_coord_values):
    occ_map_coord_values_nan_id = (occ_map_coord_values == -1).nonzero()[0]
    occ_map_coord_values_0_id = (occ_map_coord_values == 0).nonzero()[0]
    occ_map_coord_values_1_id = (occ_map_coord_values == 1).nonzero()[0]

    fig, ax = plt.subplots()
    ax.plot(occ_map_coord[occ_map_coord_values_nan_id][:, 0],
            occ_map_coord[occ_map_coord_values_nan_id][:, 1], 'k.', markersize=5)
    ax.plot(occ_map_coord[occ_map_coord_values_1_id][:, 0],
            occ_map_coord[occ_map_coord_values_1_id][:, 1], 'ro', markersize=5)
    ax.plot(occ_map_coord[occ_map_coord_values_0_id][:, 0],
            occ_map_coord[occ_map_coord_values_0_id][:, 1], 'go', markersize=5)
    ax.axis("equal")

    plt.show()


if __name__ == '__main__':
    # Create a dummy transformation p3 wrt p1
    relative_pose = np.array([0.1 + 1e-5, 1e-5, 1e-5])

    # Create an arbitrary sensors readings in m
    sensor_readings = np.array([
        0.06,
        0.06,
        0.06,
        0.06,
        0.06
    ])

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

    sensors_relative_poses = [
        np.matmul(left_rot, left_trs),
        np.matmul(center_left_rot, center_left_trs),
        np.matmul(center_rot, center_trs),
        np.matmul(center_right_rot, center_right_trs),
        np.matmul(right_rot, right_trs)
    ]

    occ_map_coord_coord = np.stack(np.meshgrid(
        np.linspace(0, .3, int(.3 / .02) + 1),
        np.linspace(-.15, .15, int(.3 / .02) + 1)
    )).reshape([2, -1]).T

    m = get_maps(relative_pose, sensor_readings,
                 sensors_relative_poses, occ_map_coord_coord)
