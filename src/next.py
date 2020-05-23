import numpy as np
import pandas as pd
from scipy import stats
from settings import coords, robot_geometry


def relative_to(positions, frame):
    '''Computes the relative positions w.r.t. the given frame of reference.

    Args:
            positions: an iterable of positions expressed in 2d homogeneous coordinates (x, y, 1).
            frame: a frame of reference from which to extract x, y and theta.

    Returns:
            the relative positions w.r.t. the given frame.
    '''
    cos = np.cos(frame['theta'])
    sin = np.sin(frame['theta'])
    x = frame['pos_x']
    y = frame['pos_y']

    inverse_frame = np.linalg.inv(np.array(
        [[cos, -sin, x],
         [sin, cos, y],
         [0, 0, 1]]
    ))
    return np.matmul(inverse_frame, np.array(positions).T).T


def get_map(relative_pose, sensor_readings, robot_geometry, coords, threshold=0.01):
    '''Given a pose, constructs the occupancy map w.r.t. that pose.
    An occupancy map has 3 possible values:
    1 = object present;
    0 = object not present;
    -1 = unknown;

    Args:
            relative_pose: the pose from which to compute the occupancy map.
            sensor_readings:  a list of sensors' readings.
            robot_geometry: the transformations from robot frame to sensors' frames.
            coords: a list of relative coordinates of the form [(x1, y1), ...].
            threshold: the maximum distance between a sensor reading and a coord to be matched.

    Returns:
            an occupancy map generated from the relative pose using coords and sensors' readings.
    '''
    coords_homo = np.array([[xy[0], xy[1], 1.] for xy in coords])

    tx, ty = relative_pose[:2]
    sin = np.sin(relative_pose[2])
    cos = np.cos(relative_pose[2])

    # get transform matrix from pose
    rel_transform = np.matmul(
        np.array([[cos, -sin, 0],
                  [sin, cos, 0],
                  [0, 0, 1]]),
        np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1],
                  ])
    )

    # get coords relative to the pose
    rel_coords = np.matmul(rel_transform,
                           coords_homo.T)[:-1].T

    # locate objects based on the distances read by the sensors
    rel_sensor_readings = np.array([[r, 0, 1.] for r in sensor_readings])

    rel_robot_geometry = np.matmul(np.linalg.inv(rel_transform), robot_geometry)
    rel_object_poses = np.einsum('ijk,ik->ij',  # batch matrix multiplication
                                 rel_robot_geometry,
                                 rel_sensor_readings)[:, :-1]

    # rel_object_poses = np.hstack((rel_object_poses,
    #                               np.ones((rel_object_poses.shape[0], 1))))
    # rel_object_poses = np.matmul(rel_transform,
    #                              rel_object_poses.T)[:-1].T

    # initialize occupancy map to -1
    occupancy_map = np.full((coords.shape[0],), -1, dtype=np.float)

    # compute distances between object poses and coords
    distances = np.linalg.norm(
        rel_coords[:, None, :] - rel_object_poses[None, :, :],
        axis=-1)

    # find closest coords with distance <= threshold
    indices = distances.argmin(axis=0)
    mask = distances[indices, np.arange(distances.shape[1])] <= threshold
    indices = indices[mask]

    # fill occupancy map based on sensors' readings
    # todo: 0.118 is the max distance for an object to be detected
    occupancy_map[indices] = sensor_readings[mask] < 0.118

    return occupancy_map


def aggregate(maps):
    """Aggregates a list of occupancy maps into a single one.

    Args:
            maps: a list of occupancy maps.

    Returns:
            an aggregate occupancy map.
    """
    aggregated_map = np.full_like(maps[0], -1)

    if (maps == -1).all():
        return aggregated_map

    map_mask = (maps != -1).any(axis=1)
    nonempty_maps = maps[map_mask]
    cell_mask = (nonempty_maps != -1).any(0)
    nonempty_cells = nonempty_maps[:, cell_mask]
    nonempty_cells[nonempty_cells == -1] = np.nan
    aggregated_map[cell_mask] = stats.mode(
        nonempty_cells, 0, nan_policy='omit')[0][0]
    return aggregated_map


def compute_occupancy_map(row, df, coords, target_columns, interval='15s', delta=0.1):
    '''Given a pose, constructs the occupancy map w.r.t. that pose.
    An occupancy map has 3 possible values:
    1 = object present;
    0 = object not present;
    -1 = unknown;

    Args:
            row: a dataframe row containing the pose from which to compute the occupancy map.
            df: the dataframe from which to extract data.
            coords: a list of relative coordinates of the form [(x1, y1), ...].
            target_columns:  a list of columns containing sensors' readings.
            interval: a time interval, expressed as string, for limiting the computation time.
            delta: the maximum distance between a sensor reading and a coord to be matched.

    Returns:
            a series composed of {target_column1: occupancy_map1, ...}.
    '''
    idx = row.name

    # consider only poses within a time interval from the reference pose
    window = df.loc[idx - pd.Timedelta(interval):idx + pd.Timedelta(interval)]

    # homogeneous coordinates of all poses within the window
    other_poses_homo = np.concatenate([
        np.expand_dims(window['pos_x'].values, axis=1),
        np.expand_dims(window['pos_y'].values, axis=1),
        np.ones((len(window), 1))], axis=1)

    # compute relative poses
    relative_poses = relative_to(other_poses_homo, row)
    factor = relative_poses[:, 2].copy()
    relative_poses /= factor[:, None]
    relative_poses[:, 2] = window['theta'].values - row['theta']

    sensor_readings = window[target_columns].values

    # compute occupancy maps (one map per relative pose)
    maps = [get_map(pose, reading, robot_geometry, coords, delta)
            for pose, reading in zip(relative_poses, sensor_readings)]

    # aggregate occupancy maps
    occupancy_map = aggregate(np.array(maps))

    return pd.Series({'target_map': occupancy_map})


def compute_occupancy_maps(df, coords, target_columns, interval='15s', delta=0.01):
    '''Creates occupancy map based on sensors' readings and odometry sored in a dataframe.

    Args:
            df: the dataframe in which to find the data to elaborate.
            coords: a list of relative coordinates of the form [(x1, y1), ...].
            target_columns: a list of columns containing sensors' readings.
            interval: a time interval, expressed as string, for limiting the computation time.
            delta: the maximum distance between a sensor reading and a coord to be matched.

    Returns:
            The new dataframe with the added columns corresponding to an occupancy map.
    '''
    next_df = df.apply(compute_occupancy_map, axis=1,
                       args=(df, coords, target_columns, interval, delta))

    output_cols = next_df.columns.values.tolist()
    df = pd.concat([df.drop(target_columns, axis=1), next_df], axis=1)

    return df, output_cols
