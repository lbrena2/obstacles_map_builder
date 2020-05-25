from pprint import pprint
import tqdm
from model_mirko import *
from dataset import get_dataset, get_dataset_ranges
from pytorchutils import *
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pandas import DataFrame




def robot_frame_to_world_transf(pose):
    cos = np.cos(pose[2])
    sin = np.sin(pose[2])
    x = pose[0]
    y = pose[1]

    transform = np.array(
        [[cos, -sin, x],
         [sin, cos, y],
         [0, 0, 1]]
    )
    return transform


def visualize_map(obstacle_map, coords):
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.plot(obstacle_map[:, 0], obstacle_map[:, 1], 'k.',
            alpha=0.1, markersize=2)
    ax.plot(coords[:, 0], coords[:, 1], 'r.',
            alpha=0.1, markersize=4)
    ax.axis("equal")
    plt.show()


if __name__ == "__main__":
    model = NN(3, 65 * 5)
    model.load_state_dict(torch.load('/home/usi/catkin_ws/src/obstacles_map_builder/data/best.pth',
                                     map_location=torch.device('cpu')))
    model.eval()
    print(model)

    filename = "/home/usi/catkin_ws/src/obstacles_map_builder/data/h5dataset_Mirko_approach/2020-05-24 23:19:55.501699.h5"

    # TODO this works only with the toy example of a .h5 generated with 1 bag file only
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        camera_images = list(f.keys())[0]
        poses = list(f.keys())[2]

        # Get the data
        camera_images = list(f[camera_images])
        poses = np.array(f[poses])

    camera_images = np.array(camera_images)
    # Image preprocessing
    for idx, img in enumerate(camera_images):
        camera_images[idx] = cv2.blur(img, (11, 11))
    camera_images = (camera_images - camera_images.mean()) / (1 + camera_images.std())
    input = torch.from_numpy(camera_images)
    input = input.permute([0, 3, 1, 2])

    # Prediction
    out = model(input.float())

    # plot of a pic and the relative predction
    # img_idx = 400
    # plt.subplot(1, 2, 1)
    # img = camera_images[img_idx] - camera_images[img_idx].min() / (camera_images[img_idx].max() - camera_images[img_idx].min())
    # plt.imshow(img[:, :, ::-1])
    # plt.subplot(1, 2, 2)
    # sns.heatmap(out[img_idx].reshape([65, 5]).detach().cpu().numpy())
    # plt.savefig('aaaaa.svg')
    # plt.show()

    # TODO: All this measuraments are in meters???
    obstacle_map_coords = np.stack(np.meshgrid(
        np.linspace(-1, 1, 200),
        np.linspace(-1, 1, 200),
    )).reshape([2, -1]).T

    obstacle_map_values = [[] for _ in range(obstacle_map_coords.shape[0])]

    # Create the prediction matrix coord
    prediction_matrix_coords_homo = [(0.063, 0.0493),
                                     (0.0756, 0.0261),
                                     (0.08, 0),
                                     (0.0756, -0.0261),
                                     (0.063, -0.0493)]

    # Create the actual set of coordinates of the predictions. Those coords are in robot reference frame.
    prediction_matrix_coords = np.array([np.array([x + (float(d) / 100), y])
                                         for x, y in prediction_matrix_coords_homo for d in range(0, 31)])

    # visualize_map(obstacle_map_coords, prediction_matrix_coords)

    # Given the prediction coords get the corresponding coord in the obstacle map and assign it the prediction value
    for prediction, pose in tqdm.tqdm(zip(out, poses), desc='creating obstacles map'):
        # create the transformation from robot reference frame to world frame
        transform_matrix = robot_frame_to_world_transf(pose)

        # prediction_matrix_coords in world ref frame
        prediction_matrix_coords_world = np.array(
            [np.matmul(transform_matrix, np.hstack((coord, 1))) for coord in prediction_matrix_coords])

        # TODO: create a util to print the  obstacle_map_coords and  prediction_matrix_coords_world->
        #   implement auto-refresh of the plot
        # TODO: ensure what happens when the thymio reach a pose in the map which is not in obstacle_maps_coords
        # visualize_map(obstacle_map_coords, prediction_matrix_coords_world)

        # For each coord in  prediction_matrix_coords_world find the corrisponding one in obstacle_map_coords.
        # when you got the corrisponding coords put in obstacle_map_coords_homo

        for coord_idx, coord in enumerate(prediction_matrix_coords_world):
            corresponding_coord_idx = np.argmin(
                np.linalg.norm(obstacle_map_coords - coord[:2], axis=1))
            # pprint('prediction_matrix_coords: (%f, %f) obstacle_map_coord: (%f, %f)'
            #         %(coord[0],
            #         coord[1],
            #         obstacle_map_coords[corresponding_coord_idx][0],
            #         obstacle_map_coords[corresponding_coord_idx][1]))
            # TODO: achtung about the prediction initialization value of obstacle_map_coords
            obstacle_map_values[corresponding_coord_idx].append(prediction[coord_idx].detach().numpy())

    # TODO: throws this warning  RuntimeWarning: Mean of empty slice
    #   obstacle_map_values = [np.nanmean(prediction_values) for prediction_values in obstacle_map_values]
    obstacle_map_values[obstacle_map_values is []] = np.nan
    obstacle_map_values = [np.nanmean(prediction_values) for prediction_values in obstacle_map_values]

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(obstacle_map_coords[:, 0], obstacle_map_coords[:, 1], 'k.',
            alpha=0.1,
            markersize=2)
    ax.axis("equal")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])

    ax = sns.heatmap(np.array(obstacle_map_values).reshape((200, 200)),
                     linewidth=0.5,
                     vmin=0,
                     vmax=1,
                     center=0.5,
                     ax=ax,
                     cmap='RdYlGn')
    plt.show()


    #TODO: remember that i had modified the pitch camera in the original launch file from 0.2 to 0.5

    # TODO:
    #   1- create a controller that spin the Thymio in place usign what Mirko said in Slack
    #   2- Wait Mirko to build the simple map/Resolve the funcking Gazebo save as bug
    #   3- Create a pipiline such as you can create a .h5 with a single command
    #   4- Create a dynamic plot?
