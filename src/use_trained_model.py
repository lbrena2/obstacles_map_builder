from pprint import pprint

import tqdm
from model_mirko import *
from dataset import get_dataset, get_dataset_ranges
from pytorchutils import *
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from settings import robot_geometry


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


def find_closest_cord(coord, obstacle_map_coords):
    closest_coord_distance = float('inf')
    clostest_coord_idx = 0

    for coord_idx, obstacle_map_coord in enumerate(obstacle_map_coords):
        eu_distance = np.sqrt((obstacle_map_coord[0] - coord[0]) ** 2 + (obstacle_map_coord[1] - coord[1]) ** 2)
        if eu_distance <= closest_coord_distance:
            closest_coord_distance = eu_distance
            clostest_coord_idx = coord_idx

    return clostest_coord_idx


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

    filename = "/home/usi/catkin_ws/src/obstacles_map_builder/data/h5dataset_Mirko_approach/2020-05-21 22:02:46.245455.h5"

    # TODO this works only with the toy example of a .h5 with 1 bag file only
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        camera_images = list(f.keys())[0]
        poses = list(f.keys())[2]

        # Get the data
        camera_images = list(f[camera_images])
        poses = np.array(f[poses])

    camera_images = np.array(camera_images)
    for idx, img in enumerate(camera_images):
        camera_images[idx] = cv2.blur(img, (11, 11))
    camera_images = (camera_images - camera_images.mean()) / (1 + camera_images.std())

    input = torch.from_numpy(camera_images)
    input = input.permute([0, 3, 1, 2])
    out = model(input.float())

    # PLOT
    # img_idx = 300
    # plt.subplot(1, 2, 1)
    # img = camera_images[img_idx] - camera_images[img_idx].min() / (camera_images[img_idx].max() - camera_images[img_idx].min())
    # plt.imshow(img[:, :, ::-1])
    # plt.subplot(1, 2, 2)
    # sns.heatmap(out[img_idx].reshape([65, 5]).detach().cpu().numpy())
    # plt.savefig('aaaaa.svg')
    # plt.show()

    obstacle_map_coords = np.stack(np.meshgrid(
        np.linspace(-70, 70, 280),
        np.linspace(-70, 70, 280)
    )).reshape([2, -1]).T

    # This is to create a cell in which I put che prediction. In other words, a element of obstacle_map_coords is
    # (x,y,prediction)
    obstacle_map_coords = np.hstack((obstacle_map_coords, np.ones((obstacle_map_coords.shape[0], 1))))


    # Create the prediction matrix coord
    # prediction_matrix_coords_homo = np.array([[x, 0, 1.] for x in range(0, 65)])
    prediction_matrix_coords_homo = [(0.063, 0.0493),
                                     (0.0756, 0.0261),
                                     (0.08, 0),
                                     (0.0756, -0.0261),
                                     (0.063, -0.0493)]

    # Create the actual set of coordinates of the predictions. Those coords are in robot reference frame.
    # The shape is 325x2.
    prediction_matrix_coords = np.array([np.array([x + (float(d)/100), y]) for x, y in prediction_matrix_coords_homo for d in range(0, 31)])

    visualize_map(obstacle_map_coords, prediction_matrix_coords)

    # Given the prediction coords get the corresponding coord in the obstacle map and assign it the prediction value
    for prediction, pose in zip(out, poses):
        # create the transformation from robot reference frame to world frame
        transform_matrix = robot_frame_to_world_transf(pose)

        # prediction_matrix_coords in world ref frame
        prediction_matrix_coords_world = np.array(
            [np.matmul(transform_matrix, coord) for coord in prediction_matrix_coords])

        # TODO: create a util to print the  obstacle_map_coords and  prediction_matrix_coords_world->
        # implement auto-refresh of the plot
        visualize_map(obstacle_map_coords, prediction_matrix_coords_world)

        # For each coord in  prediction_matrix_coords_world find the corrisponding one in obstacle_map_coords.
        # when you got the corrisponding coords put in obstacle_map_coords_homo
        # TODO: optimize, ont feasible
        for idx, coord in tqdm.tqdm(enumerate(prediction_matrix_coords_world), desc='updating obstacle map'):
            corresponding_coord_idx = find_closest_cord(coord, obstacle_map_coords)
            # pprint('prediction_matrix_coords: (%f, %f) obstacle_map_coord: (%f, %f)'
            #         %(coord[0],
            #         coord[1],
            #         obstacle_map_coords[:, :2][corresponding_coord][0],
            #         obstacle_map_coords[:, :2][corresponding_coord][1]))
            # TODO: check this assigment operation if it makes sense
            obstacle_map_coords[corresponding_coord_idx][2] = prediction[idx]
