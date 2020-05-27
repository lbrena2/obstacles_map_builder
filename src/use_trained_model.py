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


def plot_trajectory(poses):
    ax3 = plt.subplot()
    ax3.set_title('Test0_trajectory')
    ax3.plot(poses[:, 0], poses[:, 1], 'b.',
             alpha=0.1,
             markersize=7)

    plt.show()


def dummy_predictions(out):
    # Dummy prediction matrix to prove if the transformations and mean/median calcolus are ok
    for idx, pred in enumerate(out):
        if idx <= out.shape[0] / 2:
            out[idx] = np.zeros_like(pred)
        else:
            out[idx] = np.ones_like(pred)
    return out

def plot_pic_vs_prediction(img_idx, camera_images, out):
    #pick an image and plot it with the relative prediction
    plt.subplot(1, 2, 1)
    img = camera_images[img_idx] - camera_images[img_idx].min() / (
                camera_images[img_idx].max() - camera_images[img_idx].min())
    plt.imshow(img[:, :, ::-1])
    plt.subplot(1, 2, 2)
    sns.heatmap(out[img_idx].reshape([31, 5]))
    plt.savefig('aaaaa.svg')
    plt.show()

def camera_image_sequence(camera_images, last_img_idx):
    # Plot of the sequence of the images in the dataset, from the first to last_img_idx
    for i in range(1, last_img_idx):
        plt.subplot(9, 50, i)
        img = camera_images[i] - camera_images[i].min() / (camera_images[i].max() - camera_images[i].min())
        plt.imshow(img[:, :, ::-1])
    plt.show()

if __name__ == "__main__":
    model = NN(3, 65 * 5)
    model.load_state_dict(torch.load('/home/usi/catkin_ws/src/obstacles_map_builder/data/best.pth',
                                     map_location=torch.device('cpu')))
    model.eval()
    print(model)

    filename = "/home/usi/catkin_ws/src/obstacles_map_builder/data/h5dataset_Mirko_approach/2020-05-27 14:22:04.943898.h5"

    # Read the dataset content
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        camera_images = list(f.keys())[0]
        poses = list(f.keys())[2]

        # Get the data
        camera_images = np.array(f[camera_images])
        poses = np.array(f[poses])

    # Image preprocessing
    for idx, img in enumerate(camera_images):
        camera_images[idx] = cv2.blur(img, (11, 11))
        camera_images[idx] = (img - img.mean()) / (1 + img.std())
    input = torch.from_numpy(camera_images)
    input = input.permute([0, 3, 1, 2])

    # Prediction
    out = model(input.float())
    out = out.detach().numpy()[:, :155]

    # Dummy prediction
    # out = dummy_predictions(out)

    # Pick an image and plot it with the relative prediction
    # image_idx = 300
    # plot_pic_vs_prediction(image_idx, camera_images, out)

    # last_image_idx = 300
    # camera_image_sequence(camera_images, last_image_idx)


    # TODO: All this measuraments are in meters
    # 4x4 meters maps
    obstacle_map_coords = np.stack(np.meshgrid(
        np.linspace(-2, 2, 400),
        np.linspace(-2, 2, 400),
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
            obstacle_map_values[corresponding_coord_idx].append(prediction[coord_idx])

    # TODO: throws this warning  RuntimeWarning: Mean of empty slice
    #   obstacle_map_values = [np.nanmean(prediction_values) for prediction_values in obstacle_map_values]
    # obstacle_map_values[obstacle_map_values is []] = np.nan
    # obstacle_map_values_mean = [np.nanmean(prediction_values) for prediction_values in obstacle_map_values]
    # obstacle_map_values_median = [np.nanmedian(prediction_values) for prediction_values in obstacle_map_values]

    obstacle_map_values_mean = np.zeros(len(obstacle_map_values))
    for idx, prediction_values in enumerate(obstacle_map_values):
        if prediction_values == []:
            continue
        else:
            obstacle_map_values_mean[idx] = np.nansum(prediction_values, axis=0)
    obstacle_map_values_mean[obstacle_map_values_mean == 0] = np.nan

    obstacle_map_values_median = np.zeros(len(obstacle_map_values))
    for idx, prediction_values in enumerate(obstacle_map_values):
        if prediction_values == []:
            continue
        else:
            obstacle_map_values_median[idx] = np.nanmedian(prediction_values, axis=0)
    obstacle_map_values_median[obstacle_map_values_median == 0] = np.nan

    # Plotting
    ax1 = plt.subplot(2, 1, 1,
                      aspect='equal')

    ax1.set_title('Test0_mean')

    sns.heatmap(np.array(obstacle_map_values_mean).reshape((400, 400)),
                linewidth=0.5,
                center=0.5,
                ax=ax1,
                cbar=False,
                cmap='RdYlGn_r',
                )

    ax2 = plt.subplot(2, 1, 2,
                      aspect='equal',
                      label='Test0_median',
                      sharex=ax1,
                      sharey=ax1)

    ax2.set_title('Test0_median')

    sns.heatmap(np.array(obstacle_map_values_median).reshape((400, 400)),
                linewidth=0.5,
                center=0.5,
                ax=ax2,
                cmap='RdYlGn_r',
                cbar=False,
                cbar_kws={"orientation": "horizontal"})

    # ax3 = plt.subplot(3, 1, 3)
    # ax3.plot(poses[:, 0], poses[:, 1], 'b.',
    #          alpha=0.1,
    #          markersize=7)

    plt.show()

    # TODO: remember that i had modified the pitch camera in the original launch file from 0.2 to 0.5

    # TODO:
    #   - Send Mirko the median plot, perform the spin test in a Empty World. perform the go ahead test in a Empty World
    #       with a can cmd vel 0.004 to set. Compare mean and median result.
    #   - Be carefull of what is in range of the camera... you can have a can in the foreground but maybe the bot
    #       percieves the wall in the background -> the expretiments have to be performed
    #   - Start a doc with your notes and tries and settings
    #   - Invert plot colourmap
    #   - Create a pipeline such as you can create a .h5 with a single command
    #   - Create a dynamic plot?
    #   - ensure what happens when the thymio reach a pose in the map which is not in obstacle_maps_coords
