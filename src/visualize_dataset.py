import cv2
import torch
import argparse
import numpy as np
from settings import coords
from dataset import get_dataset
from utils import map_to_image, bgr_tensor_to_rgb_numpy


def visualize(filename, input_cols, target_cols, save_video=True):
    """Visualize the content of the generator along with the prediction made by the model."""
    dataset = get_dataset(filename, device='cpu', augment=False,
                          input_cols=input_cols, target_cols=target_cols)

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fps', type=int,
                        help='frames per second for the video output', default=10)
    parser.add_argument('-p', '--pred', type=bool,
                        help='wether to include in the video the CNN model\'s predictions', default=False)
    args = parser.parse_args()

    fps = args.fps
    make_preds = args.pred

    size = (400, 300)

    if save_video:
        video = cv2.VideoWriter(
            '/home/usi/catkin_ws/src/obstacles_map_builder/out/generator.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (400, 600))
        print('Making the video...')

    for x, y in dataset:
        y = y.reshape([16, 16])
        # y = torch.flip(y, (0,))

        frame = bgr_tensor_to_rgb_numpy(x.detach())
        frame = frame[:, :, ::-1]  # rgb to bgr
        frame = cv2.resize(frame, size)
        frame = (frame * 255).astype(np.uint8)

        sensor_map = map_to_image(y, (15, 15), (3, 3))
        maps = np.hstack(
            [np.full((285, 57, 3), 255, dtype=np.uint8),
             sensor_map,
             np.full((285, 58, 3), 255, dtype=np.uint8)])
        frame = np.vstack(
            [maps, np.full((15, 400, 3), 255, dtype=np.uint8), frame])

        if save_video:
            video.write(frame)

        cv2.imshow('generator', frame)
        key = chr(cv2.waitKey(1000 // fps) & 0xFF)
        if key == 'q':
            break

    if save_video:
        video.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    filename = './dataset_2020-05-20 10:36:47.663266.h5'

    input_cols = ['bag0/camera']
    target_cols = ['bag0/target_map']

    visualize(filename, input_cols, target_cols, save_video=True)
