import h5py
import torch
import numpy as np
from pytorchutils import *
from torchvision import transforms


def normalize(x):
    """Normalizes an image.

    Args:
            x: an image represented as a 3d numpy array.

    Returns:
            the normalized image.
    """
    return (x - x.mean()) / x.std()


def make_gradient(shape):
    """Creates a random gradient image.

    Args:
            shape: the shape of the gradient.

    Returns:
            the random gradient.
    """
    x, y = np.meshgrid(np.linspace(
        0, 1, shape[1]), np.linspace(0, 1, shape[0]))
    grad = x * np.random.uniform(-1, 1) + y * np.random.uniform(-1, 1)
    grad = normalize(grad)
    return grad


def additive_gradient(x, amount=np.random.uniform(0.05, 0.15), perturb_channels=True):
    """Applies a random gradient to the image.

    Args:
            x: an image represented as a 3d numpy array.
            amount: a float in [0, 1] representing the percentage of gradient over the original image.
            perturb_channels: if true the image channels are multiplied by a random numbers.

    Returns:
            the image with the added random gradient.
    """
    grad = make_gradient(x.shape)

    if perturb_channels:
        for i in range(3):
            x[:, :, i] = x[:, :, i] * np.random.uniform(0.9, 1.1)
        x = normalize(x)

    for i in range(3):
        x[:, :, i] = (1 - amount) * x[:, :, i] + amount * grad
    x = normalize(x)

    return x


def additive_noise(x, mu=0., sigma=0.5):
    """Adds gaussian noise centered on mu, with standard deviation sigma, to an image.

    Args:
            x: an image represented by a 3d numpy array.
            mu: the mean of the gaussian noise.
            sigma: the standard deviation of the gaussian noise.

    Returns:
            the image with the added noise.
    """
    gaussian = np.random.normal(mu, sigma, x.shape)
    return x + gaussian


def grayscale(x):
    """Converts an image to grayscale, mantaining 3 channels in the output.

    Args:
            x: an image represented by a 3d numpy array.

    Returns:
            the grayscale image.
    """
    return np.dstack([0.21 * x[:, :, 2] + 0.72 * x[:, :, 1] + 0.07 * x[:, :, 0]] * 3)


def flip(xy):
    """Flips an image and the corresponding labels.

    Args:
            xy: a tuple, whose first element is an image represented by a 3d numpy array.

    Returns:
            the flipped image and labels.
    """
    x, y = xy

    if np.random.choice([True, False]):
        x = np.flip(x, axis=2)  # flips columns of a batch of images

        for i in range(y.shape[1] // 5):
            # flips labels of a batch
            y[:, i * 5:(i + 1) * 5] = np.flip(y[:, i * 5:(i + 1) * 5], axis=1)

    return (x, y)


def random_augment(xy):
    """Applies a random perturbation to an image.

    Args:
            xy: a tuple, whose first element is an image represented by a 3d numpy array.

    Returns:
            the tuple with perturbed image.
    """
    x, y = xy

    choices = np.random.randint(0, 3, size=x.shape[0])
    for i, choice in enumerate(choices):
        if choice == 0:
            x[i] = additive_noise(x[i], mu=0.0, sigma=0.03)
        elif choice == 1:
            x[i] = grayscale(x[i])

        x[i] = normalize(x[i])
        x[i] = additive_gradient(x[i])

    return (x, y)


def binarize_labels(xy, theshold=0.5, zero_first=True):
    """Binarize labels using a threshold.

    Args:
            xy: a tuple, whose second element is a numpy numerical label.
            threshold: the threshold used for binarization.
            zero_first: if true set to 0 the values of labels <= threshold, otherwise 1.

    Returns:
            the tuple with binarized label.
    """
    x, y = xy

    low = 0.0 if zero_first else 1.0
    high = 1.0 - low

    y[(0 <= y) & (y <= theshold)] = low
    y[y > theshold] = high

    return (np.ascontiguousarray(x), np.ascontiguousarray(y))


def to_tensor(xy, device='cpu', dtype=torch.float):
    """Converts xy tuple of numpy arrays to tuple of torch tensors."""
    x, y = xy

    x = torch.tensor(x, device=device, dtype=dtype, requires_grad=True)
    y = torch.tensor(y, device=device, dtype=dtype, requires_grad=False)

    return (x, y)


def permute_x(xy):
    """Permutes dimensions of a batch of images (from H x W x D to D x H x W)."""
    x, y = xy

    # H x W x D to D x H x W
    x = np.swapaxes(x, -1, -3)
    x = np.swapaxes(x, -1, -2)

    return (x, y)


def contiguous(xy):
    """Returns a contiguous array version of the tuple xy."""
    x, y = xy

    return (np.ascontiguousarray(x), np.ascontiguousarray(y))


def get_dataset(filename, input_cols=['/x'], target_cols=['/y'], augment=True, device='cpu'):
    """Returns a HDF5Dataset containing examples of type (input, target).

    Args:
            filename: a tuple, whose second element is a numpy numerical label.
            input_cols: name of datasets inside the specified file, to be used as input for training.
            target_cols: name of datasets inside the specified file, to be used as target for training.
            augment: if true applies data augmentation (noise and lr-flipping).
            device: device on which to place tensors (usually "cuda" or "cpu").

    Returns:
            the HDF5Dataset containing examples of type (input, target).
    """
    transform = [lambda xy: permute_x(xy),
                 lambda xy: to_tensor(xy, device=device)]

    if augment:
        transform.insert(0, flip)
        transform.insert(0, random_augment)

    transform = transforms.Compose(transform)

    dataset = HDF5Dataset(filename,
                          x_extractor=lambda h5f: HDF5VerticalCollection(
                              [h5f[col] for col in input_cols]),
                          y_extractor=lambda h5f: HDF5VerticalCollection(
                              [h5f[col] for col in target_cols]),
                          mode='dataset', target='/', transform=transform)

    return dataset


def get_dataset_ranges(dataset):
    """Returns a dictionary of ranges, for each dataset in the HDF5 file."""
    cum_len = dataset.dataset.Y.cum_shapes[:, 0].copy()
    cum_len.put(0, 0)
    cum_range = [tuple(cum_len[i:i + 2].tolist())
                 for i in range(len(cum_len) - 1)]
    return {i: range(*interval) for i, interval in enumerate(cum_range)}
