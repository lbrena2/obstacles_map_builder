import h5py
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset


###########  Loss  ###########


class MaskedLoss(torch.nn.Module):
    def __init__(self, loss, *args, **kwargs):
        super(MaskedLoss, self).__init__()
        self.loss = loss(*args, **kwargs)

    def forward(self, prediction, labels, mask, *args, **kwargs):
        return self.loss(prediction * mask, labels * mask, *args, **kwargs)


###########  Dataset  ###########

def is_dataset(h5f):
    return isinstance(h5f, h5py.Dataset)


def is_group(h5f):
    return isinstance(h5f, h5py.Group)


class HDF5VerticalCollection(object):
    def __init__(self, datasets, axis=0):
        self.datasets = datasets
        self.axis = axis

        shapes = np.array([dataset.shape for dataset in datasets])

        if not np.all(np.delete(np.bitwise_and.reduce(shapes) == shapes[0], axis)):
            raise ValueError('All dimentions except axis ' +
                             str(axis) + ' must match')

        cumsum = np.expand_dims(np.cumsum(shapes[:, axis]), axis=1)
        rest_shape = np.delete(shapes, axis, axis=1)

        def swap_cols(matrix, index1, index2):
            temp = matrix[:, index1].copy()
            matrix[:, index1] = matrix[:, index2].copy()
            matrix[: index2] = temp

        self.cum_shapes = np.concatenate([cumsum, rest_shape], axis=1)

        if axis != 0:
            swap_cols(self.cum_shapes, 0, axis)

        self.shapes = shapes
        self.shape = self.cum_shapes[-1]

    def __len__(self):
        return self.shape[self.axis]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def batches(self, batch_size, start=None, stop=None):
        length = len(self)
        start = 0 if start is None else start
        stop = length if stop is None else stop
        batch_size = stop if batch_size == float('inf') else batch_size

        for i in range(start, stop, batch_size):
            yield self[i:min(i + batch_size, length)]

    def _get_indices(self, idx):
        length = len(self)

        if idx < 0:
            idx += length

        if idx < 0 or idx > length:
            raise ValueError('Index ' + str(idx) + ' out of bounds')

        ds_idx = 0
        while self.cum_shapes[ds_idx, self.axis] <= idx:
            ds_idx += 1

        if ds_idx > 0:
            idx -= self.cum_shapes[ds_idx - 1, self.axis]

        return ds_idx, idx

    def _get_slice(self, slice):
        if slice.step is not None:
            raise ValueError('Slice ' + str(slice) +
                             ' with a step is not supported')

        length = len(self)
        start = 0 if slice.start == None else slice.start
        stop = length if slice.stop is None else slice.stop

        if stop < 0:
            stop += length

        if stop < 0 or stop > length:
            raise ValueError('Index ' + str(stop) + ' out of bounds')

        indices = np.array([self._get_indices(idx)
                            for idx in range(start, stop)])

        result = None
        ds_indices = np.unique(indices[:, 0])
        for ds_idx in ds_indices:
            m = indices[indices[:, 0] == ds_idx, 1][0]
            M = indices[indices[:, 0] == ds_idx, 1][-1] + 1
            ds_slice = self.datasets[ds_idx][m:M]

            if result is None:
                result = ds_slice
            else:
                result = np.vstack([result, ds_slice])

        return result

    def __getitem__(self, idx):
        if type(idx) == int:
            ds_idx, idx = self._get_indices(idx)
            return self.datasets[ds_idx][idx]
        elif type(idx) == slice:
            return self._get_slice(idx)
        # note: ... doesn't exist in py2
        # elif type(idx) == type(...):
        #     return self._get_slice(slice(None, None, None))
        elif type(idx) == range:
            return self._get_slice(slice(idx.start, idx.stop, None))
        else:
            raise ValueError('Indexing not supported: ' +
                             str(idx) + ' of type ' + str(type(idx)))


class HDF5SimpleDataset(Dataset):
    def __init__(self, file, x_extractor, y_extractor, transform=None):
        if is_dataset(file) or is_group(file):
            self.h5f = file
            self.close = False
        elif type(file) == str:
            self.h5f = h5py.File(file, 'r')
            self.close = True
        else:
            raise ValueError('Unknown file type: ' + str(file))

        self.transform = transform

        self.X = x_extractor(self.h5f)
        self.Y = y_extractor(self.h5f)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, slice):
        item = (self.X[slice], self.Y[slice])

        if self.transform is not None:
            item = self.transform(item)

        return item

    def __del__(self):
        if self.close:
            self.h5f.close()


class HDF5Dataset(Dataset):
    def __init__(self, file, x_extractor, y_extractor,
                 mode='group', target='/', transform=None):
        if is_group(file):
            self.h5f = file
            self.close = False
        elif type(file) == str:
            self.h5f = h5py.File(file, 'r')
            self.close = True
        else:
            raise ValueError('Unknown file type: ' + str(file))

        self.mode = mode
        self.transform = transform

        if mode == 'dataset':
            self.dataset = HDF5SimpleDataset(
                self.h5f[target], x_extractor, y_extractor, transform)
            self.interval = (0, len(self))
        elif mode == 'group':
            self.dataset = [HDF5SimpleDataset(self.h5f[target + '/' + k],
                                              x_extractor, y_extractor, transform)
                            for k in self.h5f[target].keys()]
            self.dataset = ConcatDataset(self.dataset)

            sizes = self.dataset.cumulative_sizes
            self.intervals = []
            current_size = 0
            for size in sizes:
                self.intervals.append((current_size, size))
                current_size = size
        else:
            raise ValueError('Unknown datset mode: ' + str(mode) +
                             '; Accepted values are: datset, group')

    def tree(self, k='/'):
        current = self.h5f[k]
        if is_dataset(current):
            return {k: current.shape}
        else:
            return {k: [self.tree(sk) for sk in k.keys()]}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, slice):
        return self.dataset[slice]

    def __del__(self):
        if self.close:
            self.h5f.close()
