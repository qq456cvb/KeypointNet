import numpy as np
import h5py
import glob
import math
import cv2
import random
import os


# -- slice processing utils
def gen_slice_idx(data, resolution, axis=2):
    indices = np.zeros((data.shape[0], data.shape[2]))
    for n in range(data.shape[0]):
        indices[n] = gen_slice_idx_routine(data[n], resolution, axis)
    #
    return indices


def gen_slice_idx_routine(data, resolution, axis):
    if axis == 2:
        z_min, z_max = Z_MIN, Z_MAX
    else:
        z_min, z_max = data[:, :, axis].min(), data[:, :, axis].max()

    # gap = (z_max - z_min + 0.001) / numSlices
    gap = resolution
    indices = np.ones((data.shape[1], 1)) * float('inf')
    for i in range(data.shape[1]):
        z = data[0, i, axis]
        idx = int((z - z_min) / gap)
        indices[i, 0] = idx
    return indices[:, 0]


# -- utils for loading data, from Pointnet
def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def load_train_set(data_path):

    # load data
    f0 = h5py.File(os.path.join(data_path, 'ply_data_train0.h5'))
    f1 = h5py.File(os.path.join(data_path, 'ply_data_train1.h5'))
    f2 = h5py.File(os.path.join(data_path, 'ply_data_train2.h5'))
    f3 = h5py.File(os.path.join(data_path, 'ply_data_train3.h5'))
    f4 = h5py.File(os.path.join(data_path, 'ply_data_train4.h5'))
    f5 = h5py.File(os.path.join(data_path, 'ply_data_train5.h5'))
    f6 = h5py.File(os.path.join(data_path, 'ply_data_val0.h5'))
    f = [f0, f1, f2, f3, f4, f5, f6]

    data = f[0]['data'][:]
    seg = f[0]['pid'][:]

    for i in range(1, 7):
        data = np.concatenate((data, f[i]['data'][:]), axis=0)
        seg = np.concatenate((seg, f[i]['pid'][:]), axis=0)

    for ff in f:
        ff.close()

    return data, seg


def load_test_set(data_path):
    # load data
    f0 = h5py.File(os.path.join(data_path, 'ply_data_test0.h5'))
    f1 = h5py.File(os.path.join(data_path, 'ply_data_test1.h5'))
    f = [f0, f1]

    data = f[0]['data'][:]
    seg = f[0]['pid'][:]

    for i in range(1, 2):
        data = np.concatenate((data, f[i]['data'][:]), axis=0)
        seg = np.concatenate((seg, f[i]['pid'][:]), axis=0)

    for ff in f:
        ff.close()

    return data, seg


def loadDataFile(filename):
    return load_h5(filename)


# -- load data here
# - dataset setting, update when neccessay
block_size = 1.0
stride = 0.5

train_data, train_label = load_train_set('./data/shapenet_part_seg_hdf5_data')
test_data, test_label = load_test_set('./data/shapenet_part_seg_hdf5_data')

print("training set: ", (train_data.shape, train_label.shape))
print("testing set: ", (test_data.shape, test_label.shape))

Z_MIN, Z_MAX = min(train_data[:, :, 2].min(), test_data[:, :, 2].min()), max(train_data[:, :, 2].max(),
                                                                             test_data[:, :, 2].max())


def iterate_data(batchsize, resolution, train_flag=True, require_ori_data=False, block_size=1.0):

    if train_flag:
        data_all = train_data
        label_all = train_label
        indices = np.array(list(range(data_all.shape[0])))
        np.random.shuffle(indices)
    else:
        data_all = test_data
        label_all = test_label
        indices = np.array(list(range(data_all.shape[0])))
    # print(np.max(label_all), np.min(label_all))

    file_size = data_all.shape[0]
    num_batches = int(math.floor(file_size / float(batchsize)))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batchsize
        excerpt = indices[start_idx:start_idx + batchsize]

        inputs = data_all[excerpt].astype('float32')

        if require_ori_data:
            ori_inputs = inputs.copy()

        for b in range(inputs.shape[0]):
            minx = min(inputs[b, :, 0])
            miny = min(inputs[b, :, 1])
            inputs[b, :, 0] -= (minx + block_size / 2)
            inputs[b, :, 1] -= (miny + block_size / 2)

        inputs = np.expand_dims(inputs, 3).astype('float32')
        inputs = inputs.transpose(0, 3, 1, 2)

        seg_target = label_all[excerpt].astype('int64')  # num_batch, num_points
        # print(np.max(seg_target), np.min(seg_target))

        if len(resolution) == 1:
            resolution_x = resolution_y = resolution_z = resolution
        else:
            resolution_x, resolution_y, resolution_z = resolution

        x_slices_indices = gen_slice_idx(inputs, resolution_x, 0).astype('int32')
        y_slices_indices = gen_slice_idx(inputs, resolution_y, 1).astype('int32')
        z_slices_indices = gen_slice_idx(inputs, resolution_z, 2).astype('int32')

        if not require_ori_data:
            yield inputs, x_slices_indices, y_slices_indices, z_slices_indices, seg_target
        else:
            yield inputs, x_slices_indices, y_slices_indices, z_slices_indices, seg_target, ori_inputs


if __name__ == '__main__':
    print('test')
    pass
