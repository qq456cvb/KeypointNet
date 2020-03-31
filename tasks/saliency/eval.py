import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable

# -- import RSNet utils
from net import RSNet
from utils import *
import load_data
from load_data import iterate_data, gen_slice_idx

NUM_CATEGORIES = 16
NUM_PART_CATS = 50


def convert_label_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0], NUM_CATEGORIES))
    for idx in range(labels.shape[0]):
        label_one_hot[idx, labels[idx]] = 1
    return label_one_hot


def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate
    :param a, b, c: ZYZ-Euler angles
    """

    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, -np.sin(a), 0],
                         [0, 1, 0, 0],
                         [np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
    if hom_coord:
        return r
    else:
        return r[:3, :3]


def rnd_rot():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, np.arccos(z), c, False)
    return rot


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == torch.Tensor:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


if __name__ == '__main__':
    print(torch.__version__)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                   'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                   'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                   'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15],
                   'Knife': [22, 23]}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    all_obj_cats_file = os.path.join('data/shapenet_part_seg_hdf5_data', 'all_object_categories.txt')
    fin = open(all_obj_cats_file, 'r')
    lines = [line.rstrip() for line in fin.readlines()]
    all_obj_cats = [(line.split()[0], line.split()[1]) for line in lines]
    fin.close()

    shape_accs = {cat: [] for cat in seg_classes.keys()}
    shape_ious = {cat: [] for cat in seg_classes.keys()}

    pool_type = 'Max_Pool'

    RANGE_X, RANGE_Y, RANGE_Z = 1.0, 1.0, load_data.Z_MAX
    resolution_true = [0.02, 0.02, 0.02]
    # - modified resolution for easy indexing
    resolution = [i + 0.00001 for i in resolution_true]
    num_slice = [0, 0, 0]
    num_slice[0] = int(RANGE_X / resolution[0]) + 1
    num_slice[1] = int(RANGE_Y / resolution[1]) + 1
    num_slice[2] = int(RANGE_Z / resolution[2]) + 1

    model = RSNet(pool_type, num_slice)
    model = model.cuda()

    pre_trained_model = torch.load('./model_best.pth.tar')

    start_epoch = pre_trained_model['epoch']
    best_prec1 = pre_trained_model['best_prec1']

    model_state = model.state_dict()
    model_state.update(pre_trained_model['state_dict'])
    model.load_state_dict(model_state)

    # - disable cudnn. cudnn raises error here due to irregular number of slices
    cudnn.benchmark = False

    batch_size = 24
    model.eval()
    hidden_list = model.init_hidden(batch_size)
    for batch in iterate_data(batch_size, resolution, train_flag=False, require_ori_data=True, block_size=1.0):
        inputs, x_indices, y_indices, z_indices, targets_raw, inputs_ori = batch
        rots = [rnd_rot() for _ in range(inputs.shape[0])]
        inputs = np.einsum('bij,btnj->btni', rots, inputs).astype(np.float32)
        # measure data loading time
        targets = targets_raw.reshape(-1)

        input_var = torch.autograd.Variable(torch.from_numpy(inputs).cuda(), requires_grad=True)
        target_var = torch.autograd.Variable(torch.from_numpy(targets).cuda(), requires_grad=False)

        x_indices_var = torch.autograd.Variable(torch.from_numpy(x_indices).cuda(), requires_grad=False)
        y_indices_var = torch.autograd.Variable(torch.from_numpy(y_indices).cuda(), requires_grad=False)
        z_indices_var = torch.autograd.Variable(torch.from_numpy(z_indices).cuda(), requires_grad=False)

        # compute output
        hidden_list = repackage_hidden(hidden_list)

        with torch.no_grad():
            output = model(input_var, x_indices_var, y_indices_var, z_indices_var, hidden_list)
            seg_pred_val = output.permute(0, 2, 1, 3).contiguous().view(-1, output.size(2), output.size(1)).cpu().numpy()

        # print(seg_pred_val.shape, targets_raw.shape)

        # rot = rnd_rot()
        pred_seg_res = np.argmax(seg_pred_val, -1)
        for k in range(pred_seg_res.shape[0]):
            pred = pred_seg_res[k]
            target = targets_raw[k]
            cat = seg_label_to_cat[target[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(target == l) == 0) and (
                        np.sum(pred == l) == 0):  # part is not present, no prediction as well
                    part_ious[l - seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l - seg_classes[cat][0]] = np.sum((target == l) & (pred == l)) / float(
                        np.sum((target == l) | (pred == l)))

            shape_ious[cat].append(np.mean(part_ious))
            shape_accs[cat].append(np.mean(pred == target))
            # print(cat, np.mean(shape_ious[cat]), np.mean(shape_accs[cat]))

    all_shape_ious = []
    all_shape_accs = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        for acc in shape_accs[cat]:
            all_shape_accs.append(acc)
        shape_ious[cat] = np.mean(shape_ious[cat])
        shape_accs[cat] = np.mean(shape_accs[cat])
    print("all shape mIoU: %f, class shape mIoU: %f" % (np.mean(all_shape_ious), np.mean(list(shape_ious.values()))))
    print("all shape macc: %f, class shape macc: %f" % (np.mean(all_shape_accs), np.mean(list(shape_accs.values()))))
    print("per shape mIoU")
    print(shape_ious)
    print("per shape macc")
    print(shape_accs)