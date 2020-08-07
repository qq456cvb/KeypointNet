import os
import sys
import yaml
import logging
import torch
import shutil
import numpy as np


def naive_read_pcd(path):
    lines = open(path, 'r').readlines()
    idx = -1
    for i, line in enumerate(lines):
        if line.startswith('DATA ascii'):
            idx = i + 1
            break
    lines = lines[idx:]
    lines = [line.rstrip().split(' ') for line in lines]
    data = np.asarray(lines)
    pc = np.array(data[:, :3], dtype=np.float)
    colors = np.array(data[:, -1], dtype=np.int)
    colors = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], -1)
    return pc, colors


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
        return np.array([[np.cos(a), 0, np.sin(a), 0],
                         [0, 1, 0, 0],
                         [-np.sin(a), 0, np.cos(a), 0],
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

def add_noise(x, sigma=0.015, clip=0.05):
    noise = np.clip(sigma*np.random.randn(*x.shape), -1*clip, clip)
    return x + noise

def clean_logs_and_checkpoints(cfg):
    checkpoint_path = os.path.join(
        cfg["root_path"], cfg["checkpoint_path"], cfg["name"])
    log_path = os.path.join(cfg["root_path"], cfg["log_path"], cfg["name"])
    print("Overwriting existing directory {} and {}".format(
        checkpoint_path, log_path))
    print("Proceed (y/[n])?", end=' ')
    choice = input()
    if choice == 'y':
        pass
    else:
        sys.exit()
    if os.path.exists(checkpoint_path):
        print("Deleting ", checkpoint_path)
        shutil.rmtree(checkpoint_path)
        os.mkdir(checkpoint_path)
        # os.unlink(checkpoint_path)
    if os.path.exists(log_path):
        print("Deleting ", log_path)
        shutil.rmtree(log_path)
        os.mkdir(log_path)
        # for f in os.listdir(log_path):
        #     print("Deleting ", os.path.join(log_path, f))
        #     os.unlink(os.path.join(log_path, f))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        dirname = os.path.dirname(filename)
        torch.save(state, os.path.join(dirname, 'model_best.pth.tar'))
    else:
        torch.save(state, filename)


def get_logger(cfg):
    format = "%(asctime)s - %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=format
    )
    logger = logging.getLogger(cfg["name"] + " - " + cfg["mode"])
    log_folder_path = os.path.join(
        cfg["root_path"], cfg["log_path"], cfg["name"])
    ckp_folder_path = os.path.join(
        cfg["root_path"], cfg["checkpoint_path"], cfg["name"])
    if not os.path.exists(log_folder_path):
        os.mkdir(log_folder_path)
    if not os.path.exists(ckp_folder_path):
        os.mkdir(ckp_folder_path)
    file_handler = logging.FileHandler(
        os.path.join(log_folder_path, cfg['log_name']))
    file_handler.setFormatter(logging.Formatter(format))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger

def get_cfg(args=None):
    parent_path = os.path.dirname(__file__)
    cfg_path = os.path.join(parent_path, '..', 'cfg/{}.yml'.format('demo'))
    with open(cfg_path, "r") as file:
        cfg = yaml.load(file)
        file.close()
    if args is not None:
        for arg in vars(args):
            if getattr(args, arg) != '-1':
                cfg[arg] = getattr(args, arg)
    return cfg

def convert_kp_to_one_hot(kps, num_point):
    """
    :param kps: (4994, 10)]
    :return: (4994, 2048, 10)
    """
    one_hot = torch.zeros(kps.shape[0], num_point, kps.shape[1])
    for i in range(kps.shape[0]):
        for j in range(kps.shape[1]):
            one_hot[i, kps[i, j].long(), j] = 1 if kps[i, j] != -1 else 0
    return one_hot

def convert_kp_to_vector(kps, num_point):
    """
    Convert the indices of keypoints into one vector \in {0, 1}^N
    :param kps: (B, K)
    :param num_point: N
    :return: (B, N, 2)
    """
    vector = torch.zeros(kps.shape[0], num_point, 2)
    for i, kp in enumerate(kps):
        for j in range(vector.size(1)):
            if j in kp:
                vector[i, j, 1] = 1
            else:
                vector[i, j, 0] = 1
    return vector

def convert_kp_to_binary(kps, num_point):
    """
    Convert the indices of keypoints into one vector \in {0, 1}^N
    :param kps: (B, K)
    :param num_point: N
    :return: (B, N, 2)
    """
    vector = torch.zeros(kps.shape[0], num_point)
    for i, kp in enumerate(kps):
        for k in kp:
            if k!= -1:
                vector[i, k] = 1
    return vector

def pck(P, KP, pred_KP, geo_dists):
    n_data = P.shape[0]
    n_points = P.shape[1]
    n_labels = KP.shape[1]
    K = pred_KP.shape[1]

    # dists_info: (point_cloud_index, label, basis_index, distance)
    dists_info = []

    for k in range(n_data):
        # NOTE:
        # Skip if the keypoint does not exist.
        labels = [i for i in range(n_labels) if KP[k, i] >= 0]

        # Find the closest prediction (w/o matching).
        for i, label in enumerate(labels):
            idx_i = KP[k, label]  # KP index
            assert (idx_i < n_points)
            p_i = P[k, idx_i]  # KP coordinates

            idx_j = pred_KP[k, label]
            p_j = P[k, idx_j]

            if geo_dists is None:
                all_dists = np.linalg.norm(p_i - p_j)
            else:
                all_dists = geo_dists[k][idx_i, idx_j]
            dists_info.append((k, i, i, all_dists))

    dists_info = np.array(dists_info)

    return dists_info


def calculate_correspondence(dists_info, KP):
    """
    Calculate the correspondence under different distance threshold
    :param dists_info: (point_cloud_index, label, basis_index, distance)
    :param KP: (B, 10)
    :return: correspondence list
    """
    threshold_list = [0.01*i for i in range(11)]
    corr_list = []
    # x = torch.ones(KP.size())
    # y = torch.zeros(KP.size())
    # num_KP = torch.sum((torch.where(KP.cpu() != -1, x, y)))
    num_KP = len(np.where(KP != -1)[0])
    for i, threshold in enumerate(threshold_list):
        correct = 0
        for j, dists in enumerate(dists_info):
            if dists[3] <= threshold:
                correct += 1
        corr_list.append(correct*1.0/num_KP)
    return corr_list

def judge(kp, logits, geo_dists):
    """
    :param kp: (B, num_keypoints)
    :param logits: (B, num_points)
    :param geo_dists: (B, num_points, num_points)
    :param threshold: geodesic threshold
    :return: return the true mask
    """
    dists = torch.zeros(logits.size(0), logits.size(1))
    for i in range(logits.size(0)):
        for j in range(logits.size(1)):
            min_dist = sys.maxsize
            for k in kp[i]:
                if k == -1:
                    continue
                dist = geo_dists[i, j, k]
                if dist < min_dist:
                    min_dist = dist
            dists[i, j] = min_dist
    return dists

def IoUMetric(kp, logits, geo_dists, prob_thr=0.1, geo_thr=0.1):
    """
    Calculate the iou between gt and pred keypoints
    :param pcds: (B, num_point, 3)
    :param kp: (B, num_keypoint)
    :param pred_kp: (B, num_kypoint)
    :return: IoU
    """
    # print ("max probability: {}".format(torch.max(logits, dim=1)))
    pred = torch.argmax(logits, dim=-1)
    pred = pred.squeeze(dim=-1).int()

    num_pos = 0
    FP_sum = 0
    FN_sum = 0

    for i in range(pred.size(0)):
        idx_pos = torch.nonzero(logits[i, :, 1] > prob_thr).squeeze(dim=-1)
        num_pos += torch.sum(logits[i, :, 1] > prob_thr)
        gt = kp[i]
        gt = gt >= 0
        gt = torch.nonzero(gt)
        idx_true = kp[i][gt].squeeze(dim=-1).long()

        #tmp = geo_dists[i][idx_pos][:, idx_true] > geo_thr
        #tmp1 = geo_dists[i][idx_true][:, idx_pos]
        FP = torch.sum((geo_dists[i][idx_pos][:, idx_true] > geo_thr).type(torch.ByteTensor).all(dim = -1))
        FN = torch.sum((geo_dists[i][idx_true][:, idx_pos] > geo_thr).type(torch.ByteTensor).all(dim = -1))
        FP_sum += FP
        FN_sum += FN

    print ("\nTP: {}, FP: {}, FN: {}".format(num_pos.float() - FP_sum, FP_sum, FN_sum))
    IoU = (num_pos.float() - FP_sum) / (num_pos + FN_sum)
    return IoU

def APMetric(kp, logits, geo_dists, prob_thr=0.1, geo_thr=0.1):
    """

    :param kp: (B, num_keypoints)
    :param logits: (B, num_points)
    :param geo_dists: (B, num_points, num_points)
    :param threshold:
    :return: AP
    """
    dists_info = judge(kp, logits, geo_dists)
    batch = kp.size(0)
    num_points = logits.size(1)
    sorted_logits, indices = torch.sort(logits[:, :, 1], dim=1, descending=True) # Sort logits with confidence

    pred_pos = logits[:, :, 1] > prob_thr  # Positive mask
    pred_pos = pred_pos.squeeze(dim=-1).int()

    gt = convert_kp_to_binary(kp, logits.size(1))

    sorted_pred_pos = torch.zeros(pred_pos.size())
    sorted_gt = torch.zeros(gt.size())
    sorted_dists = torch.zeros(dists_info.size())
    for i in range(batch):
        sorted_pred_pos[i, :] = pred_pos[i, indices[i]]
        sorted_gt[i, :] = gt[i, indices[i]]
        sorted_dists[i, :] = dists_info[i, indices[i]]


    TP = torch.zeros(batch, num_points)
    FP = torch.zeros(batch, num_points)
    FN = torch.zeros(batch, num_points)
    for i in range(batch):
        for j in range(num_points):
            if sorted_pred_pos[i,j] == 1 and sorted_dists[i, j] < geo_thr:
                TP[i, j] = 1
            elif sorted_pred_pos[i,j] == 1 and sorted_dists[i, j] >= geo_thr:
                FP[i, j] = 1
            else:
                if sorted_gt[i, j] == 1:
                    FN[i, j] = 1

    print ("TP: {}, FP: {}, FN: {}".format(torch.sum(TP), torch.sum(FP), torch.sum(FN)))

    AP_info = torch.zeros(batch, num_points, 2)
    for i in range(batch):
        for j in range(num_points):
            sum_tp = torch.sum(TP[i,:j+1])
            sum_fp = torch.sum(FP[i,:j+1])
            sum_fn = torch.sum(FN[i,:j+1])
            AP_info[i,j,0] = (sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) > 0 else 0
            AP_info[i,j,1] = (sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) > 0 else 0


    AP_cat = 0.
    for i, info_i in enumerate(AP_info):
        AP_instance = 0.
        for idx in range(11):
            recall_thr = idx * 0.1
            pre_i = info_i[:, 0]
            rec_i = info_i[:, 1]

            mask = rec_i >= recall_thr
            max_pre = torch.max(pre_i * mask.float())
            AP_instance += max_pre
        AP_instance /= 11
        AP_cat += AP_instance

    AP_cat /= len(AP_info)
    return AP_cat











