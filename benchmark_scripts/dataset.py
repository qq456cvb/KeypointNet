import numpy as np
from glob import glob
import os
import torch
import json
from scipy.stats import special_ortho_group
from sklearn.metrics import pairwise_distances_argmin


BASEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
ID2NAMES = {"02691156": "airplane",
            "02808440": "bathtub",
            "02818832": "bed",
            "02876657": "bottle",
            "02954340": "cap",
            "02958343": "car",
            "03001627": "chair",
            "03467517": "guitar",
            "03513137": "helmet",
            "03624134": "knife",
            "03642806": "laptop",
            "03790512": "motorcycle",
            "03797390": "mug",
            "04225987": "skateboard",
            "04379243": "table",
            "04530566": "vessel",}

NAMES2ID = {v: k for k, v in ID2NAMES.items()}


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


def add_noise(x, sigma=0.015, clip=0.05):
    noise = np.clip(sigma*np.random.randn(*x.shape), -1*clip, clip)
    return x + noise


def normalize_pc(pc):
    pc = pc - pc.mean(0)
    pc /= np.max(np.linalg.norm(pc, axis=-1))
    return pc


class KeypointSaliencyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.aug = cfg.data_aug
        self.catg = cfg.class_name
        self.cfg = cfg
        
        annots = json.load(open(os.path.join(BASEDIR, cfg.data.annot_path)))
        annots = [annot for annot in annots if annot['class_id'] == NAMES2ID[cfg.class_name]]
        keypoints = dict([(annot['model_id'], [kp_info['pcd_info']['point_index'] for kp_info in annot['keypoints']]) for annot in annots])
        
        split_models = open(os.path.join(BASEDIR, "splits/{}.txt".format(split))).readlines()
        split_models = [m.split('-')[-1].rstrip('\n') for m in split_models]
        
        self.pcds = []
        self.keypoints = []
        self.mesh_names = []
        for fn in glob(os.path.join(BASEDIR, cfg.data.pcd_root, NAMES2ID[cfg.class_name], '*.pcd')):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in split_models:
                continue
            self.keypoints.append(keypoints[model_id])
            self.pcds.append(naive_read_pcd(fn)[0])
            self.mesh_names.append(model_id)

        self.nclasses = 2

    
    def __getitem__(self, idx):
        pc = self.pcds[idx]
        mesh_name = self.mesh_names[idx]
        label = self.keypoints[idx]
        bin_label = np.zeros((pc.shape[0],), dtype=np.int64)
        bin_label[label] = 1
        
        if self.cfg.normalize_pc:
            pc = normalize_pc(pc)
        
        if self.cfg.augmentation.gaussian_noise:
            pc = add_noise(pc, sigma=0.004, clip=0.01)

        if self.cfg.augmentation.translation:
            tr = np.random.rand() * 0.02
            rot = special_ortho_group.rvs(3)
            pc = pc @ rot
            pc += np.array([[tr, 0, 0]])
            pc = pc @ rot.T

        # allow rotation along gravity axis
        if self.cfg.augmentation.rot_gravity:
            angle = np.random.rand() * 2 * np.pi
            rot = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
            pc = (rot @ pc.T).T
        return pc.astype(np.float32), bin_label, mesh_name

    def __len__(self):
        return len(self.pcds)
    

class KeypointCorrespondenceDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.aug = cfg.data_aug
        self.catg = cfg.class_name
        self.cfg = cfg
            
        annots = json.load(open(os.path.join(BASEDIR, cfg.data.annot_path)))
        annots = [annot for annot in annots if annot['class_id'] == NAMES2ID[cfg.class_name]]
        keypoints = dict([(annot['model_id'], [(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']]) for annot in annots])
        rotation_groups = dict([(annot['model_id'], annot['symmetries']['rotation']) for annot in annots])
        
        self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
        
        split_models = open(os.path.join(BASEDIR, "splits/{}.txt".format(split))).readlines()
        split_models = [m.split('-')[-1].rstrip('\n') for m in split_models]
        
        self.pcds = []
        self.keypoints = []
        self.mesh_names = []
        self.rotation_infos = []
        self.idx2semids = []
        for fn in glob(os.path.join(BASEDIR, cfg.data.pcd_root, NAMES2ID[cfg.class_name], '*.pcd')):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in split_models:
                continue
            
            idx2semid = dict()
            curr_keypoints = -np.ones((self.nclasses,), dtype=np.int)
            for i, kp in enumerate(keypoints[model_id]):
                curr_keypoints[kp[1]] = kp[0]
                idx2semid[i] = kp[1]
            self.keypoints.append(curr_keypoints)
            self.rotation_infos.append(rotation_groups[model_id])
            self.idx2semids.append(idx2semid)
            self.pcds.append(naive_read_pcd(fn)[0])
            self.mesh_names.append(model_id)

    def __getitem__(self, idx):
        pc = self.pcds[idx]
        label = self.keypoints[idx]
        keypoint_coords = np.array([pc[idx] for idx in label if idx >= 0])
        rotation_info = self.rotation_infos[idx]
        idx2semid = self.idx2semids[idx]
        mesh_name = self.mesh_names[idx]
        
        if self.cfg.normalize_pc:
            pc = normalize_pc(pc)
        
        labels = [label]
        if len(rotation_info) > 0 and not self.catg == 'chair':  # rotational axis exists
            group_idxs = [info['kp_indexes'] for info in rotation_info]
            all_circle = not (False in [info['is_circle'] for info in rotation_info])
            
            group = group_idxs[0]
            group_points = np.array([pc[label[idx2semid[i]]] for i in group])
            rotation_center = np.mean(group_points, 0)
            rotation_center[1] = 0  # set y to 0
            rotation_multiplex = len(group)
            if all_circle:
                rotation_multiplex = 36
                
            for i in range(1, rotation_multiplex):
                angle = 2 * np.pi / rotation_multiplex * i
                keypoints_rotated = (keypoint_coords - rotation_center) @ np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]]).T + rotation_center
                label_rotated = -np.ones_like(label, dtype=np.int)
                label_rotated[label >= 0] = pairwise_distances_argmin(keypoints_rotated, pc)
                
                labels.append(label_rotated)
            
        if self.cfg.augmentation.gaussian_noise:
            pc = add_noise(pc, sigma=0.004, clip=0.01)

        if self.cfg.augmentation.translation:
            tr = np.random.rand() * 0.02
            rot = special_ortho_group.rvs(3)
            pc = pc @ rot
            pc += np.array([[tr, 0, 0]])
            pc = pc @ rot.T

        # allow rotation along gravity axis
        if self.cfg.augmentation.rot_gravity:
            angle = np.random.rand() * 2 * np.pi
            rot = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
            pc = (rot @ pc.T).T
            
        return pc.astype(np.float32), labels, mesh_name

    def __len__(self):
        return len(self.pcds)
