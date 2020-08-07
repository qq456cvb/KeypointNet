import os
import h5py
import torch
import numpy as np
import torch.utils.data as DT
import scipy
import json
import hydra
from glob import glob

import sys
sys.path.append('..')
from tasks.correspondence.utils.tools import *
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, SequentialSampler
import itertools
from sklearn.metrics import pairwise_distances_argmin
import visdom
from scipy.stats import special_ortho_group


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
BASEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')


def normalize_pc(pc):
    pc = pc - pc.mean(0)
    pc /= np.max(np.linalg.norm(pc, axis=-1))
    return pc


def my_collate(batch):
    pcs = torch.stack([torch.from_numpy(item[0]) for item in batch]).float()
    labels = [torch.from_numpy(np.stack(item[1])).long() for item in batch]
    mesh_names = [item[2] for item in batch]
    return pcs, labels, mesh_names


class KeypointDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.class_name = cfg.class_name
        self.aug = cfg.data_aug
        self.rot_gravity = cfg.rot_gravity
        self.training = 'train' in split
        self.rot = (self.training and cfg.rot_exp.rot_train) or (not self.training and cfg.rot_exp.rot_test)
            
        annots = json.load(open(os.path.join(BASEDIR, cfg.data.annot_path)))
        annots = [annot for annot in annots if annot['class_id'] == NAMES2ID[cfg.class_name]]
        keypoints = dict([(annot['model_id'], [(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']]) for annot in annots])
        rotation_groups = dict([(annot['model_id'], annot['symmetries']['rotation']) for annot in annots])
        
        self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
        
        split_models = open(os.path.join(BASEDIR, split)).readlines()
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
        
        pc = normalize_pc(pc)
        
        labels = [label]
        if len(rotation_info) > 0 and not self.class_name == 'chair':  # rotational axis exists
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
                keypoints_rotated = (keypoint_coords - rotation_center) @ rotmat(0, angle, 0, False).T + rotation_center
                label_rotated = -np.ones_like(label, dtype=np.int)
                label_rotated[label >= 0] = pairwise_distances_argmin(keypoints_rotated, pc)
                
                labels.append(label_rotated)
            
        if self.aug:
            pc = add_noise(pc, sigma=0.004, clip=0.01)

            tr = np.random.rand() * 0.02
            rot = rnd_rot()
            pc = pc @ rot
            pc += np.array([[tr, 0, 0]])
            pc = pc @ rot.T

        # allow rotation along gravity axis
        if self.rot_gravity and not self.rot:
            rot = rotmat(0, np.arccos(np.random.rand() * 2 - 1), 0, False)
            pc = (rot @ pc.T).T
        
        if self.rot:
            rot = special_ortho_group.rvs(3)
            pc = pc @ rot
            
        return pc.astype(np.float32), labels, mesh_name

    def __len__(self):
        return len(self.pcds)


@hydra.main(config_path='../config/config.yaml', strict=False)
def main(cfg):
    vis = visdom.Visdom(port=21391)
    ds = KeypointDataset(cfg, cfg.data.train_txt)
    val_data = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        collate_fn=my_collate
    )
    for d in val_data:
        pc, labels, mesh_names = d
        import pdb; pdb.set_trace()
        pc = pc[0]
        labels = labels[0]
        
        for label in labels:
            colors = np.ones((pc.shape[0]), dtype=np.int)
            for i, kp_idx in enumerate(label):
                if kp_idx >= 0:
                    colors[kp_idx] = i + 2
            vis.scatter(X=pc, Y=colors, win='test')
            import pdb; pdb.set_trace()
        

if __name__ == "__main__":
    main()