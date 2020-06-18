import os
import h5py
import torch
import numpy as np
import torch.utils.data as DT
import scipy
import json
from glob import glob

import sys
sys.path.append('..')
from utils.tools import *
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, SequentialSampler


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


class KeypointDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        # self.aug = cfg.data_aug
        # self.catg = NAMES2ID[cfg.class_name]
        # self.rot_train = cfg.rot_exp.rot_train
        # self.rot_test = cfg.rot_exp.rot_test
        # if split == 'train':
        #     self.rot_exp = self.rot_train
        # elif split == 'val' or split == 'test':
        #     self.rot_exp = self.rot_test
        
        # filename = os.path.join(BASEDIR,
        #     cfg.data.data_path, '{}.h5'.format(self.catg))
        # with h5py.File(filename, 'r') as f:
        #     self.pcds = f['point_clouds'][:]
        #     self.keypoints = f['keypoints'][:]
        #     self.mesh_names = f['mesh_names'][:]

        # num_train = int(self.pcds.shape[0] * 0.7)
        # num_divide = int(self.pcds.shape[0] * 0.85)

        # if split == 'train':
        #     self.pcds = self.pcds[:num_train]
        #     self.keypoints = self.keypoints[:num_train]
        #     self.mesh_names = self.mesh_names[:num_train]
        # elif split == 'val':
        #     self.pcds = self.pcds[num_train:num_divide]
        #     self.keypoints = self.keypoints[num_train:num_divide]
        #     self.mesh_names = self.mesh_names[num_train:num_divide]
        # elif split == 'test':
        #     self.pcds = self.pcds[num_divide:]
        #     self.keypoints = self.keypoints[num_divide:]
        #     self.mesh_names = self.mesh_names[num_divide:]
        # elif split == 'all':
        #     pass
        # else:
        #     raise ValueError("{}".format(split))
        
        self.nclasses = 2
        self.aug = cfg.data_aug
        self.rot_gravity = cfg.rot_gravity
            
        annots = json.load(open(os.path.join(BASEDIR, cfg.data.annot_path)))
        annots = [annot for annot in annots if annot['class_id'] == NAMES2ID[cfg.class_name]]
        keypoints = dict([(annot['model_id'], [(kp_info['pcd_info']['point_index'], kp_info['semantic_id']) for kp_info in annot['keypoints']]) for annot in annots])
        
        self.nclasses = max([max([kp_info['semantic_id'] for kp_info in annot['keypoints']]) for annot in annots]) + 1
        
        split_models = open(os.path.join(BASEDIR, split)).readlines()
        split_models = [m.split('-')[-1].rstrip('\n') for m in split_models]
        
        self.pcds = []
        self.keypoints = []
        self.mesh_names = []
        for fn in glob(os.path.join(BASEDIR, cfg.data.pcd_root, NAMES2ID[cfg.class_name], '*.pcd')):
            model_id = os.path.basename(fn).split('.')[0]
            if model_id not in split_models:
                continue
            
            curr_keypoints = np.ones((self.nclasses,), dtype=np.int)
            for kp in keypoints[model_id]:
                curr_keypoints[kp[1]] = kp[0]
            self.keypoints.append(curr_keypoints)
            self.pcds.append(naive_read_pcd(fn)[0])
            self.mesh_names.append(model_id)


    def __getitem__(self, idx):
        pc = self.pcds[idx]
        label = self.keypoints[idx]
        if self.aug:
            pc = add_noise(pc, sigma=0.004, clip=0.01)

            tr = np.random.rand() * 0.02
            rot = rnd_rot()
            pc = pc @ rot
            pc += np.array([[tr, 0, 0]])
            pc = pc @ rot.T

        # allow rotation along gravity axis
        if self.rot_gravity:
            rot = rotmat(0, np.arccos(np.random.rand() * 2 - 1), 0, False)
            pc = (rot @ pc.T).T
        return pc.astype(np.float32), label

    def __len__(self):
        return len(self.pcds)

