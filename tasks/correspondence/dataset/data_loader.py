import os
import h5py
import torch
import numpy as np
import torch.utils.data as DT
import scipy

from utils.tools import *
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, SequentialSampler

NAME2ID = {"airplane":    "02691156",
           "bathtub":     "02808440",
           "bed":         "02818832",
           "bench":       "02828884",
           "bottle":      "02876657",
           "bus":         "02924116",
           "cap":         "02954340",
           "car":         "02958343",
           "chair":       "03001627",
           "dishwasher":  "03207941",
           "display":     "03211117",
           "earphone":    "03261776",
           "faucet":      "03325088",
           "guitar":      "03467517",
           "helmet":      "03513137",
           "knife":       "03624134",
           "lamp":        "03636649",
           "laptop":      "03642806",
           "motorcycle":   "03790512",
           "mug":         "03797390",
           "pistol":      "03948459",
           "rocket":      "04099429",
           "skateboard":  "04225987",
           "table":       "04379243",
           "vessel":      "04530566"}

class KPDataset(DT.Dataset):
    def __init__(self, cfg, flag='train'):
        super(KPDataset, self).__init__()
        self.num_points = int(cfg['num_points'])
        self.task = cfg['task_type']
        self.category = cfg['category']
        self.data_path = os.path.join(cfg["data_path"], cfg['data_type']+'_divided')
        self.train_flag = flag

        with h5py.File(os.path.join(self.data_path, self.category+'_' + flag +'.h5')) as f:
            self.pcds = f['point_clouds'][:]
            self.keypoints = f['keypoints'][:]
            self.mesh_names = f['mesh_names'][:]

        if self.task == "iou":
            with h5py.File(os.path.join(cfg["data_path"], "kp/data_unified/model_geodesic_mat", NAME2ID[self.category] + "_geo_dists.h5")) as f:
                geo_dists = f["geo_dists"][:]
                geo_mesh_names = f["mesh_names"][:]

            indices = []
            for m in self.mesh_names:
                idx = np.where(geo_mesh_names == m)[0][0]
                indices.append(idx)
            self.dists = geo_dists[indices]

        self.len = self.pcds.shape[0]

    def __getitem__(self, item):
        pcd = self.pcds[item]
        keypoint_index = np.array(self.keypoints[item], dtype=np.int32)

        if self.train_flag:
            pcd = add_noise(pcd, sigma=0.004, clip=0.1)
            tr = np.random.rand() * 0.02
            rot = rnd_rot()
            pcd = pcd @ rot
            pcd += np.array([[tr, 0, 0]])
            pcd = pcd @ rot.T

        if self.task == "iou":
            dist = self.dists[item]
            return torch.tensor(pcd).float(), torch.tensor(keypoint_index), torch.tensor(dist).float()
        else:
            return torch.tensor(pcd).float(), torch.tensor(keypoint_index), 0

    def __len__(self):
        return self.len




if __name__ == '__main__':
    # import h5py
    # import numpy as np
    # filename = 'syncspec/Chair_test.h5'
    # f = h5py.File(filename, 'r')
    # keys = ['keypoints', 'mesh_names', 'point_clouds']
    # key_points = f[keys[0]]
    # key_points = np.array(key_points)
    # pcds = f[keys[-1]]
    # pcds = np.array(pcds)
    # print(key_points)
    # print(pcds.shape)
    cfg = get_cfg()
