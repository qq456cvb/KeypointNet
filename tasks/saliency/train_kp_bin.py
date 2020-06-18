import os
import numpy as np

import hydra
from hydra import utils
import torch
import torch.nn as nn
import math
import shutil
from itertools import chain
from glob import glob
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import json
import io
import logging
import h5py
logger = logging.getLogger(__name__)

from models.rsnet import RSNet
from models.point_net import PointNetDenseCls
from models.dgcnn import DGCNN
from models.graphcnn import GraphConvNet
from models.spidercnn import Spidercnn_seg_fullnet
from models.pointconv import PointConvDensityClsSsg
# from models.RSCNN.rscnn import RSCNN_MSN
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
BASEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')


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


class KeypointDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        super().__init__()
        self.aug = cfg.data_aug
        # self.rot_gravity = cfg.rot_gravity
        self.catg = cfg.class_name
        self.rot_train = cfg.rot_exp.rot_train
        self.rot_test = cfg.rot_exp.rot_test
        if split == 'train':
            self.rot_exp = self.rot_train
        elif split == 'val' or split == 'test':
            self.rot_exp = self.rot_test
        
        if split== 'val':
            split = 'validate'
        filename = os.path.join(BASEDIR,
            cfg.data.data_path, '{}_{}.h5'.format(self.catg, split))
        with h5py.File(filename, 'r') as f:
            self.pcds = f['point_clouds'][:]
            self.keypoints = f['keypoints'][:]
            self.mesh_names = f['mesh_names'][:]
        
        self.nclasses = 2

    def __getitem__(self, idx):
        pc = self.pcds[idx]
        label = self.keypoints[idx]
        bin_label = np.zeros((pc.shape[0],), dtype=np.int64)
        bin_label[label] = 1
        if self.aug:
            pc = add_noise(pc, sigma=0.004, clip=0.01)

            tr = np.random.rand() * 0.02
            rot = rnd_rot()
            pc = pc @ rot
            pc += np.array([[tr, 0, 0]])
            pc = pc @ rot.T

        if self.rot_exp:
            rot = special_ortho_group.rvs(3)
            pc = pc @ rot

        # allow rotation along gravity axis
        # if self.rot_gravity:
        #     rot = rotmat(0, np.arccos(np.random.rand() * 2 - 1), 0, False)
        #     pc = (rot @ pc.T).T
        return pc.astype(np.float32), bin_label

    def __len__(self):
        return len(self.pcds)


def train(cfg):
    name = cfg.class_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log_dir = os.path.curdir

    train_dataset = KeypointDataset(cfg, 'train')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    val_dataset = KeypointDataset(cfg, 'val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, num_workers=4)

    # Load different models
    if cfg.network == 'rsnet':
        model = RSNet(train_dataset.nclasses).to(device)
    elif cfg.network == 'pointnet':
        model = PointNetDenseCls(train_dataset.nclasses).to(device)
    elif cfg.network == 'dgcnn':
        model = DGCNN(train_dataset.nclasses).to(device)
    elif cfg.network == 'graphconv':
        model = GraphConvNet([3, 1024, 5, 1024, 5], [512, train_dataset.nclasses]).to(device)
    elif cfg.network == 'spidercnn':
        model = Spidercnn_seg_fullnet(train_dataset.nclasses).to(device)
    elif cfg.network == 'rscnn':
        model = RSCNN_MSN(train_dataset.nclasses).to(device)
    elif cfg.network == 'pointconv':
        model = PointConvDensityClsSsg(train_dataset.nclasses).to(device)
    else:
        logger.error('unrecognized network name')
        exit()
    
    logger.info('Start training on binary keypoint detection...')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1).to(device)
    accum_iter, tot_iter = 0, 0
    accum_loss = {'loss': 0}

    writer = SummaryWriter(log_dir)

    best_loss = 1e10
    for epoch in range(cfg.max_epoch + 1):
        train_iter = train_dataloader.__iter__()

        # Training
        model.train()
        for i, data in enumerate(train_iter):
            pc, label = data

            logits = model(pc.transpose(1,2).to(device))
            loss = criterion(logits.reshape(-1, 2), label.view(-1).cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum_loss['loss'] += loss.item()
            accum_iter += 1
            tot_iter += 1

            if tot_iter % 10 == 0 or tot_iter == 1:
                for key in accum_loss:
                    writer.add_scalar('Train/%s' % key, accum_loss[key] / accum_iter, tot_iter)
                    writer.flush()

                    logger.info(
                        f'Iter: {tot_iter}, Epoch: {epoch}, {key}: {accum_loss[key] / accum_iter}'
                    )
                    accum_loss[key] = 0
                accum_iter = 0

        model.eval()
        # validation loss
        val_loss = {'loss': 0}
        for i, data in enumerate(val_dataloader):
            pc, label = data

            with torch.no_grad():
                logits = model(pc.transpose(1,2).to(device))

                loss = criterion(logits.reshape(-1, 2), label.view(-1).cuda())

                val_loss['loss'] += loss.item()

        if val_loss['loss'] / len(val_dataloader) < best_loss:
            logger.info("best epoch: {}".format(epoch))
            best_loss = val_loss['loss'] / len(val_dataloader)
            torch.save(model.state_dict(), os.path.join(log_dir, 'pck_best.pth'))

        for key in val_loss:
            writer.add_scalar('Val/%s' % key, val_loss[key] / len(val_dataloader), epoch)
            # writer.add_scalar('ValPck0.01', corr[0])
            writer.flush()
            logger.info(
                f'Epoch: {epoch}, Val {key}: {val_loss[key] / len(val_dataloader)}'
            )

    writer.close()


@hydra.main(config_path='config/train.yaml')
def main(cfg):
    logger.info(cfg.pretty())
    train(cfg)


if __name__ == '__main__':
    main()