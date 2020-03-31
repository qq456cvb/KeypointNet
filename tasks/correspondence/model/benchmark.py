import torch
import torch.nn as nn
import torch.nn.functional as F

from model.point_net import *
from model.options import Options
from model.dgcnn import *
from model.spidercnn import Spidercnn_seg_fullnet as spidercnn
from model.graphcnn import *
from model.pointconv import *
# from model.SONet.segmenter import Model as sonet
from model.spidercnn import Spidercnn_seg_fullnet
# from model.pointnet2.net import Pointnet2SSG
from utils.tools import *
from utils.losses import *
# from model.RSCNN.rscnn import RSCNN_MSN


class BenchMark(nn.Module):
    def __init__(self, cfg):
        super(BenchMark, self).__init__()
        self.num_points = int(cfg['num_points'])
        self.task = cfg['task_type']
        self.net = cfg["net"]

        if self.task == 'pck':
            self.num_kps = int(cfg['num_kps'])
        elif self.task == 'iou':
            self.num_kps = 2
        else:
            raise NotImplementedError

        if self.net == "pointnet":
            self.backbone = PointNetDenseCls(self.num_kps, cfg=cfg)
        elif self.net == "dgcnn":
            self.backbone = DGCNN(self.num_kps, cfg=cfg)
        elif self.net == "pointconv":
            self.backbone = PointConvDensityClsSsg(self.num_kps)
        elif self.net == "graphcnn":
            self.backbone = GraphConvNet([3, 1024, 5, 1024, 5], [512, self.num_kps])
        # elif self.net == "sonet":
        #     opt = Options().parse()
        #     opt.classes = self.num_kps
        #     self.backbone = sonet(opt)
        elif self.net == "spidercnn":
            self.backbone = Spidercnn_seg_fullnet(self.num_kps)
        elif self.net == "rscnn":
            from model.RSCNN.rscnn import RSCNN_MSN
            self.backbone = RSCNN_MSN(self.num_kps)

    def forward(self, input_):
        if self.task == 'pck':
            pcd = input_[0]
            pcd = pcd.cuda()
            logits = self.backbone(pcd.transpose(1, 2).cuda())
        elif self.task == 'iou':
            pcd = input_[0]
            pcd = pcd.cuda()
            logits = self.backbone(pcd.transpose(1, 2))
        else:
            raise NotImplementedError
        return logits

class BenchMarkLoss(nn.Module):
    def __init__(self, cfg):
        super(BenchMarkLoss, self).__init__()
        self.task = cfg['task_type']
        self.pck_criterion = PCKLoss()
        self.iou_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.cfg = cfg

    def forward(self, input_var):
        loss = {}
        if self.task == 'pck':
            pred, kps = input_var
            kps_one_hot = convert_kp_to_one_hot(kps, pred.size(1))
            loss_pck, pred_kps = self.pck_criterion(pred, kps_one_hot.cuda())
            loss["total"] = loss_pck
            return loss
        elif self.task == 'iou':
            pred, kps = input_var
            pred = pred.squeeze()
            binary_label = convert_kp_to_binary(kps, pred.size(1))
            l = 0.
            for i in range(len(pred)):
                loss_iou = self.iou_criterion(pred[i], binary_label[i].long().cuda())
                l += loss_iou
            loss["total"] = l / len(pred)
        else:
            raise NotImplementedError
        return loss


class BenchMarkMetric(nn.Module):
    def __init__(self, cfg):
        super(BenchMarkMetric, self).__init__()
        self.task = cfg['task_type']
        self.num_kps = int(cfg['num_kps'])
        self.prob_threshold = cfg['prob_threshold']
        self.geo_threshold = cfg['geo_threshold']
        if self.task == 'pck':
            self.__name__ = 'PCK'
        elif self.task == 'iou':
            self.__name__ = 'IoU'

    def forward(self, input):
        if self.task == 'pck':
            pts, gt_index, pred_index = input
            pts = pts.cpu().numpy()
            gt_index = gt_index.cpu().numpy().astype(np.int32)
            pred_index = pred_index.cpu().numpy().astype(np.int32)
            dist_info = pck(pts, gt_index, pred_index)
            corr_list = calculate_correspondence(dist_info, gt_index)
            return np.array(corr_list)
        elif self.task == 'iou':
            gt_index, logits, geo_dists = input
            logits = F.softmax(logits, dim=-1)
            print("prob: {}".format(logits[0]))
            print("num pos: {}".format(logits[0, :, 1].max()))
            iou = IoUMetric(gt_index, logits, geo_dists, prob_thr=self.prob_threshold, geo_thr=self.geo_threshold)
            map = APMetric(gt_index, logits, geo_dists, prob_thr=self.prob_threshold, geo_thr=self.geo_threshold)
            return iou, map
        else:
            raise NotImplementedError


if __name__ == '__main__':
    a = torch.rand(32, 2048, 3)
    b = torch.zeros(32, 2048, 10)
