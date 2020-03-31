import torch
from train_kp_bin import NAMES2ID, KeypointDataset
import os
import h5py
from models.net import RSNet
from models.point_net import PointNetDenseCls
from models.pointconv import PointConvDensityClsSsg
from models.dgcnn import DGCNN
from models.graphcnn import GraphConvNet
from models.spidercnn import Spidercnn_seg_fullnet
from models.RSCNN.rscnn import RSCNN_MSN
import torch.nn.functional as F
from evaluate_map import eval_map
from evaluate_iou import eval_iou
import numpy as np

EVAL_MAP = False
IOU_THRESH = 0.1

if __name__ == '__main__':

    MODE = 'geo'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    names = list(NAMES2ID.keys())

    names = ["airplane", "bathtub", "bed", "bottle", "cap", "car", "chair", "guitar", "helmet", "knife", "laptop",
             "motorcycle", "mug", "skateboard", "table", "vessel"]

    net = "rscnn"

    pred_all_map = {}
    pred_all_iou = {}
    gt_all = {}
    for name in names:
        print(name)
        pred_all_iou[name] = {}
        pred_all_map[name] = {}
        gt_all[name] = {}
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), net + '_binlog_{}_{}'.format(name, MODE))
        if MODE == 'geo':
            fn = '/data/home/xxx/project/dataset/geo_divided/{}_test.h5'.format(name)
        elif MODE == 'gt':
            fn = '/data/home/xxx/project/dataset/gt_divided/{}_test.h5'.format(name)
        elif MODE == 'nms':
            fn = '/data/home/xxx/project/dataset/nms_divided/{}_test.h5'.format(name)
        else:
            raise ValueError('Invalid mode!')

        test_dataset = KeypointDataset(fn=fn, train=False)
        for i in range(len(test_dataset)):
            if test_dataset.mesh_names[i] not in pred_all_map[name]:
                pred_all_map[name][test_dataset.mesh_names[i]] = []
            if test_dataset.mesh_names[i] not in pred_all_iou[name]:
                pred_all_iou[name][test_dataset.mesh_names[i]] = []

            if test_dataset.mesh_names[i] not in gt_all[name]:
                gt_all[name][test_dataset.mesh_names[i]] = []

        # Load different models
        model = RSNet(test_dataset.nclasses).to(device)
        # model = RSCNN_MSN(test_dataset.nclasses).to(device)
        # model = DGCNN(test_dataset.nclasses).to(device)
        # model = PointConvDensityClsSsg(test_dataset.nclasses).to(device)
        # model = PointNetDenseCls(test_dataset.nclasses).to(device)
        # model = GraphConvNet([3, 1024, 5, 1024, 5], [512, test_dataset.nclasses]).to(device)
        # model = Spidercnn_seg_fullnet(test_dataset.nclasses).to(device)
        model.load_state_dict(torch.load(os.path.join(log_dir, 'pck_best.pth')))
        model.eval()

        for i, data in enumerate(test_dataset):
            pc, label = data

            with torch.no_grad():
                logits = model(torch.from_numpy(pc[None, ...]).transpose(1,2).cuda())
                logits = F.softmax(logits, dim=-1)

        # if EVAL_MAP:
            pred_all_map[name][test_dataset.mesh_names[i]].extend(list(zip(np.arange(pc.shape[0]), logits[0, :, 1].cpu().numpy())))
        # else:
            pred_all_iou[name][test_dataset.mesh_names[i]].extend(np.where(logits[0, :, 1].cpu().numpy() > IOU_THRESH)[0])

            for kp in np.where(label == 1)[0]:
                gt_all[name][test_dataset.mesh_names[i]].append(kp)

    # need 30+ G memory!!!
    geo_dists = {}
    for name in names:
        file_path = os.path.join('/data/home/xxx/project/dataset/kp/data_unified',
                                 'model_geodesic_mat/{}_geo_dists.h5'.format(NAMES2ID[name]))

        with h5py.File(file_path, 'r') as f:
            dist_mats = f['geo_dists'][:]
            mesh_names = f['mesh_names'][:].tolist()
        for i in range(len(mesh_names)):
            if mesh_names[i] not in geo_dists:
                geo_dists[mesh_names[i]] = dist_mats[i]

# if EVAL_MAP:
    for i in range(11):
        dist_thresh = 0.01 * i
        rec, prec, ap = eval_map(pred_all_map, gt_all, geo_dists, dist_thresh=dist_thresh)

        ap_l = list(ap.values())
        s = ""
        for x in ap_l:
            s += "{}\t".format(x)
        print (s)
# else:
    print ("iou")
    for i in range(11):
        dist_thresh = 0.01 * i
        iou = eval_iou(pred_all_iou, gt_all, geo_dists, dist_thresh=dist_thresh)

        iou_l = list(iou.values())
        s = str(dist_thresh)
        for x in iou_l:
            s += "\t{}".format(x)
        print(s)

    print("Network: {}".format(net))