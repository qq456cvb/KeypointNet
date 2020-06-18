import torch
from train_kp_bin import NAMES2ID, KeypointDataset, BASEDIR
import os
from models.rsnet import RSNet
from models.point_net import PointNetDenseCls
from models.pointconv import PointConvDensityClsSsg
from models.dgcnn import DGCNN
from models.graphcnn import GraphConvNet
from models.spidercnn import Spidercnn_seg_fullnet
# from models.RSCNN.rscnn import RSCNN_MSN
import torch.nn.functional as F
from evaluate_map import eval_map
from evaluate_iou import eval_iou
from sklearn import neighbors
from sklearn.utils.graph import graph_shortest_path
import numpy as np
import pickle
import hydra
import logging
from hydra import utils
from tqdm import tqdm
logger = logging.getLogger(__name__)


def gen_geo_dists(pc):
    graph = neighbors.kneighbors_graph(pc, 20, mode='distance', include_self=False)
    return graph_shortest_path(graph, directed=False)


@hydra.main(config_path='config/test.yaml')
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = cfg.class_name

    net = cfg.network

    pred_all_map = {}
    pred_all_iou = {}
    gt_all = {}
    pred_all_iou[name] = {}
    pred_all_map[name] = {}
    gt_all[name] = {}
    log_dir = os.path.curdir

    test_dataset = KeypointDataset(cfg, 'test')
    for i in range(len(test_dataset)):
        if test_dataset.mesh_names[i] not in pred_all_map[name]:
            pred_all_map[name][test_dataset.mesh_names[i]] = []
        if test_dataset.mesh_names[i] not in pred_all_iou[name]:
            pred_all_iou[name][test_dataset.mesh_names[i]] = []

        if test_dataset.mesh_names[i] not in gt_all[name]:
            gt_all[name][test_dataset.mesh_names[i]] = []

    # Load different models
    if cfg.network == 'rsnet':
        model = RSNet(test_dataset.nclasses).to(device)
    elif cfg.network == 'pointnet':
        model = PointNetDenseCls(test_dataset.nclasses).to(device)
    elif cfg.network == 'dgcnn':
        model = DGCNN(test_dataset.nclasses).to(device)
    elif cfg.network == 'graphconv':
        model = GraphConvNet([3, 1024, 5, 1024, 5], [512, test_dataset.nclasses]).to(device)
    elif cfg.network == 'spidercnn':
        model = Spidercnn_seg_fullnet(test_dataset.nclasses).to(device)
    elif cfg.network == 'rscnn':
        model = RSCNN_MSN(test_dataset.nclasses).to(device)
    elif cfg.network == 'pointconv':
        model = PointConvDensityClsSsg(test_dataset.nclasses).to(device)
    else:
        logger.error('unrecognized network name')
        exit()
    
    state_dict_tmp = torch.load('pck_best.pth')
    print(state_dict_tmp.keys())
    # state_dict = {}
    # for k, v in state_dict_tmp.items():
    #     state_dict[k.replace("backbone.","")] = v
    # model.load_state_dict(state_dict)
    model.load_state_dict(torch.load('pck_best.pth'))
    model.eval()

    for i, data in enumerate(test_dataset):
        pc, label = data

        with torch.no_grad():
            logits = model(torch.from_numpy(pc[None, ...]).transpose(1,2).to(device))
            logits = F.softmax(logits, dim=-1)

        pred_all_map[name][test_dataset.mesh_names[i]].extend(list(zip(np.arange(pc.shape[0]), logits[0, :, 1].cpu().numpy())))
        pred_all_iou[name][test_dataset.mesh_names[i]].extend(np.where(logits[0, :, 1].cpu().numpy() > cfg.iou_thresh)[0])

        for kp in np.where(label == 1)[0]:
            gt_all[name][test_dataset.mesh_names[i]].append(kp)

    # need a large amount of memory to load geodesic distances!!!
    if not os.path.exists(os.path.join(BASEDIR, 'cache')):
        os.makedirs(os.path.join(BASEDIR, 'cache'))
    if os.path.exists(os.path.join(BASEDIR, 'cache', '{}_geodists.pkl'.format(name))):
        logger.info('Found geodesic cache...')
        geo_dists = pickle.load(open(os.path.join(BASEDIR, 'cache', '{}_geodists.pkl'.format(name)), 'rb'))
    else:
        geo_dists = {}
        logger.info('Generating geodesics, this may take some time...')
        for i in tqdm(range(len(test_dataset.mesh_names))):
            if test_dataset.mesh_names[i] not in geo_dists:
                geo_dists[test_dataset.mesh_names[i]] = gen_geo_dists(test_dataset.pcds[i]).astype(np.float32)
        pickle.dump(geo_dists, open(os.path.join(BASEDIR, 'cache', '{}_geodists.pkl'.format(name)), 'wb'))

    for i in range(11):
        dist_thresh = 0.01 * i
        rec, prec, ap = eval_map(pred_all_map, gt_all, geo_dists, dist_thresh=dist_thresh)

        ap_l = list(ap.values())
        s = ""
        for x in ap_l:
            s += "{}\t".format(x)
        logger.info('mAP-{}: {}'.format(dist_thresh, s))
        
    for i in range(11):
        dist_thresh = 0.01 * i
        iou = eval_iou(pred_all_iou, gt_all, geo_dists, dist_thresh=dist_thresh)

        iou_l = list(iou.values())
        s = ""
        for x in iou_l:
            s += "{}\t".format(x)
        logger.info('mIoU-{}: {}'.format(dist_thresh, s))
    
    
if __name__ == '__main__':
    main()