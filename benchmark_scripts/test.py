from utils import load_geodesics
import hydra
import omegaconf
import logging
logger = logging.getLogger(__name__)
import os
import torch
import numpy as np
import importlib
import dataset
from utils import ModelWrapper
import torch.nn.functional as F
from metrics import eval_iou, eval_map
from tqdm import tqdm
import utils
from metrics import eval_pck


def test(cfg):
    name = cfg.class_name
    KeypointDataset = getattr(dataset, 'Keypoint{}Dataset'.format(cfg.task.capitalize()))

    test_dataset = KeypointDataset(cfg, 'test')
    
    geo_dists = load_geodesics(test_dataset, 'test')
    
    cfg.num_classes = test_dataset.nclasses
    model_impl = getattr(importlib.import_module('.{}'.format(cfg.network.name), package='models'), '{}Model'.format(cfg.task.capitalize()))(cfg).cuda()
    model = ModelWrapper(model_impl).cuda()

    model.load_state_dict(torch.load('best.pth'))
    model.eval()
    
    if cfg.task == 'saliency':
        pred_all_map = {}
        pred_all_iou = {}
        gt_all = {}
        pred_all_iou[name] = {}
        pred_all_map[name] = {}
        gt_all[name] = {}
        
        for i in range(len(test_dataset)):
            if test_dataset.mesh_names[i] not in pred_all_map[name]:
                pred_all_map[name][test_dataset.mesh_names[i]] = []
            if test_dataset.mesh_names[i] not in pred_all_iou[name]:
                pred_all_iou[name][test_dataset.mesh_names[i]] = []

            if test_dataset.mesh_names[i] not in gt_all[name]:
                gt_all[name][test_dataset.mesh_names[i]] = []

        for i, data in enumerate(test_dataset):
            pc, label, _ = data
            data = [np.array([item]) for item in data]
            with torch.no_grad():
                logits = model(data)
                logits = F.softmax(logits, dim=-1)[0]

            pred_all_map[name][test_dataset.mesh_names[i]].extend(list(zip(np.arange(pc.shape[0]), logits[:, 1].cpu().numpy())))
            pred_all_iou[name][test_dataset.mesh_names[i]].extend(np.where(logits[:, 1].cpu().numpy() > cfg.iou_thresh)[0])

            for kp in np.where(label == 1)[0]:
                gt_all[name][test_dataset.mesh_names[i]].append(kp)

        for i in range(11):
            dist_thresh = 0.01 * i
            rec, prec, ap = eval_map(pred_all_map, gt_all, geo_dists, dist_thresh=dist_thresh)
            logger.info('mAP-{:.2f}: {:.3f}'.format(dist_thresh, list(ap.values())[0]))
            
        for i in range(11):
            dist_thresh = 0.01 * i
            iou = eval_iou(pred_all_iou, gt_all, geo_dists, dist_thresh=dist_thresh)
            logger.info('mIoU-{:.2f}: {:.3f}'.format(dist_thresh, list(iou.values())[0]))
            
    elif cfg.task == 'correspondence':
        pcs = []
        gt_kps = []
        pred_kps = []
        geos = []
        for data in tqdm(test_dataset):
            pc, kp_index, mesh_name = data
            data = [np.array([item]) for item in data]
            with torch.no_grad():
                logits = model(data)
            pred_index = torch.argmax(logits, dim=1)[0].cpu().numpy()
            
            # consider rotational symmetry and choose the best target
            loss_rot = []
            for rot_kp_index in kp_index:
                loss_rot.append(F.cross_entropy(logits, torch.from_numpy(rot_kp_index[None]).long().cuda(), ignore_index=-1))
            best_kp_index = kp_index[torch.argmin(torch.stack(loss_rot))]
            
            pcs.append(pc)
            gt_kps.append(best_kp_index)
            pred_kps.append(pred_index)
            geos.append(geo_dists[mesh_name])

        pck = eval_pck(np.stack(pcs), np.stack(gt_kps), np.stack(pred_kps), np.stack(geos))
        for i, corr in enumerate(pck):
            logger.info('PCK-{:.2f}: {:.3f}'.format(i * 0.01, corr))
    else:
        raise Exception('unknown task')
    

@hydra.main(config_path='config', config_name='config')
def main(cfg):
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.log_path = '{}_log'.format(cfg.task)
    logger.info(cfg.pretty())
    test(cfg)


if __name__ == '__main__':
    main()