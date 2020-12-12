import numpy as np


def voc_ap(rec, prec):
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_det_cls_map(pred, gt, geo_dists, dist_thresh=0.1):
    # construct gt objects
    class_recs = {}  # {mesh name: {'kp': kp list, 'det': matched list}}
    npos = 0
    for mesh_name in gt.keys():
        keypoints = np.array(gt[mesh_name])
        det = [False] * len(keypoints)
        npos += len(keypoints)
        class_recs[mesh_name] = {'kp': keypoints, 'det': det}
    # pad empty list to all other imgids
    for mesh_name in pred.keys():
        if mesh_name not in gt:
            class_recs[mesh_name] = {'kp': np.array([]), 'det': []}

    # construct dets
    mesh_names = []
    confidence = []
    KP = []
    for mesh_name in pred.keys():
        for kp, score in pred[mesh_name]:
            mesh_names.append(mesh_name)
            confidence.append(score)
            KP.append(kp)
    confidence = np.array(confidence)
    KP = np.array(KP)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    KP = KP[sorted_ind, ...]
    mesh_names = [mesh_names[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(mesh_names)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        # if d % 100 == 0:
        #     print(d)
        R = class_recs[mesh_names[d]]
        kp = KP[d]
        dmin = np.inf
        KPGT = R['kp']

        if KPGT.size > 0:
            # compute overlaps
            for j in range(KPGT.shape[0]):
                geo_dist = geo_dists[mesh_names[d]][kp, KPGT[j]]
                if geo_dist < dmin:
                    dmin = geo_dist
                    jmin = j

        # print dmin
        if dmin < dist_thresh:
            if not R['det'][jmin]:
                tp[d] = 1.
                R['det'][jmin] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # print('NPOS: ' + str(npos))
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)

    return rec, prec, ap


def eval_map(pred_all, gt_all, geo_dists, dist_thresh=0.1):
    """ Generic functions to compute precision/recall for keypoint detection
        for multiple classes.
        Input:
            pred_all: map of {classname: {meshname: [(kp, score)]}}
            gt_all: map of {classname: {meshname: [kp]}}
            dist_thresh: scalar, iou threshold
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """

    rec = {}
    prec = {}
    ap = {}
    for classname in gt_all.keys():
        rec[classname], prec[classname], ap[classname] = eval_det_cls_map(pred_all[classname], gt_all[classname], geo_dists, dist_thresh)

    return rec, prec, ap


def eval_det_cls_iou(pred, gt, geo_dists, dist_thresh=0.1):
    npos = 0
    fp_sum = 0
    fn_sum = 0
    for mesh_name in gt.keys():
        gt_kps = np.array(gt[mesh_name]).astype(np.int32)
        npos += len(gt_kps)
        pred_kps = np.array(pred[mesh_name]).astype(np.int32)
        fp = np.count_nonzero(np.all(geo_dists[mesh_name][pred_kps][:, gt_kps] > dist_thresh, axis=-1))
        fp_sum += fp
        fn = np.count_nonzero(np.all(geo_dists[mesh_name][gt_kps][:, pred_kps] > dist_thresh, axis=-1))
        fn_sum += fn

    return (npos - fn_sum) / np.maximum(npos + fp_sum, np.finfo(np.float64).eps)


def eval_iou(pred_all, gt_all, geo_dists, dist_thresh=0.05):
    """ Generic functions to compute iou for keypoint detection
        for multiple classes.
        Input:
            pred_all: map of {classname: {meshname: [kp]}}
            gt_all: map of {classname: {meshname: [kp]}}
            dist_thresh: scalar, iou threshold
        Output:
            iou
    """

    iou = {}
    for classname in gt_all.keys():
        iou[classname] = eval_det_cls_iou(pred_all[classname], gt_all[classname], geo_dists, dist_thresh)

    return iou


def eval_pck(P, KP, pred_KP, geo_dists):
    n_data = P.shape[0]
    
    threshold_list = [0.01 * i for i in range(11)]

    pcks = []
    for b in range(n_data):
        # NOTE:
        # Skip if the keypoint does not exist.
        valid_idx = [i for i in range(KP.shape[1]) if KP[b, i] >= 0]
        dists = []
        for idx in valid_idx:
            dists.append(geo_dists[b][KP[b, idx], pred_KP[b, idx]])
        dists = np.stack(dists)
        pck = np.stack([np.sum(dists < th) / len(valid_idx) for th in threshold_list])
        pcks.append(pck)
    
    pcks = np.stack(pcks)
    return np.mean(pcks, 0)