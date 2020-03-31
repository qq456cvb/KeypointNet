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


def eval_det_cls(pred, gt, geo_dists, dist_thresh=0.1):
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


def eval_map(pred_all, gt_all, geo_dists, dist_thresh=0.05):
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
        # print('Computing AP for class: ' + classname)
        rec[classname], prec[classname], ap[classname] = eval_det_cls(pred_all[classname], gt_all[classname], geo_dists, dist_thresh)
        # print(classname + ':' + str(ap[classname]))
        # logger.info(classname + ' rec:' + str(rec[classname]))

    return rec, prec, ap


