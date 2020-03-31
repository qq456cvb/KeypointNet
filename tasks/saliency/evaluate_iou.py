import numpy as np


def eval_det_cls(pred, gt, geo_dists, dist_thresh=0.1):
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
    """ Generic functions to compute precision/recall for keypoint detection
        for multiple classes.
        Input:
            pred_all: map of {classname: {meshname: [kp]}}
            gt_all: map of {classname: {meshname: [kp]}}
            dist_thresh: scalar, iou threshold
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """

    iou = {}
    for classname in gt_all.keys():
        # print('Computing IOU for class: ' + classname)
        iou[classname] = eval_det_cls(pred_all[classname], gt_all[classname], geo_dists, dist_thresh)
        # print(classname + ':' + str(iou[classname]))
        # logger.info(classname + ' rec:' + str(rec[classname]))

    return iou


