import numpy as np


def pck(P, KP, pred_KP):
    n_data = P.shape[0]
    n_points = P.shape[1]
    n_labels = KP.shape[1]
    K = pred_KP.shape[1]

    # dists_info: (point_cloud_index, label, basis_index, distance)
    dists_info = []

    for k in range(n_data):
        # NOTE:
        # Skip if the keypoint does not exist.
        labels = [i for i in range(n_labels) if KP[k, i] >= 0]

        # Find the closest prediction (w/o matching).
        for i, label in enumerate(labels):
            idx_i = KP[k, label] # KP index
            assert (idx_i < n_points)
            p_i = P[k, idx_i] # KP coordinates

            idx_j = pred_KP[k, label]
            p_j = P[k, idx_j]

            all_dists =  np.linalg.norm(p_i - p_j)
            dists_info.append((k, i, i, all_dists))

    dists_info = np.array(dists_info)

    return dists_info


def calculate_correspondence(dists_info, KP):
    """
    Calculate the correspondence under different distance threshold
    :param dists_info: (point_cloud_index, label, basis_index, distance)
    :param KP: (B, 10)
    :return: correspondence list
    """
    threshold_list = [0.01*i for i in range(11)]
    corr_list = []
    num_KP = len(np.where(KP != -1)[0])
    for i, threshold in enumerate(threshold_list):
        correct = 0
        for j, dists in enumerate(dists_info):
            if dists[3] <= threshold:
                correct += 1
        corr_list.append(correct*1.0/num_KP)
    return corr_list
