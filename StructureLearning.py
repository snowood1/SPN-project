from SumProductNets import *
import numpy as np
from sklearn.metrics import mutual_info_score


k_mean_k = 2
k_mean_its = 20

dependent_threshold = 0.05


def train(data, rvs):
    root = split_instances(data, rvs)
    return SPN(root, rvs)


def split_instances(data, rvs):
    d, m = data.shape
    k = min(k_mean_k, m)

    centroid = data[:, np.random.choice(m, k, replace=False)].T
    distance = ((data - centroid[:, :, np.newaxis]) ** 2).sum(axis=1)
    clustering = distance.argmin(axis=0)

    for _ in range(k_mean_its):
        centroid = np.array([data[:, clustering == c].mean(axis=1) for c in range(k)])
        distance = ((data - centroid[:, :, np.newaxis]) ** 2).sum(axis=1)
        clustering = distance.argmin(axis=0)

    ch = list()
    w = list()
    for c in range(k):
        ch_data = data[:, clustering == c]
        _, ch_m = ch_data.shape
        ch.append(split_variables(ch_data, rvs))
        w.append(ch_m / m)

    return SumNode(ch, np.array(w))


def split_variables(data, rvs):
    d, m = data.shape

    ch = list()

    remaining_rvs_idx = set(range(d))
    for x_idx in range(d):
        if x_idx in remaining_rvs_idx:
            ch_rvs_idx = [x_idx]
            remaining_rvs_idx -= {x_idx}
            for y_idx in range(d):
                if y_idx in remaining_rvs_idx:
                    mi = mutual_info(data[x_idx], data[y_idx], rvs[x_idx], rvs[y_idx])
                    if mi > dependent_threshold:
                        remaining_rvs_idx -= {y_idx}
                        ch_rvs_idx.append(y_idx)

            if len(ch_rvs_idx) == 1 or (ch_data == ch_data[0]).all():
                for i in ch_rvs_idx:
                    ch.append(RVNode(rvs[i]))
            else:
                ch_data = data[ch_rvs_idx, :]
                ch.append(split_instances(ch_data, [rvs[i] for i in ch_rvs_idx]))

    return ProductNode(ch)


def mutual_info(x, y, rv_x, rv_y):
    c_xy, _, _ = np.histogram2d(x, y, bins=(rv_x, rv_y))
    res = mutual_info_score(None, None, contingency=c_xy)
    return res


