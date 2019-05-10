from SumProductNets import *
import numpy as np
from itertools import product


k_mean_k = 2
k_mean_its = 20

dependent_threshold = 0.1


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
        if ch_m == 0:
            continue
        ch.append(split_variables(ch_data, rvs))
        w.append(ch_m / m)

    print('sum')
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
                    # mi = mutual_info(data[x_idx], data[y_idx], rvs[x_idx], rvs[y_idx])
                    mi = np.random.rand()
                    if mi > dependent_threshold:
                        remaining_rvs_idx -= {y_idx}
                        ch_rvs_idx.append(y_idx)

            ch_data = data[ch_rvs_idx, :]
            _, m = ch_data.shape
            if len(ch_rvs_idx) == 1 or (ch_data == ch_data[0]).all():
                for i in ch_rvs_idx:
                    print('rv_node')
                    ch.append(RVNode(rvs[i], np.array([sum(data[i] == j) / m for j in rvs[i].domain])))
            else:
                ch.append(split_instances(ch_data, [rvs[i] for i in ch_rvs_idx]))

    print('product')
    return ProductNode(ch)


def mutual_info(x, y, rv_x, rv_y):
    m = len(x)
    a, a_p = dict(), dict()
    b, b_p = dict(), dict()

    for s in rv_x.domain:
        a[s] = x == s
        a_p[s] = sum(a[s]) / m
    for s in rv_y.domain:
        b[s] = y == s
        b_p[s] = sum(b[s]) / m

    res = 0
    for s in product(rv_x.domain, rv_y.domain):
        ab_p = sum(a[s[0]] * b[s[1]]) / m
        if ab_p == 0:
            continue
        res += ab_p * np.log(ab_p / (a_p[s[0]] * b_p[s[1]]))

    return res

