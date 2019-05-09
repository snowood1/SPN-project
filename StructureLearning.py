from SumProductNets import *
import numpy as np


k_mean_k = 3
k_mean_its = 20


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
    pass



