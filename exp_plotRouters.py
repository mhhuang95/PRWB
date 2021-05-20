#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle
import matplotlib.pyplot as plt

from d2_clustering import D2Clustering
from Projection_d2_clustering import ProjectionD2Clustering

def solve_dis_opt(a, X):
    n = X.shape[1]
    minval = a[0] * a[1] * np.linalg.norm(X[:, 0] - X[:, 1]) ** 2 / (a[0] + a[1])
    res_i, res_j = 0, 1
    for i in range(n - 1):
        for j in range(i + 1, n):
            val = a[i] * a[j] * np.linalg.norm(X[:, i] - X[:, j]) ** 2 / (a[i] + a[j])
            if minval > val:
                minval = val
                res_i, res_j = i, j
    return res_i, res_j


def merge(X, a, target_n):
    assert target_n >= 2

    d = X.shape[0]
    n = X.shape[1]
    if n <= target_n:
        return X, a, False
    else:
        for k in range(n - target_n):
            i, j = solve_dis_opt(a, X)
            new_a_ele, new_x_ele = a[i] + a[j], (a[i] * X[:, i] + a[j] * X[:, j]) / (a[i] + a[j])
            a = np.concatenate((a[:i], a[i + 1:j], a[j + 1:], np.array([new_a_ele])))
            X = np.concatenate((X[:, :i], X[:, i + 1:j], X[:, j + 1:], np.array(new_x_ele).reshape([d, 1])), axis=1)
        return X, a, True


def loaddata(cluster, cluster_vocab, d, n):
    stride = np.empty([1, 0], dtype=np.int)
    probs = np.empty([1, 0])
    supps = np.empty([d, 0])

    dic = dict()
    with open(cluster_vocab, 'r') as f:
        s = f.readline()
        idx = 1
        while True:
            s = f.readline()
            if not s:
                break
            dic[idx] = np.array(s.strip('\n').strip(' ').split(' '), dtype='f8')
            idx += 1

    idx = 0
    with open(cluster, 'r') as f:
        while True:
            dim = f.readline()
            if not dim:
                break
            s = int(f.readline())
            # print(s)
            stride = np.concatenate((stride, np.array([[min(s, n)]], dtype=np.int)), axis=1)
            a = np.array(f.readline().strip('\n').strip(' ').split(' '), dtype='f8')
            a = a / np.sum(a)
            Xidx = np.array(f.readline().strip('\n').strip(' ').split(' '), dtype=np.int)
            X = np.empty([d, int(s)])
            for i in range(int(s)):
                X[:, i] = dic[Xidx[i]]
            X, a, flag = merge(X, a, n)

            probs = np.concatenate((probs, a.reshape([1, -1])), axis=1)
            supps = np.concatenate((supps, X), axis=1)
            idx += 1
            print(idx, s, a.shape)
            # if idx >= 2:
            #     break
    print('Data loaded!')

    return stride, probs, supps, idx


def run_clustering():
    d = 300
    n = 16
    maxiter = 10

    if 1:
        cluster = 'reuters_cluster.d2s'
        cluster_vocab = 'reuters_cluster.d2s.vocab0'
        stride, probs, supps, idx = loaddata(cluster, cluster_vocab, d, n)

        with open('./reuters_data32.pkl', 'wb') as f:
            pickle.dump([stride, probs, supps], f)

    else:
        with open('./reuters_data32.pkl', 'rb') as f:
            [stride, probs, supps] = pickle.load(f)


    nb_exp = 5
    AMIs = np.zeros([2, maxiter, nb_exp])

    label_true = []
    with open("reuters_cluster.d2s.labels", 'r') as f:
        while True:
            la = f.readline()
            if not la:
                break
            label_true.append(int(la))

    initpoints = [600, 700, 800, 900, 1000]

    if 1:
        for exp in range(nb_exp):
            print("Experiments num: ", exp)
            init_point = initpoints[exp]

            d2Clustering = D2Clustering(n_clusters=5, max_iter=maxiter)
            AMIs[0, :, exp] = d2Clustering.parallel_fit(stride, probs, supps, n, label_true=label_true,
                                                        init_point=init_point, eta=1, otreg=0.5, fixed_supp=False)

            k = 3
            projd2Clustering = ProjectionD2Clustering(n_clusters=5, max_iter=maxiter, tau=0.05)
            AMIs[1, :, exp] = projd2Clustering.parallel_fit(stride, probs, supps, n, k, label_true=label_true,
                                                            init_point=init_point, eta=1, otreg=0.5, fixed_supp=False)

        with open('./reuters_ami.pkl', 'wb') as f:
            pickle.dump([AMIs], f)

    else:
        with open('./reuters_ami.pkl', 'rb') as f:
            [AMIs] = pickle.load(f)

    print(AMIs)

    line = ['-', '-']
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'purple']
    plt.figure(figsize=(12, 8))

    captions = ['D2 clustering', 'PD2 clustering']

    for t in range(2):
        AMI_mean = np.mean(AMIs[t, :, :], axis=1)
        plt.plot(np.arange(maxiter), AMI_mean, ls=line[t], c=colors[t], lw=4, ms=20, label=captions[t])

    plt.xlabel('Iteration', fontsize=25)
    plt.ylabel('AMI', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc='best', fontsize=18, ncol=2)
    plt.savefig('figs/exp5_reuters.png')

if __name__ == "__main__":
    run_clustering()
