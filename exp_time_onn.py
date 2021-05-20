#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from sklearn.cluster import KMeans
import pickle

from Optimization.RBCD_IBPde import RiemannianBlockCoordinateDescentIBP
from Optimization.RGA_IBP import RiemannianGradientAscentIBP
from Optimization.IterativeBregmanProjection import WassersteinBarycenterIBP


def main():

    d = 10
    m = 3
    k = 2
    num_points = [20, 50, 100, 250, 500, 1000]

    nb_exp = 1
    
    time_val = np.zeros([3, len(num_points), nb_exp])

    if 1:
        for exp in range(nb_exp):
            print("Experiments num: " , exp )

            for n_id, n in enumerate(num_points):
                print("n: " , n)
                center_supp_size = n

                stride = n * np.ones([1, m], dtype=np.int)

                supps = np.zeros([d, m*n])
                probs = np.zeros([1, m*n])
                mu = np.zeros(d)

                for i in range(m):
                    cov = np.random.randn(d, k)
                    cov = cov.dot(cov.T)

                    supps[:, i * n:(i + 1) * n] = np.random.multivariate_normal(mu, cov, size=n).T
                    prob = np.ones([1, n])
                    prob = prob / np.sum(prob)
                    probs[0, i * n: (i + 1) * n] = prob

                kmeans = KMeans(n_clusters=center_supp_size, random_state=0).fit(supps.T)
                center_supp = kmeans.cluster_centers_.T

                ones = np.ones((probs.shape[1], center_supp.shape[1]))
                C = np.diag(np.diag(supps.T.dot(supps))).dot(ones) + ones.dot(
                    np.diag(np.diag(center_supp.T.dot(center_supp)))) - 2 * supps.T.dot(center_supp)

                reg = 0.5*np.median(C)
                tau = 0.01

                U0 = np.zeros([d, k])
                U0[0:k, :] = np.identity(k)

                RBCD = RiemannianBlockCoordinateDescentIBP(reg, tau, max_iter=1000, gradThr=1e-2, ibpThr=1e-2, verbose=True)
                t3 = time.time()
                RBCD.run(stride, supps, probs, center_supp, U0)
                t4 = time.time()
                time_val[0, n_id, exp] = t4-t3

                RGA = RiemannianGradientAscentIBP(reg, tau, max_iter=1000,gradThr=1e-2, ibpThr=1e-2, verbose=True)
                t1 = time.time()
                RGA.run(stride, supps, probs, center_supp, U0)
                t2 = time.time()
                time_val[1, n_id, exp] = t2-t1

                t5 = time.time()
                q = WassersteinBarycenterIBP(stride, supps, probs, center_supp, reg, stopThr=1e-2, verbose=False)
                t6 = time.time()
                time_val[2, n_id, exp] = t6 - t5

                print('wb time',t6 - t5,  'rga time', t2 - t1, 'rbcd time', t4 - t3)

        with open('./results/time.pkl', 'wb') as f:
            pickle.dump([time_val], f)
    else:
        with open('./results/time.pkl', 'rb') as f:
            [time_val] = pickle.load(f)
            
    
    line = ['-', '-', '-' ]
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'purple']
    plt.figure(figsize=(12, 8))

    captions = ['RBCD', 'RGA-IBP', 'WB']
    for t in range(3):
        
        values_mean = np.mean(time_val[t, :, :], axis=1)
        # values_min = np.min(time_val[t, :, :], axis=1)
        # values_max = np.max(time_val[t, :, :], axis=1)

        mean, = plt.loglog(num_points , values_mean, ls=line[t],
                         c=colors[t], lw=4, ms=11,
                         label= captions[t])
        # col = mean.get_color()
        # plt.fill_between(num_points, values_min, values_max, facecolor=col, alpha=0.15)


    plt.xlabel('Number of samples n', fontsize=25)
    plt.ylabel('Time (seconds)', fontsize=25)
    plt.legend(loc='best', fontsize=18, ncol=2)
    # plt.title('Mean Estimation Error', fontsize=30)
    
    plt.xticks(num_points, fontsize=20)
    plt.yticks(fontsize=20)
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    plt.grid(ls=':')
    plt.savefig('./figs/exp4_time_n.png')


if __name__ == "__main__":
    main()

