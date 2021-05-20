#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans
from Optimization.RBCD_IBP import RiemannianBlockCoordinateDescentIBP
from Optimization.RGA_IBP import RiemannianGradientAscentIBP
from Optimization.IterativeBregmanProjection import WassersteinBarycenterIBP


def main():
    
    d = 100
    m = 3
    n = 10
    k_star = 2
    k = 6
    stride = n * np.ones([1, m], dtype=np.int)
    center_supp_size = n
    
    nb_exp = 1
    noise_levels = [0., 0.01, 0.1, 1, 2, 4, 7, 10]
    obj_val = np.zeros([3, len(noise_levels), nb_exp])

    if 1:
        for exp in range(nb_exp):
            print("Experiments num: " , exp )

            supps = np.zeros([d, m*n])
            probs = np.zeros([1, m*n])
            mu = np.zeros(d)

            for i in range(m):
                cov = np.random.randn(d, k_star)
                cov = cov.dot(cov.T)

                supps[:, i*n:(i+1)*n] = np.random.multivariate_normal(mu, cov, size=n).T
                prob = np.ones([1, n])
                prob = prob / np.sum(prob)
                probs[0, i*n : (i+1)*n] = prob

            print(supps.shape)
            kmeans = KMeans(n_clusters=center_supp_size, random_state=0).fit(supps.T)
            center_supp = kmeans.cluster_centers_.T

            ones = np.ones((probs.shape[1], center_supp.shape[1]))
            C = np.diag(np.diag(supps.T.dot(supps))).dot(ones) + ones.dot(
                np.diag(np.diag(center_supp.T.dot(center_supp)))) - 2 * supps.T.dot(center_supp)
            C = C.T
            reg = 0.5 * np.median(C)

            for nlidx in range(len(noise_levels)):

                nlevel = noise_levels[nlidx]
                noiseX = np.random.randn(d, m*n)
                supps_noisy = supps + nlevel * noiseX

                q, Pi = WassersteinBarycenterIBP(stride, supps_noisy, probs, center_supp, reg,stopThr=1e-4, returnpi=True, verbose=True)
                C_noisy = np.diag(np.diag(supps_noisy.T.dot(supps_noisy))).dot(ones) + ones.dot(
                    np.diag(np.diag(center_supp.T.dot(center_supp)))) - 2 * supps_noisy.T.dot(center_supp)
                C_noisy = C_noisy.T
                obj_val[2, nlidx, exp] = np.sum(Pi *  C_noisy) / m

                tau = 0.001
                if nlevel >= 7:
                    tau = 0.0005

                RBCD = RiemannianBlockCoordinateDescentIBP(reg, tau, max_iter=1000, gradThr=1e-2, ibpThr=1e-4, verbose=True)
                RGA = RiemannianGradientAscentIBP(reg, tau, max_iter=1000, gradThr=1e-2, ibpThr=1e-4, verbose=True)

                U0 = np.zeros([d, k])
                U0[0:k, :] = np.identity(k)

                RBCD.run(stride, supps_noisy, probs, center_supp, U0)
                obj_val[0, nlidx,  exp] = RBCD.f_val

                RGA.run(stride, supps_noisy, probs, center_supp, U0)
                obj_val[1, nlidx, exp] = RGA.f_val

        with open('./results/noiselevel.pkl', 'wb') as f:
            pickle.dump([obj_val], f)
    else:
        with open('./results/noiselevel.pkl', 'rb') as f:
            [obj_val] = pickle.load(f)

    line = ['-', '--', '-']
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'purple']
    plt.figure(figsize=(12, 8))

    captions = ['RBCD', 'RGA-IBP', 'WB']
    for t in range(3):
        values_mean = np.mean(obj_val[t, :, :], axis=1)
        # values_min = np.min(obj_val[t, :, :], axis=1)
        # values_max = np.max(obj_val[t, :, :], axis=1)

        error_mean = np.abs(values_mean - values_mean[0]) / values_mean[0]
        # error_min = np.abs(values_min - values_min[0]) / values_min[0]
        # error_max = np.abs(values_max - values_max[0]) / values_max[0]

        mean, = plt.loglog(noise_levels[1:], error_mean[1:], ls=line[t],
                         c=colors[t], lw=4, ms=20,
                         label=captions[t])
        # col = mean.get_color()
        # plt.fill_between(noise_levels[1:] , error_min[1:], error_max[1:], facecolor=col, alpha=0.15)


    plt.xlabel('Noise Level $\sigma$', fontsize=25)
    plt.ylabel('Relative Error', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(ls=':')
    plt.legend(loc='best', fontsize=18, ncol=2)
    plt.savefig('./figs/exp3_noiselevel.png')


if __name__ == "__main__":
    main()

