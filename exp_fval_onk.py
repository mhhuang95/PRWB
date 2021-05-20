#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans
from Optimization.RBCD_IBP import RiemannianBlockCoordinateDescentIBP
from Optimization.RGA_IBP import RiemannianGradientAscentIBP


def main():
    
    d = 100
    m = 3
    n = 10
    center_supp_size = n
    stride = n * np.ones([1, m], dtype=np.int)
    k_stars = [2, 3, 4, 5]
    
    max_k = 20
    nb_exp = 100
    
    obj_val = np.zeros([2, len(k_stars), max_k, nb_exp])

    if 1:
        for exp in range(nb_exp):
            print("Experiments num: " , exp )

            for k_star_id, k_star in enumerate(k_stars):
                print("k_star: " , k_star )

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

                reg = 0.5*np.median(C)
                tau = 0.001

                if k_star >= 4:
                    tau = 0.002

                RBCD = RiemannianBlockCoordinateDescentIBP(reg, tau, max_iter=1000, gradThr=1e-2, ibpThr=1e-4, verbose=True)
                RGA = RiemannianGradientAscentIBP(reg, tau, max_iter=1000,gradThr=1e-2, ibpThr=1e-4, verbose=True)

                for k in range(1, max_k+1):
                    print("k: " , k )
                    U0 = np.zeros([d, k])
                    U0[0:k, :] = np.identity(k)

                    RBCD.run(stride, supps, probs, center_supp, U0)
                    obj_val[0, k_star_id, k-1, exp] = RBCD.f_val

                    RGA.run(stride, supps, probs, center_supp, U0)
                    obj_val[1, k_star_id, k-1, exp] = RGA.f_val

        with open('./results/fval_onk.pkl', 'wb') as f:
            pickle.dump([obj_val], f)

    else:
        with open('./results/fval_onk.pkl', 'rb') as f:
            [obj_val] = pickle.load(f)
    
    colors = [['b', 'orange', 'g', 'r'], ['c', 'm', 'y', 'purple']]
    plt.figure(figsize=(20, 12))

    Xs = list(range(1, max_k + 1))
    line_styles = ['-', '--']
    captions = ['RBCD', 'RGA-IBP']
    for t in range(2):
        for i, k_star in enumerate(k_stars):
            values_mean = np.mean(obj_val[t, i, :, :], axis=1)
            # values_min = np.min(obj_val[t, i, :, :], axis=1)
            # values_max = np.max(obj_val[t, i, :, :], axis=1)

            mean, = plt.plot(Xs, values_mean, ls=line_styles[t],
                             c=colors[t][i], lw=4, ms=20,
                             label='$k^*=%d$, %s' % (k_star,captions[t]))
            # col = mean.get_color()
            # plt.fill_between(Xs, values_min, values_max, facecolor=col, alpha=0.15)

    for i in range(len(k_stars)):
        ks = k_stars[i]*3
        vm1 = np.mean(obj_val[0, i, ks, :], axis=0)
        vm2 = np.mean(obj_val[1, i, ks, :], axis=0)
        print(vm1,vm2)
        tt = max(vm1,vm2)
        plt.plot([ks, ks], [0, tt], color=colors[0][i], linestyle='--')


    plt.xlabel('Dimension k', fontsize=25)
    plt.ylabel('PRWB values', fontsize=25)
    plt.xticks(Xs, fontsize=20)
    plt.yticks(np.arange(100, 900, 100), fontsize=20)
    plt.legend(loc='best', fontsize=18, ncol=2)
    plt.ylim(0)
    # plt.title('PWB objective values depending on dimension k', fontsize=30)
    plt.minorticks_on()
    plt.grid(ls=':')
    plt.savefig('./figs/exp1_dim_k.png')


if __name__ == "__main__":
    main()
