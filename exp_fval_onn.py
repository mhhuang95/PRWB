#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import pickle

from Optimization.RBCD_IBPde import RiemannianBlockCoordinateDescentIBP
from Optimization.RGA_IBP import RiemannianGradientAscentIBP
from Optimization.IterativeBregmanProjection import WassersteinBarycenterIBP

def Gausscenters(covs, w):
    d = covs.shape[0]
    m = w.shape[1]
    S = np.identity(d)
    max_iter = 300
    for iter in range(max_iter):
        S_old = S
        sqS = sqrtm(S_old)
   
        S = np.zeros(d)
        for k in range(m):
            S = S + w[0, k] * sp.linalg.sqrtm(sqS.dot(covs[:, :, k]) .dot(sqS))
        rela_err = np.linalg.norm(S - S_old)

    V = 0
    sqS = sp.linalg.sqrtm(S)
    for k in range(m):
        V = V + w[0, k]* (np.trace (covs[:, :, k]) - np.trace(2 * sp.linalg.sqrtm(sqS.dot(covs[:, :, k]).dot(sqS))))
    V = V + np.trace(S)
    V = np.real(V)

    print("Relative error:", rela_err, "Barycener value for Gaussian:", V)
    
    return V

    

def main():
    
    d = 10
    m = 2
    k = 2
    num_points = [20, 50, 100, 250, 500, 1000]
    
    thre = 0.1
    covs = np.zeros([d, d, m])
    covs[0, 0, 0] = 10
    covs[1, 1, 1] = 10
    covs[:, :, 0] = covs[:, :, 0] + thre * np.identity(d)
    covs[:, :, 1] = covs[:, :, 1] + thre * np.identity(d)
    
    w = 1/m * np.ones([1, m])
    
    ground_truth = Gausscenters(covs, w)
    
    nb_exp = 500
    
    obj_val = np.zeros([3, len(num_points), nb_exp])

    if 1:
        for exp in range(nb_exp):
            print("Experiments num: " , exp )

            for n_id, n in enumerate(num_points):
                print("n: " , n)

                stride = n * np.ones([1, m], dtype=np.int)

                supps = np.zeros([d, m*n])
                probs = np.zeros([1, m*n])
                mu = np.zeros(d)

                for i in range(m):
                    cov = covs[:,:,i]
                    X = np.random.multivariate_normal(mu, cov, size=n)
                    X = X.T

                    supps[:, i*n:(i+1)*n] = X

                    xsx = np.diag(X.T.dot(np.linalg.inv(cov).dot(X)))

                    prob = 1/((2*np.pi)**(d/2) * np.linalg.det(cov)) * np.exp(-0.5 * xsx)

                    prob = prob / np.sum(prob)
                    np.reshape(prob, [1, n])
                    probs[:, i*n : (i+1)*n] = prob

                center_supp = 4 * np.random.rand(d, n) - 2

                ones = np.ones((probs.shape[1], center_supp.shape[1]))
                C = np.diag(np.diag(supps.T.dot(supps))).dot(ones) + ones.dot(
                    np.diag(np.diag(center_supp.T.dot(center_supp)))) - 2 * supps.T.dot(center_supp)

                reg = 0.2*np.median(C)
                tau = 0.05

                U0 = np.zeros([d, k])
                U0[0:k, :] = np.identity(k)

                RBCD = RiemannianBlockCoordinateDescentIBP(reg, tau, max_iter=1000, gradThr=1e-4, ibpThr=1e-4, verbose=True)
                t3 = time.time()
                RBCD.run(stride, supps, probs, center_supp, U0)
                t4 = time.time()
                obj_val[0, n_id, exp] = abs(RBCD.f_val - ground_truth)

                RGA = RiemannianGradientAscentIBP(reg, tau, max_iter=1000,gradThr=1e-4, ibpThr=1e-4, verbose=True)
                t1 = time.time()
                RGA.run(stride, supps, probs, center_supp, U0)
                t2 = time.time()
                obj_val[1, n_id, exp] = abs(RGA.f_val -ground_truth)

                print('rga time', t2-t1, 'rbcd time', t4-t3)

                q, Pi = WassersteinBarycenterIBP(stride, supps, probs, center_supp, reg, returnpi=True, verbose=False)
                C = C.T
                obj_val[2, n_id, exp] = abs(np.sum(Pi * C)/m - ground_truth)

        with open('./results/fval_onn.pkl', 'wb') as f:
            pickle.dump([obj_val], f)
    else:
        with open('./results/fval_onn.pkl', 'rb') as f:
            [obj_val] = pickle.load(f)

    
    line = ['-', '--', '-' ]
    colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'purple']
    plt.figure(figsize=(12, 8))

    captions = ['RBCD', 'RGA-IBP', 'WB']
    for t in range(3):
        
        values_mean = np.mean(obj_val[t, :, :], axis=1)
        # values_min = np.min(obj_val[t, :, :], axis=1)
        # values_max = np.max(obj_val[t, :, :], axis=1)

        mean, = plt.loglog( num_points , values_mean, ls=line[t],
                         c=colors[t], lw=4, ms=11,
                         label= captions[t])
        # col = mean.get_color()
        # plt.fill_between(num_points, values_min, values_max, facecolor=col, alpha=0.15)


    plt.xlabel('Number of samples n', fontsize=25)
    plt.ylabel('MEE', fontsize=25)
    plt.legend(loc='best', fontsize=18, ncol=2)
    # plt.title('Mean Estimation Error', fontsize=30)
    
    plt.xticks( num_points, fontsize=20)
    plt.yticks(np.array([0.01, 0.1, 1, 10, 20]), fontsize=20)
    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    plt.grid(ls=':')
    plt.savefig('./figs/exp2_samples_n.png')


if __name__ == "__main__":
    main()

