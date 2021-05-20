#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import ot
import gc
import multiprocessing
import time
from sklearn.metrics.cluster import adjusted_mutual_info_score

from Optimization.RBCD_IBP import RiemannianBlockCoordinateDescentIBP


def parallel_PWB(para):
    res = {}
    if para['fixed_supp']:
        res['center_prob'] = para['pwb_solver'].run(para['cluster_stride'], para['cluster_supps'], para['cluster_probs'], para['center_supp'], para['U'], fixed_supp=para['fixed_supp'])
    else:
        res['center_prob'], res['center_supp'] = para['pwb_solver'].run(para['cluster_stride'], para['cluster_supps'], para['cluster_probs'], para['center_supp'], para['U'], fixed_supp=para['fixed_supp'])
    return res

def parallel_Distance(para):

    dis_to_center = []
    for sample_id in range(para['m']):

        #Todo: Compute wasserstein distance
        sample_prob = para['probs'][0, para['posvec'][sample_id]:para['posvec'][sample_id+1]]
        X = para['supps'][:, para['posvec'][sample_id]:para['posvec'][sample_id+1]]
        center_prob = para['centers_probs'][0, para['center_id']*para['n']:(para['center_id']+1)*para['n']]
        Y = para['centers_supps'][:, para['center_id']*para['n']:(para['center_id']+1)*para['n']]
        ones = np.ones([para['stride'][0, sample_id],para['n']])
        cost_mat = np.diag(np.diag(X.T.dot(X))).dot(ones) + ones.dot(np.diag(np.diag(Y.T.dot(Y)))) - 2 * X.T.dot(Y)

        pi = ot.sinkhorn(sample_prob, center_prob, cost_mat, para['otreg'], stopThr=1e-3)
        dis_to_center.append(np.sum(pi*cost_mat))
    return dis_to_center


def solve_dis_opt(a, X):
    n = X.shape[1]
    minval = a[0]*a[1]*np.linalg.norm(X[:, 0] - X[:, 1])**2 /(a[0]+a[1])
    res_i, res_j = 0, 1
    for i in range(n-1):
        for j in range(i+1, n):
            val = a[i]*a[j]*np.linalg.norm(X[:, i] - X[:, j])**2 /(a[i]+a[j])
            if minval > val:
                minval = val
                res_i, res_j = i,j
    return res_i, res_j
            

def merge(X, a, target_n):
    
    assert target_n >= 2
    
    d = X.shape[0]
    n = X.shape[1]
    if n <= target_n:
        return X, a
    else:
        for k in range(n - target_n):
            i,j = solve_dis_opt(a, X)
            new_a_ele, new_x_ele = a[i] + a[j],  (a[i] * X[:,i] + a[j] * X[:,j] )/(a[i] + a[j])
            a = np.concatenate((a[:i], a[i+1:j], a[j+1:], np.array([new_a_ele])))
            X = np.concatenate((X[:,:i], X[:, i+1:j], X[:, j+1:], np.array(new_x_ele).reshape([d, 1])), axis=1)
        return X, a

class ProjectionD2Clustering():
    
    """
    Projection discrete distribution clustering.
    
    Parameters
    ----------
    """
    
    def __init__(self, n_clusters=10, tau=0.005, fixed_supp=True, max_iter=100, verbose=False):
        
        self.n_clusters = n_clusters
        self.fixed_supp = fixed_supp
        self.max_iter = max_iter
        self.verbose = verbose
        self.tau = tau
        
    def getwarmlabels(self, labels, stride):
        length = np.sum(stride)
        warmlabels = np.zeros(length)
        pos = 0
        for i in range(stride.shape[1]):
            warmlabels[pos:pos + stride[0, i]] = labels[i]
            pos = pos + stride[0, i]
        return warmlabels

    def findinitcenters(self, stride, probs, supps, n, posvec, center_num, start_point):

        d = supps.shape[0]
        centers_stride = n * np.ones([1, center_num], dtype=int)
        centers_supps = np.zeros([d, n * center_num])
        centers_probs = np.zeros([1, n * center_num])

        center_id = 0
        sample_id = start_point
        while center_id < center_num:
            if stride[0, sample_id] >= n:
                centers_stride[0, center_id] = n
                X = supps[:, posvec[sample_id]:posvec[sample_id + 1]]
                a = probs[0, posvec[sample_id]:posvec[sample_id + 1]]
                #                 X, a = merge(X, a, n)
                centers_supps[:, center_id * n: (center_id + 1) * n] = X
                centers_probs[0, center_id * n: (center_id + 1) * n] = a
                center_id += 1

            sample_id = (sample_id + 1) % stride.shape[1]

        return centers_stride, centers_supps, centers_probs
        

    
            
    def parallel_fit(self, stride, probs, supps, n,k, label_true,init_point=0, nproc=30, eta=0.5,otreg=0.5, fixed_supp=True):
        
        tau = self.tau
        d = supps.shape[0]
        m = stride.shape[1]
        assert np.sum(stride) == supps.shape[1]
        assert np.sum(stride) == probs.shape[1]
        posvec = np.concatenate((np.array([0]),np.cumsum(stride)))
               
        if m < self.n_clusters:
            print("The number of clusters surpasses the number of samples")
            return
               
        #Initialization
        print(self.n_clusters)
        labels = np.arange(m) % self.n_clusters
        print('Initial labels:', labels)
        centers_stride, centers_supps, centers_probs = self.findinitcenters(stride, probs, supps,n, posvec, self.n_clusters, init_point)
        RBCDIBP = RiemannianBlockCoordinateDescentIBP(eta, tau, max_iter=4000, gradThr=1e-2, ibpThr=1e-2, verbose=True)

        U = np.zeros([d,k])
        U[:k, :] = np.identity(k)

        p = multiprocessing.Pool(min(nproc, self.n_clusters))
        amis = []
        for iter in range(self.max_iter):
            print('Iter:', iter)
            
            labels_old = labels
            t3 = time.time()
            paras = []
            for center_id in range(self.n_clusters):
                para = {}
                para['m'] = m
                para['k'] = k
                para['n'] = n
                para['stride'] = stride
                para['center_id'] = center_id
                para['probs'] = probs
                para['supps'] = supps
                para['centers_probs'] = centers_probs
                para['centers_supps'] = centers_supps
                para['posvec'] = posvec
                para['otreg'] = otreg
                paras.append(para)
                
            res_distances = p.map(parallel_Distance, paras)
            W_distances = np.array(res_distances)
            
            t4 = time.time()
            print('sinkhorn time', t4-t3)

            labels = np.argmin(W_distances, axis=0)
            amis.append(adjusted_mutual_info_score(label_true, labels, average_method='arithmetic'))
            print('Label changing rate:', np.sum(labels != labels_old), 'AMI=', amis[-1])
            
            if (labels_old == labels).all():
                print("Get stable clusters, terminate!")
                break
            
            #Compute Wasserstein barycenters for each cluster
            warmlabels = self.getwarmlabels(labels, stride)
            paras = []
            t1 = time.time()
            for label_id in range(self.n_clusters):
                para = {}
                para['pwb_solver'] = RBCDIBP
                para['U'] = U
                para['n'] = n
                para['reg'] = eta
                para['fixed_supp'] = fixed_supp
                para['center_supp'] = centers_supps[:, label_id*n:(label_id+1)*n]
                para['cluster_stride'] = stride[:, label_id == labels]
                para['cluster_supps'] = supps[:, label_id == warmlabels]
                para['cluster_probs'] = probs[:, label_id == warmlabels]
                print('label_id:',label_id, para['cluster_supps'].shape, para['cluster_stride'].shape[1])
                
                paras.append(para)
            
            res_centers = p.map(parallel_PWB, paras)
   
            for label_id in range(self.n_clusters):
                if res_centers[label_id]['center_prob'].shape[0] < n:
                    _, centers_supps[:, label_id * n:(label_id + 1) * n], centers_probs[:, label_id * n:(label_id + 1) * n] = self.findinitcenters(stride, probs, supps, n, posvec, 1, np.random.randint(m))
                else:
                    centers_probs[:, label_id * n:(label_id + 1) * n] = res_centers[label_id]['center_prob'].reshape([1, n])
                    if not fixed_supp:
                        centers_supps[:, label_id * n:(label_id + 1) * n] = res_centers[label_id]['center_supp']

            t2 = time.time()
            print('center time', t2-t1)

        return np.array(amis)
