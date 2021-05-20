#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from numpy import matlib as mb
import time

def geometricBar(weights, alldistribT):
    """return the weighted geometric mean of distributions"""
    assert (len(weights) == alldistribT.shape[1])
    return np.exp(np.dot(np.log(alldistribT), weights.T))


def WassersteinBarycenterIBP(stride, supps, probs, center_supp, reg, weights=None, fixed_supp=True, numItermax=500, stopThr=1e-4, returnpi=False, verbose=False):
    
    m = stride.shape[1]

    if m == 1:
        if verbose:
            print("Got one distribution, return itself!")
        if fixed_supp:
            return probs.reshape([-1, 1])
        else:
            return probs.reshape([-1, 1]), center_supp

    lp = probs.shape[1]
    d = supps.shape[0]
    posvec = np.concatenate((np.array([0]),np.cumsum(stride)))
    supp_size = center_supp.shape[1]
    
    assert lp == supps.shape[1]
    assert d == center_supp.shape[0]
    
    if weights is None:
        weights = np.ones(m) / m
    else:
        assert (len(weights) == m)
        
    #Cost matrix
    ones = np.ones((lp, supp_size))
    C = np.diag(np.diag(supps.T.dot(supps))).dot(ones) + ones.dot(np.diag(np.diag(center_supp.T.dot(center_supp)))) - 2 * supps.T.dot(center_supp)
    C = C.T
    
    K = np.empty(C.shape, dtype=C.dtype)
    np.divide(C, -reg, out=K)
    np.exp(K, out=K)
    
    K[K<1e-200] = 1e-200

    spIDX_rows = np.zeros(supp_size * lp, dtype = np.int)
    spIDX_cols = np.zeros(supp_size * lp, dtype = np.int)
    for i in range(m):
        [xx, yy] = np.meshgrid( i * supp_size + np.arange(supp_size), np.arange(posvec[i],posvec[i+1]))
        ii = supp_size*posvec[i] + np.arange(supp_size*stride[0,i])
        spIDX_rows[ii] = np.matrix.flatten(xx)
        spIDX_cols[ii] = np.matrix.flatten(yy)
    spIDX = mb.repmat(np.identity(supp_size), 1, m)

    K = csr_matrix((np.matrix.flatten(K.T), (spIDX_rows, spIDX_cols)), shape=(supp_size * m, lp), dtype=np.float)
    
    v = np.ones([lp,1])
    p_l = probs.T
    q = np.ones([supp_size,1]) / supp_size
    
    f_val = np.inf
    
    for iter in range(numItermax):
        q0 = mb.repmat(q, m, 1)
        u = q0 / K.dot(v)
        v = p_l / K.T.dot(u)
        
        q_l = np.reshape(u * K.dot(v), (supp_size, m), order='F')
        q = geometricBar(weights, q_l).reshape((supp_size,1))
        
        Pi = spIDX * spdiags(np.matrix.flatten(u), 0, supp_size*m, supp_size*m).dot(K.dot(spdiags(np.matrix.flatten(v), 0, lp, lp)))
        # Pi = OTplan(m,supp_size, lp,posvec, u, v, Kde)
        
        q_bar = np.average(q_l, axis=1, weights=weights).reshape((supp_size,1))
        tol = np.sum(abs((q_l - mb.repmat(q_bar, 1, m))*weights))

        if tol < stopThr and not fixed_supp:
            q_old = q
            Pi_old = Pi
            center_supp_old = center_supp
            
            center_supp = supps.dot(Pi.T) /  mb.repmat(np.sum(Pi, 1), d, 1)
            C = np.diag(np.diag(supps.T.dot(supps))).dot(ones) + ones.dot(np.diag(np.diag(center_supp.T.dot(center_supp)))) - 2 * supps.T.dot(center_supp)
            C = C.T

            K = np.empty(C.shape, dtype=C.dtype)
            np.divide(C, -reg, out=K)
            np.exp(K, out=K)

            K[K<1e-200] = 1e-200
            K = csr_matrix((np.matrix.flatten(K.T), (spIDX_rows, spIDX_cols)), shape=(supp_size * m, lp), dtype=np.float)
            v = np.ones([lp,1])
            
            f_val_old = f_val
            f_val = np.sum(Pi * C)/m
            if f_val > f_val_old:
                if verbose:
                    print('Terminate!', f_val_old, f_val)
                q = q_old
                Pi = Pi_old
                center_supp = center_supp_old
                break
            tol = np.inf
            if verbose:
                print( 'Center support updated!', f_val_old, f_val)

        if tol < stopThr:
            if verbose:
                print('Total iteration number: ', iter)
            break

    if fixed_supp:
        if returnpi:
            return q, Pi
        else:
            return q
    else:
        if returnpi:
            return q, center_supp, Pi
        else:
            return q, center_supp
            
