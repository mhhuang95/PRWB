#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from numpy import matlib as mb

from .IterativeBregmanProjectionDense import WassersteinBarycenterIBP


class RiemannianGradientAscentIBP():
    
    def __init__(self, reg, tau, max_iter, gradThr=1e-4, ibpThr=1e-4, verbose=False):
        
        assert reg >= 0
        if tau is not None:
            assert tau > 0
        assert isinstance(max_iter, int)
        assert max_iter > 0
        assert gradThr > 0
        assert isinstance(verbose, bool)
        
        self.reg = reg
        self.tau = tau
        self.max_iter = max_iter
        self.grad_threshold = gradThr
        self.ibp_threshold = ibpThr
        self.verbose = verbose
        
    @staticmethod    
    def InitialStiefel(d, k):
        U = np.random.randn(d, k)
        q, r = np.linalg.qr(U)
        return q
    
    @staticmethod
    def StiefelRetraction(U, G):
        q, r = np.linalg.qr(U + G)
        return q
    
    @staticmethod
    def StiefelGradientProj(U, G):
        # project G onto the tangent space of Stiefel manifold at Z
        temp = G.T.dot(U)
        PG = G - U.dot(temp + temp.T) / 2
        return PG

    @staticmethod
    def geometricBar(weights, alldistribT):
        """return the weighted geometric mean of distributions"""
        assert (len(weights) == alldistribT.shape[1])
        return np.exp(np.dot(np.log(alldistribT), weights.T))
    
    def Vpi(self, X, Y,p, q, Pi, m):
        #Return the second order matrix of the displacements: sum_ij { (OT_plan)_ij (X_i-Y_j)(X_i-Y_j)^T }.
        A = X.dot(Pi.T).dot(Y.T)
        return X.dot(np.diag(p[:, 0])).dot(X.T) + m * Y.dot(np.diag(q[:, 0])).dot(Y.T) - A - A.T
    
    def run(self, stride, supps, probs, center_supp, U, fixed_supp=True, weights=None, warm_u=None, warm_v=None):

        m = stride.shape[1]
        if m == 1:
            print("Got one distribution, return itself!")
            if fixed_supp:
                return probs.reshape([-1, 1])
            else:
                return probs.reshape([-1, 1]), center_supp

        lp = probs.shape[1]
        d = supps.shape[0]
        tau = self.tau
        reg = self.reg

        assert lp == supps.shape[1]
        assert d == center_supp.shape[0]

        if weights is None:
            weights = np.ones(m) / m
        else:
            assert (len(weights) == m)

        p_l = probs.T
        f_val = np.inf
        
        for iter in range(self.max_iter):
            
            
            q, Pi = WassersteinBarycenterIBP(stride, U.T.dot(supps), probs, U.T.dot(center_supp), reg,stopThr=self.ibp_threshold, returnpi=True, verbose=False)
            
            V = self.Vpi(supps, center_supp,p_l, q, Pi, m)
            grad = 2 / m * V.dot(U)
            rgrad = self.StiefelGradientProj(U, grad)
            U = self.StiefelRetraction(U, tau * rgrad)           
            
            if np.linalg.norm(rgrad) < self.grad_threshold and not fixed_supp:
                q_old = q
                center_supp_old = center_supp
                
#                 center_supp = U.dot(U.T).dot(supps.dot(Pi.T)) /  mb.repmat(np.sum(Pi, 1), d, 1)
                center_supp = supps.dot(Pi.T) /  mb.repmat(np.sum(Pi, 1), d, 1)
    
                V = self.Vpi(supps, center_supp,p_l, q, Pi, m)
                grad = 2 / m * V.dot(U)
                rgrad = self.StiefelGradientProj(U, grad)

                f_val_old = f_val
                f_val = np.trace(1 / m * U.T.dot(V.dot(U)))
                if f_val >= f_val_old:
                    if self.verbose:
                        print('Terminate!', f_val_old, f_val)
                    q = q_old
                    center_supp = center_supp_old
                    break
                if self.verbose:
                    print( 'Center support updated!', f_val_old, f_val)
                    
            if np.linalg.norm(rgrad) < self.grad_threshold:
                break
            
        
        f_val = np.trace(1 / m * U.T.dot(V.dot(U)))
        if self.verbose:
            print("RGA done!", "Iter:", iter,  "Gradient norm:", np.linalg.norm(rgrad), 'fval:', f_val)

        self.f_val = f_val
        
        if fixed_supp:
            return q
        else:
            return q, center_supp
 