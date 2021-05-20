#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags
from numpy import matlib as mb


class RiemannianBlockCoordinateDescentIBP():
    
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
    
    @staticmethod
    def OTplan(m,supp_size, lp,posvec, u, v, K):
        Pi = np.zeros([supp_size, lp])
        for i in range(m):
            ui = u[i*supp_size:(i+1)*supp_size, :]
            vi = v[posvec[i]:posvec[i+1],:]
            Pi[:, posvec[i]:posvec[i+1]] = np.diag(np.matrix.flatten(ui)).dot(K[:, posvec[i]:posvec[i+1]]).dot(np.diag(np.matrix.flatten(vi)))
        return Pi
        
    
    def Vpi(self, X, Y, Pi):
        #Return the second order matrix of the displacements: sum_ij { (OT_plan)_ij (X_i-Y_j)(X_i-Y_j)^T }.
        A = X.dot(Pi.T).dot(Y.T)
        return X.dot(np.diag(np.sum(Pi, 0))).dot(X.T) + Y.dot(np.diag(np.sum(Pi, 1))).dot(Y.T) - A - A.T
    
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
        UUT = U.dot(U.T)
        M = np.diag(np.diag(supps.T.dot(UUT.dot(supps)))).dot(ones) + ones.dot(np.diag(np.diag(center_supp.T.dot(UUT.dot(center_supp))))) - 2 * supps.T.dot(UUT.dot(center_supp))
        M = M.T
        
        A = np.empty(M.shape, dtype=M.dtype)
        np.divide(M, -reg, out=A)
        np.exp(A, out=A)

        A[A<1e-200] = 1e-200

        spIDX_rows = np.zeros(supp_size * lp, dtype = np.int)
        spIDX_cols = np.zeros(supp_size * lp, dtype = np.int)
        for i in range(m):
            [xx, yy] = np.meshgrid( i * supp_size + np.arange(supp_size), np.arange(posvec[i],posvec[i+1]))
            ii = supp_size*posvec[i] + np.arange(supp_size*stride[0,i])
            spIDX_rows[ii] = np.matrix.flatten(xx)
            spIDX_cols[ii] = np.matrix.flatten(yy)
        spIDX = mb.repmat(np.identity(supp_size), 1, m);
        
        Ade = A
        A = csr_matrix((np.matrix.flatten(A.T), (spIDX_rows, spIDX_cols)), shape=(supp_size * m, lp), dtype=np.float)

        v = np.ones([lp,1])
        p_l = probs.T
        q = np.ones([supp_size,1]) / supp_size
        f_val = np.inf

        for iter in range(self.max_iter):
            q0 = mb.repmat(q, m, 1)
            u = q0 / A.dot(v)
            v = p_l / A.T.dot(u)
            q_l = np.reshape(u * A.dot(v), (supp_size, m), order='F')
            q = self.geometricBar(weights, q_l).reshape((supp_size,1))

#             Pi = spIDX * spdiags(np.matrix.flatten(u), 0, supp_size*m, supp_size*m).dot(A.dot(spdiags(np.matrix.flatten(v), 0, lp, lp)))
            Pi = self.OTplan(m,supp_size, lp,posvec, u, v, Ade)
            V = self.Vpi(supps, center_supp, Pi)
            grad = 2 / m * V.dot(U)

            rgrad = self.StiefelGradientProj(U, grad)
            U = self.StiefelRetraction(U, tau * rgrad) 
            
            q_bar = np.average(q_l, axis=1, weights=weights).reshape((supp_size,1))
            tol = np.sum(abs((q_l - mb.repmat(q_bar, 1, m))*weights))
            
            if np.linalg.norm(rgrad) < self.grad_threshold and tol < self.ibp_threshold and not fixed_supp:
                q_old = q
                center_supp_old = center_supp
                
#                 center_supp = U.dot(U.T).dot(supps.dot(Pi.T)) /  mb.repmat(np.sum(Pi, 1), d, 1)
                center_supp = supps.dot(Pi.T) /  mb.repmat(np.sum(Pi, 1), d, 1)
                
                UUT = U.dot(U.T)
                M = np.diag(np.diag(supps.T.dot(UUT.dot(supps)))).dot(ones) + ones.dot(np.diag(np.diag(center_supp.T.dot(UUT.dot(center_supp))))) - 2 * supps.T.dot(UUT.dot(center_supp))
                M = M.T

                A = np.empty(M.shape, dtype=M.dtype)
                np.divide(M, -reg, out=A)
                np.exp(A, out=A)

                A[A<1e-200] = 1e-200
                Ade = A
                A = csr_matrix((np.matrix.flatten(A.T), (spIDX_rows, spIDX_cols)), shape=(supp_size * m, lp), dtype=np.float)
                v = np.ones([lp,1])

                f_val_old = f_val
                f_val = np.sum(Pi * M)/m
                if f_val >= f_val_old:
                    if self.verbose:
                        print('Terminate!', f_val_old, f_val)
                    q = q_old
                    center_supp = center_supp_old
                    break
                tol = np.inf
                if self.verbose:
                    print( 'Center support updated!', f_val_old, f_val)
                
            if np.linalg.norm(rgrad) < self.grad_threshold and tol < self.ibp_threshold:
                break

#             f_val =  np.trace(1 / m * U.T.dot(V.dot(U)))
#             if self.verbose:
#                 print("Iter:", iter, "Tol:", tol, "Gradient norm:", np.linalg.norm(rgrad), 'fval:',  np.sum(Pi * M)/m )
            
            UUT = U.dot(U.T)
            M = np.diag(np.diag(supps.T.dot(UUT.dot(supps)))).dot(ones) + ones.dot(np.diag(np.diag(center_supp.T.dot(UUT.dot(center_supp))))) - 2 * supps.T.dot(UUT.dot(center_supp))
            M = M.T
                
            A = np.empty(M.shape, dtype=M.dtype)
            np.divide(M, -reg, out=A)
            np.exp(A, out=A)

            A[A<1e-200] = 1e-200
            Ade = A
            A = csr_matrix((np.matrix.flatten(A.T), (spIDX_rows, spIDX_cols)), shape=(supp_size * m, lp), dtype=np.float)

            if np.linalg.norm(rgrad) < self.grad_threshold and tol < self.ibp_threshold:
                break
        
        f_val = np.sum(Pi * M)/m
        if self.verbose:
            print("RBCD done! Iter:", iter,  "Gradient norm:", np.linalg.norm(rgrad), 'fval:', f_val )

        self.f_val = f_val
        
        if fixed_supp:
            return q
        else:
            return q, center_supp
 
 