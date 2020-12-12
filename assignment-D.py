# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 09:07:57 2020

@author: Thomas Verduyn
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from sympy import Matrix


def system_solver(N, e):  # sets up system Au=f and solves it
    # Initial values
    h = 1 / N  # nr of lines

    # Constant values - Boundary conditions
    u0 = 1
    unp1 = 0

    # Discretisation
    A = scipy.sparse.diags([-e / h - 1, 2 * e / h + 1, -e / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    f = np.zeros(N - 1)
    f[0] = e / h + 1  # bring bc to rhs
    un = np.linalg.inv(A) @ f
    return np.concatenate(([u0], un, [unp1])), A, f

def backwardGS(N, eps, rtol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Constant values - Boundary conditions
    u0 = 1
    unp1 = 0

    # Discretization scheme and right-hand vector; CDS
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs
    res = 1  # initial residual
    un = np.zeros(N - 1)  # solution vector; iterated
    # Solver
    while res > rtol:
        for i, ai in zip(range(N - 2, -1, -1), A[::-1]):  # iterate backwards; ai: rows of A
            if i != 0 and i != N-2:
                un[i] = (f[i] - ai[i+1:] @ un[i+1:] - ai[:i] @ un[:i]) / ai[i]
            elif i == N-2:
                un[i] = (f[i] - ai[:i] @ un[:i]) / ai[i]
            else:
                un[i] = (f[i] - ai[(i + 1):] @ un[(i + 1):]) / ai[i]
            #print("iter:{} sol:{}".format(i, un))
        res = np.max(f - A @ un) / np.max(f)  # update residual;  using the infinity norm
        print(res)
    return un

def backwardGS1(N, eps, rtol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Constant values - Boundary conditions
    u0 = 1
    unp1 = 0

    # Discretization scheme and right-hand vector; CDS
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs
    res = 1  # initial residual
    u_old = np.zeros(N - 1)  # solution vector; iterated
    u_new = np.zeros(N - 1)
#     Solver
    while res > rtol:
        for i in range(N - 2, -1, -1):  # iterate backwards;
#            print(i,u_new)
            s1=0
            for j in range(i+1,N-1): # sum of first part
#                print('series1',j)
                s1+=A[i,j]*u_new[j]
            s2=0
            for j in range(i,-1,-1): # sum second part
#                print('series2',j)
                s2+=A[i,j]*u_old[j]
#            print(s1,s2)
            u_new[i]=(f[i]-s1-s2)/A[i,i]
        u_old=u_new
        res = np.max(f - A @ u_new) / np.max(f)  # update residual;  using the infinity norm
        print(res)
    return u_new

N = 5
eps = 0.5

sol = backwardGS1(N, eps)
print(sol)