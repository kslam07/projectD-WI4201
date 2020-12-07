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

N = 5
eps = 0.5

sol = backwardGS(N, eps)
print(sol)