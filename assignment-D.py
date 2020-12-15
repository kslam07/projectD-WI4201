# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 09:07:57 2020

@author: Thomas Verduyn
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import copy
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
        res = np.max(f - A @ un) / np.max(f)  # update residual;  using the infinity norm
    return un

def jacobi(N, eps, rtol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Constant values - Boundary conditions
    u0 = 1
    unp1 = 0

    # Discretization scheme and right-hand vector; CDS
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs
    u_current = np.zeros(N - 1)
    res_scaled = 1
    u_new = np.zeros(N - 1)
    res_lst = []
    while res_scaled > rtol:
        for i, row in enumerate(A):
            u_new[i] = (f[i] - (row @ u_current - A[i, i] * u_current[i])) / A[i, i]
            # print(row,u_current)
        u_current = copy.deepcopy(u_new)
        res = f - A @ u_new  # res_k+1 = f - A*u_k
        res_scaled = np.sum(np.sqrt(res ** 2)) / np.sum(np.sqrt(f ** 2))
        res_lst.append(np.sum(np.sqrt(res ** 2)))

    return u_current, res_lst

def forwardGS(N, eps, rtol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Discretization scheme and right-hand vector; CDS
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs
    u_current = np.zeros(N - 1)
    res_scaled = 1  # initial residual
    res_lst = []
    while res_scaled > rtol:
        for i, row in enumerate(A):
            u_current[i] = (f[i] - row[:i] @ u_current[:i] - row[i + 1:] @ u_current[i + 1:]) / A[i, i]
        res = np.sum(np.sqrt((f - A @ u_current) ** 2))
        res_scaled = res / np.sum(np.sqrt(f ** 2))
        res_lst.append(res)
    return u_current, res_lst

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
    un = np.zeros(N - 1)  # solution vector; iterated
    res_scaled = 1  # initial residual
    res_lst = []
    #     Solver
    while res_scaled > rtol:
        for i in range(N - 2, -1, -1):  # iterate backwards;
            s1 = 0
            for j in range(i + 1, N - 1):  # sum of first part
                s1 += A[i, j] * un[j]
            s2 = 0
            for j in range(i, 0, -1):  # sum second part
                s2 += A[i, j - 1] * un[j - 1]
            un[i] = (f[i] - s1 - s2) / A[i, i]
            res = np.sum(np.sqrt((f - A @ un) ** 2))  # update residual;  using the infinity norm
            res_scaled = res / np.sum(np.sqrt(f ** 2))
            res_lst.append(res)
    return un, res_lst

def symmGS(N, eps, rtol=1e-6):
    # initial values
    h = 1 / N

    # Discretization scheme and right-hand vector; CDS
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs
    res = 1  # initial residual
    un = np.zeros(N - 1)  # solution vector; iterated
    res_scaled = 1  # initial residual
    res_lst = []

    while res_scaled > rtol:
        #        for i, row in enumerate(A):
        #             u_current[i] = (f[i]-row[:i]@u_current[:i]-row[i+1:]@u_current[i+1:])/A[i,i]
        #             res = f - A@u_current
        #             tol=np.max(res)/np.max(f)
        #        return u_current

        for i, row in enumerate(A):
            un[i] = (f[i] - row[:i] @ un[:i] - row[i + 1:] @ un[i + 1:]) / A[i, i]
        while res_scaled > rtol:
            for i in range(N - 2, -1, -1):  # iterate backwards;
                s1 = 0
                for j in range(i + 1, N - 1):  # sum of first part
                    s1 += A[i, j] * un[j]
                s2 = 0
                for j in range(i, 0, -1):  # sum second part
                    s2 += A[i, j - 1] * un[j - 1]
                un[i] = (f[i] - s1 - s2) / A[i, i]
            res = np.sum(np.sqrt((f - A @ un) ** 2))  # update residual;  using the infinity norm
            res_scaled = res / np.sum(np.sqrt(f ** 2))
            res_lst.append(res)
    return un, res_lst


fig, ax = plt.subplots(1, 2, dpi=150)
N = 8
eps = 1.0

#=======================================================================================================================
# RUN JACOBIAN AND DIRECT SOLVER
u_jac, res_lst_jac = jacobi(N, eps)
u0,A,f=system_solver(N,eps)

# compute residual plots for JAC
res_km1 = np.roll(res_lst_jac, shift=1)
red_jac = res_lst_jac / res_km1

# residual plots for JAC
ax[0].plot(res_lst_jac / np.max(f))
ax[1].plot(red_jac[2:], label="JAC")

#=======================================================================================================================
# RUN FORWARD FORWARD GAUSS-SEIDEL
u_gs, res_lst_gs = forwardGS(N, eps, rtol=1e-6)
# u_exact,A,f=system_solver(N,eps)
# print("Sol is close to exact sol: {}".format(np.allclose(u, u_exact[1:-1], rtol=1e-5)))
# compute residual parameters for GS
res_km1 = np.roll(res_lst_gs, shift=1)
red_gs = res_lst_gs / res_km1

# residual plots for GS
ax[0].plot(res_lst_gs / np.max(f))
ax[1].plot(red_gs[1:], label="forward GS")

#=======================================================================================================================
# RUN FORWARD FORWARD GAUSS-SEIDEL
u, res_lst_bgs = backwardGS1(N, eps)
# u_exact,A,f=system_solver(N,eps)
res_km1 = np.roll(res_lst_bgs, shift=1)
red_bgs = res_lst_bgs / res_km1

# add curve to plot
ax[0].plot(res_lst_bgs / np.max(f))
ax[1].plot(red_bgs[1:], label="backward GS")
#print("Sol is close to exact sol: {}".format(np.allclose(u, u_exact[1:-1])))

#=======================================================================================================================
# RUN FORWARD FORWARD GAUSS-SEIDEL
u, res_lst_symmgs = symmGS(N, eps)
# u_exact,A,f=system_solver(N,eps)

#print("Sol is close to exact sol: {}".format(np.allclose(u, u_exact[1:-1])))
res_km1 = np.roll(res_lst_symmgs, shift=1)
red_symmgs = res_lst_symmgs / res_km1

# add curve to plot
ax[0].plot(res_lst_symmgs / np.max(f))
ax[1].plot(red_symmgs[1:], label="symmetric GS")

# %%

# plot settings
ax[0].set_yscale("log")
# ax[1].set_yscale("log")
ax[0].grid()
ax[1].grid()
# ax[1].set_xlim(0, 5)
# ax[0].set_xlim(0, 5)
ax[1].legend()
# ax[1].set_yscale("log")
