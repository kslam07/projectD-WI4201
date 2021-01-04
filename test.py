# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 17:55:00 2020

@author: Thomas Verduyn
"""
# setup problem parameters/functions
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import sympy
import copy

def system_solver(N,e): # sets up system Au=f and solves it
    # Initial values
    h=1/N # nr of lines
    
    # Constant values - Boundary conditions
    u0=1
    unp1=0
    
    # Discretisation
    A=scipy.sparse.diags([-e/h-1, 2*e/h+1, -e/h], [-1, 0, 1], shape=(N-1, N-1)).toarray() 
    f=np.zeros(N-1)
    f[0]=e/h+1  # bring bc to rhs
    un=np.linalg.inv(A)@f
    return np.concatenate(([u0],un,[unp1])), A, f

def system_solver2(N,e):
    # Initial values
    h=1/N
    
    # Constant values - Boundary conditions
    u0=1
    unp1=0
    
    # Discretisation
    A=scipy.sparse.diags([-e/h-1, 2*e/h+1, -e/h], [-1, 0, 1], shape=(N-1, N-1)).toarray()
    A=np.vstack((np.zeros((1,N-1)),A,np.zeros((1,N-1))))
    A=np.hstack((np.zeros((N+1,1)),A,np.zeros((N+1,1))))
    A[0,0]=u0
    A[-1,-1]=unp1
    print(A)
    f=np.zeros(N+1)
    f[0]=e/h+1
    un=np.linalg.inv(A)@f
    return np.concatenate(([u0],un,[unp1])), A, f

def testfunc(N,e):
    x=np.linspace(0,1,N+1)
    return (np.exp(x/e)-np.exp(1/e))/(1-np.exp(1/e))

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
        res_scaled = np.linalg.norm(res) / np.linalg.norm(f)
        res_lst.append(np.linalg.norm(res))

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
        res = np.linalg.norm(f - A @ u_current)
        res_scaled = res / np.linalg.norm(f)
        res_lst.append(res)
    return u_current, res_lst

def backwardGS1(N, eps, rtol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

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
        res = np.linalg.norm(f - A @ un)
        res_scaled = res / np.linalg.norm(f)
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
            res = np.linalg.norm(f - A @ un)
            res_scaled = res / np.linalg.norm(f)
            res_lst.append(res)
    return un, res_lst




def plot_jac(N):
    u_jac, res_lst_jac = jacobi(N, eps)
    u0,A,f=system_solver(N,eps)
    # compute residual plots for JAC
    res_km1 = np.roll(res_lst_jac, shift=1)
    red_jac = res_lst_jac / res_km1
    
    # residual plots for JAC
    # fig,ax=plt.subplots(1,2)
    # ax[0].semilogy(res_lst_jac / np.max(f))
    # ax[1].plot(red_jac[1:], label="JAC")
    plt.semilogy(res_lst_jac/np.max(f),label='N='+str(N))
    print('For N='+str(N)+r'$red^k$',np.average(red_jac[-5:]))
    

def plot_fgs(N):
    u_fgs, res_lst_fgs = forwardGS(N, eps)
    u0,A,f=system_solver(N,eps)
    # compute residual plots for JAC
    res_km1 = np.roll(res_lst_fgs, shift=1)
    red_fgs = res_lst_fgs / res_km1
    
    # residual plots for JAC
    # fig,ax=plt.subplots(1,2)
    # ax[0].semilogy(res_lst_jac / np.max(f))
    # ax[1].plot(red_jac[1:], label="JAC")
    plt.semilogy(res_lst_fgs/np.max(f),label='N='+str(N))
    print('For N='+str(N)+r'$red^k$',np.average(red_fgs[-5:]))


def plot_bgs(N):
    u_bgs, res_lst_bgs = backwardGS1(N, eps)
    u0,A,f=system_solver(N,eps)
    # compute residual plots for JAC
    res_km1 = np.roll(res_lst_bgs, shift=1)
    red_bgs = res_lst_bgs / res_km1
    
    # residual plots for JAC
    # fig,ax=plt.subplots(1,2)
    # ax[0].semilogy(res_lst_jac / np.max(f))
    # ax[1].plot(red_jac[1:], label="JAC")
    plt.semilogy(res_lst_bgs/np.max(f),label='N='+str(N))
    print('For N='+str(N)+r'$red^k$',np.average(red_bgs[-5:]))
    
    
def plot_sgs(N):
    u_sgs, res_lst_sgs = symmGS(N, eps)
    u0,A,f=system_solver(N,eps)
    # compute residual plots for JAC
    res_km1 = np.roll(res_lst_sgs, shift=1)
    red_sgs = res_lst_sgs / res_km1
    
    # residual plots for JAC
    # fig,ax=plt.subplots(1,2)
    # ax[0].semilogy(res_lst_jac / np.max(f))
    # ax[1].plot(red_jac[1:], label="JAC")
    plt.semilogy(res_lst_sgs/np.max(f),label='N='+str(N))
    print('For N='+str(N)+r'$red^k$',np.average(red_sgs[-5:]))
    
    
# =============================================================================
#     subquestion 5
# =============================================================================
# N=512
# eps=0.5
# d=system_solver(N, eps)
# u,A,f=d
# D = np.diag(np.ones(N-1)*A[0,0])
# B_jac=np.identity(N-1)-np.matmul(np.linalg.inv(D),A)
# print(np.linalg.eigvals(B_jac))
# print(np.max(np.linalg.eigvals(B_jac)))
    
# =============================================================================
#     q6
# =============================================================================
# eps=1
# lst=[16,32,64,128,256]
# for i in lst:
#     plot_jac(i)
    
# plt.xlabel('Iterations (-)')
# plt.ylabel('r$||r^k||/||f^h||$')
# plt.legend()
# plt.show
# plt.grid()
# plt.savefig('jacobi_iter',dpi=250)


# for N in lst:
#     # N=N-1
#     d=system_solver(N,eps)
#     u,A,f=d
#     D = np.diag(np.ones(N-1)*A[0,0])
#     print(np.sqrt(np.size(D)))
#     B_jac=np.identity(N-1)-np.matmul(np.linalg.inv(D),A)
#     print(np.sqrt(np.size(B_jac)))
#     print('For N='+str(N)+r' $\rho$ (a)=',np.max(np.abs(np.linalg.eigvals(B_jac))))


# =============================================================================
#  subquestion 7
# =============================================================================
# eps=1
# lst=[8]
# lst=[16,32,64,128,256]
# for i in lst:
#     plot_fgs(i)
    
# plt.xlabel('Iterations (-)')
# plt.ylabel('r$||r^k||/||f^h||$')
# plt.legend()
# plt.show
# plt.grid()
# plt.savefig('fgs_iter',dpi=250)


# for N in lst:
#     # N=N-1
#     d=system_solver(N,eps)
#     u,A,f=d
#     D = np.diag(np.ones(N-1)*A[0,0])
#     print(np.sqrt(np.size(D)))
#     E = np.diag(np.ones(N-2)*A[1,0],-1)
#     B_fgs=np.identity(N-1)-np.matmul(np.linalg.inv(D+E),A)
#     ev=np.linalg.eigvals(B_fgs)
#     # print(np.sqrt(np.size(B_fgs)))
#     # if N==7:
#         # print(E)
#     print('For N='+str(N)+r' $\rho$ (a)=',np.max(np.abs(np.linalg.eigvals(B_fgs))))
    
#plt.scatter(ev.real,ev.imag)

# =============================================================================
#       sub 8
# =============================================================================
# eps=1
# lst=[16,32,64,128]
# for i in lst:
#     plot_bgs(i)
    
# plt.xlabel('Iterations (-)')
# plt.ylabel('r$||r^k||/||f^h||$')
# plt.legend()
# plt.show
# plt.grid()
# plt.savefig('bgs_iter',dpi=250)


# for N in lst:
#     N=N-1
#     d=system_solver(N,1)
#     u,A,f=d
#     D = np.diag(np.ones(N-1)*A[0,0])
#     F = np.diag(np.ones(N-2)*A[0,1],1)
#     B_bgs=np.identity(N-1)-np.matmul(np.linalg.inv(D+F),A)
#     # print(np.sqrt(np.size(B_bgs)))
#     if N==7:
#         print(F)
#     print('For N='+str(N)+r' $\rho$ (a)=',np.max(np.abs(np.linalg.eigvals(B_bgs))))


# =============================================================================
#       sub 9
# =============================================================================
# eps=1
# lst=[16,32,64,128]
# for i in lst:
#     plot_sgs(i)
    
# plt.xlabel('Iterations (-)')
# plt.ylabel('r$||r^k||/||f^h||$')
# plt.legend()
# plt.show
# plt.grid()
# plt.savefig('sgs_iter',dpi=250)


# for N in lst:
#     N=N-1
#     d=system_solver(N,1)
#     u,A,f=d
#     D = np.diag(np.ones(N-1)*A[0,0])
#     F = np.diag(np.ones(N-2)*A[0,1],1)
#     B_sgs=np.identity(N-1)-np.matmul(np.linalg.inv(D+F),A)
#     print(np.sqrt(np.size(B_sgs)))
#     print('For N='+str(N)+r' $\rho$ (a)=',np.max(np.abs(np.linalg.eigvals(B_sgs))))

# =============================================================================
#       sub 10/11
# =============================================================================
# N=50
# eps=1
# u,A,f=system_solver(N,eps)
# D = np.diag(np.ones(N-1)*A[0,0])
# B_jac=np.identity(N-1)-np.matmul(np.linalg.inv(D),A)
# ev, ef=np.linalg.eig(B_jac)
# #print(np.linalg.eig(A)[0])
# # print(max(np.abs(ev)))
# # #print(A)
# # print(B_jac)
# # print(ev)
# y=np.zeros(len(ev))
# plt.scatter(ev,y,marker='o')
# a=(np.max(ev)-np.min(ev))/2
# b=0.0
# ev=np.sort(ev)
# y=np.sqrt(b*(1-ev**2/a))
# t=np.linspace(0,2*np.pi,100)
# plt.plot(np.max(ev)*np.cos(t),b*np.sin(t))
# # plt.axis('scaled')
# plt.grid()
# plt.xlabel('Re (-)')
# plt.ylabel('Im (-)')
# # plt.savefig('ev_plot11',dpi=250)