# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 09:07:57 2020

@author: Thomas Verduyn
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from sympy import Matrix

def opgaveD1(N,e):
    # Initial values
    x= np.linspace(0,1,N)
    h=1/N
    
    # Constant values - Boundary conditions
    u0=1
    unp1=0
    
    # Discretisation
    A=scipy.sparse.diags([-e/h-1, 2*e/h+1, -e/h], [-1, 0, 1], shape=(N, N)).toarray()
    print (A)
    f=np.zeros(N)
    f[0]=e/h+1
    un=np.linalg.inv(A)@f
    return np.concatenate(([u0],un,[unp1])), A, f

def testfunc(N,e):
    x=np.linspace(0,1,N)
    return (np.exp(x/e)-np.exp(1/e))/(1-np.exp(1/e))

#N=2
#eps = np.linspace(1e-2,1,11)
#for e in eps:
##    plt.plot(np.linspace(0,1,N+2),opgaveD1(N,e)[0],label=str(e),ls='dotted')
##    plt.plot(np.linspace(0,1,N+2),testfunc(N+2,e),label='testfunc'+str(e))
#    plt.plot(np.linspace(0,1,N+2),(testfunc(N+2,e)-opgaveD1(N,e)[0])/testfunc(N+2,e),label=str(e))
#plt.legend()

# =============================================================================
# OpgaveD2 
# =============================================================================

#n=5
#eps=0.5
#error=[]
#for i in range(4,n+4):
#    N=2**i
#    print(N)
#    un=opgaveD1(N,eps)
#    error.append(max(np.abs(opgaveD1(N,eps)[0]-testfunc(N+2,eps))))    
#plt.plot(range(5),error)
#plt.yscale('log')


# =============================================================================
#  OpgaveD3
# =============================================================================
A=opgaveD1(10,0.5)[1]
np.allclose(A,A.T)
print(sympy.Matrix(A).rref())