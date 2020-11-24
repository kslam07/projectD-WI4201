# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 09:07:57 2020

@author: Thomas Verduyn
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse

def opgaveD1(N,e):
    # Initial values
    x= np.linspace(0,1,N)
    h=1/N
    
    # Constant values - Boundary conditions
    u0=1
    unp1=0
    
    # Discretisation
    A=scipy.sparse.diags([-e/h-1, -2*e/h+1, -e/h], [-1, 0, 1], shape=(N, N)).toarray()
    f=np.zeros(N)
    f[0]=e/h+1
    return np.linalg.inv(A)@f

def testfunc(N,e):
    x=np.linspace(0,1,N)
    return (np.exp(x/e)-np.exp(1/e))/(1-np.exp(1/e))

N=3
eps = np.linspace(0.000001,1,5)
for e in eps:
    plt.plot(np.linspace(0,1,N),opgaveD1(N,e),label=str(e),ls='dotted')
    plt.plot(np.linspace(0,1,N),testfunc(N,e),label='testfunc'+str(e))

plt.legend()