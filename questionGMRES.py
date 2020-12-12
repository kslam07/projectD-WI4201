import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import gmres

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

def gmres_solver(N, eps, rtol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Discretisation
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs

    return gmres(A, f, tol=rtol)



N=15
eps=0.5
u_exact, A, f =system_solver(N, eps)
D = np.diag(np.ones(N-1)*A[0,0])
B_jac=np.identity(N-1)-np.matmul(np.linalg.inv(D),A)
ev, ef=np.linalg.eig(B_jac)
a = gmres_solver(15, eps)

#print(max(np.abs(ev)))
#print(ev)
