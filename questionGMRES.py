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


def test_gmres_solver(N, eps, rtol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Discretisation
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs

    return gmres(A, f, tol=rtol)


def full_gmres(N, eps, u0, m=None, atol=1e-6):

    # Initial values
    h = 1 / N  # nr of lines

    # Discretisation
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    M_inv = np.linalg.inv(np.identity(N - 1) * (2 * eps / h))  # Jacobi left preconditioner
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs
    un = (u0.copy()).reshape(1, -1)
    res_scaled = 1
    res_lst = []
    iter_outer = min(m, A.shape[0]) if m is not None else A.shape[0]

    while res_scaled > atol:
        ui = un[-1]  # update starting vector in case of restart GMRES
        vmat = np.zeros((N - 1, A.shape[0]))

        # compute r0 and v1, and M
        r = M_inv @ (f - A @ ui)
        vmat[:, 0] = r / np.linalg.norm(r)

        # initialise Hessenberg matrix
        hmat = np.zeros((A.shape[0] + 1, A.shape[0]))

        # compute Hessenberg matrix
        for j in range(iter_outer):
            # compute the Hessenerg matrix
            v_iter = M_inv @ A @ vmat[:, j]  # Krylov vector of column j with Jacobi left preconditioner
            for i in range(j + 1):
                hmat[i, j] = v_iter.T @ vmat[:, i]
                v_iter = v_iter - hmat[i, j] * vmat[:, i]

            # update last element in ith row, jth col
            hmat[j + 1, j] = np.linalg.norm(v_iter)
            if j != A.shape[0] - 1:
                vmat[:, j+1] = v_iter / hmat[j + 1, j]

            # find the optimal fit of the sol. vector in the Krylov space
            b = np.zeros(A.shape[0] + 1)
            b[0] = M_inv[0, 0] * f[0]  # b[0] = ||r||2 but r = f in this case and ||.||2 is the first value

            yi = np.linalg.lstsq(hmat, b, rcond=None)[0]

            # updates
            un = np.vstack((un, vmat.T @ yi + ui))  # assuming initial guess is the 0th vector
            res = np.linalg.norm(M_inv @ (f - A @ un[-1]))  # residual 2-norm
            res_scaled = res / np.linalg.norm(f)
            res_lst.append(res_scaled)
        print(res)
    return un, res_lst

N = 8
eps = 0.5
u_exact, A, f = system_solver(N, eps)
D = np.diag(np.ones(N - 1) * A[0, 0])
B_jac = np.identity(N - 1) - np.matmul(np.linalg.inv(D), A)
ev, ef = np.linalg.eig(B_jac)
a = test_gmres_solver(N, eps)
a1, res_lst_gm = full_gmres(N, eps, np.zeros(A.shape[0]), m=None)


fig, ax = plt.subplots(1, 1, dpi=100)

ax.plot(res_lst_gm)
ax.grid()
ax.set_yscale("log")