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


def gmres_solver(N, eps, rtol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Discretisation
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs
    un = np.zeros(A.shape)
    vmat = np.zeros((N - 1, A.shape[0]))
    # compute r0 and v1
    r = f - A @ un[:, 0]
    vmat[:, 0] = r / np.linalg.norm(r)

    # initialise Hessenberg matrix
    hmat = np.zeros((A.shape[0] + 1, A.shape[0]))

    # compute Hessenberg matrix
    for j in range(A.shape[0]):
        v_iter = A @ vmat[:, j]  # Krylov vector of column j
        for i in range(j + 1):
            hmat[i, j] = v_iter.T @ vmat[:, i]
            v_iter = v_iter - hmat[i, j] * vmat[:, i]
        # update last element in ith row, jth col
        hmat[j + 1, j] = np.linalg.norm(v_iter)
        if j != A.shape[0] - 1:
            vmat[:, j+1] = v_iter / hmat[j + 1, j]

    # TODO: remove this I don't know what happens from here
        b = np.zeros(A.shape[0] + 1)
        b[0] = f[0]  # b[0] = np.linalg.norm(r) but r = f in this case and norm2 is just the first value

        ui = np.linalg.lstsq(hmat, b, rcond=None)[0]
        un[j, :] = vmat.T @ ui  # assuming initial guess is the 0th vector
    return un

def GMRes(A, b, x0, nmax_iter, restart=None):
    r = b - np.asarray(np.dot(A, x0)).reshape(-1)

    x = []
    q = [0] * (nmax_iter)

    x.append(r)

    q[0] = r / np.linalg.norm(r)

    h = np.zeros((nmax_iter + 1, nmax_iter))

    for k in range(min(nmax_iter, A.shape[0])):
        y = np.asarray(np.dot(A, q[k])).reshape(-1)

        for j in range(k + 1):
            h[j, k] = np.dot(q[j], y)
            y = y - h[j, k] * q[j]
        h[k + 1, k] = np.linalg.norm(y)
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            q[k + 1] = y / h[k + 1, k]

        b = np.zeros(nmax_iter + 1)
        b[0] = np.linalg.norm(r)

        result = np.linalg.lstsq(h, b)[0]

        x.append(np.dot(np.asarray(q).transpose(), result) + x0)

    return h


N = 8
eps = 0.5
u_exact, A, f = system_solver(N, eps)
D = np.diag(np.ones(N - 1) * A[0, 0])
B_jac = np.identity(N - 1) - np.matmul(np.linalg.inv(D), A)
ev, ef = np.linalg.eig(B_jac)
a = test_gmres_solver(15, eps)
a1 = gmres_solver(N, eps)
a2 = GMRes(A, f, np.zeros(N - 1), nmax_iter=A.shape[0])
pass
# print(max(np.abs(ev)))
# print(ev)
