import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import gmres
from scipy.linalg import solve_triangular

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

    return gmres(A, f, tol=rtol, maxiter=3, restart=1)


def full_gmres(N, eps, u0, max_iter=None, atol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Discretisation
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    M_inv = np.linalg.inv(np.identity(N - 1) * (2 * eps / h))  # Jacobi left preconditioner
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs

    # initialise arrays
    #res_scaled = np.linalg.norm(f - A @ un) / np.linalg.norm(f)
    m = min(max_iter, A.shape[0]) if max_iter is not None else A.shape[0] # full GMRES or restarted GMRES
    vmat = np.zeros((N - 1, m))  # basis functions in Krylov space
    hmat = np.zeros((m + 1, m))  # Hessenberg matrix
    sn_vec = np.zeros(m)
    cs_vec = np.zeros(m)
    beta = np.zeros(m + 1)  # error vector

    # compute r0 and v1
    r = M_inv @ (f - A @ u0)
    vmat[:, 0] = r / np.linalg.norm(r)

    # residual stuff
    beta[0] = 1
    beta = np.linalg.norm(r) * beta
    res_lst = [1]

    # compute Hessenberg matrix
    for j in range(m):
        # compute the Hessenerg matrix
        v_iter = M_inv @ A @ vmat[:, j]  # Krylov vector of column j with Jacobi left preconditioner
        for i in range(j + 1):
            hmat[i, j] = v_iter.T @ vmat[:, i]
            v_iter = v_iter - hmat[i, j] * vmat[:, i]

        # update last element in ith row, jth col
        hmat[j + 1, j] = np.linalg.norm(v_iter)
        if j != m - 1:
            vmat[:, j + 1] = v_iter / hmat[j + 1, j]

        # triangulize Hessenberg matrix
        hcol, cs_vec, sn_vec = apply_rotation(hmat[:, j], cs_vec, sn_vec, j)

        # update residual vector
        beta[j + 1] = -sn_vec[j] * beta[j]
        beta[j] = cs_vec[j] * beta[j]
        res_lst.append(np.abs(beta[j + 1]) / np.linalg.norm(f))

    y = solve_triangular(hmat[:m, :m], beta[:m])  # solve the triangular system without inversion
    un = u0 + vmat @ y

    return un, res_lst

def apply_rotation(hcol, cs_vec, sn_vec, k):
    """
    :param hmat: Hessenberg Matrix
    :param cs: cosine value vector
    :param sn: sine value vector
    :param k: kth iteration
    :return:
    """

    for i in range(k):
        temp = cs_vec[i] * hcol[i] + sn_vec[i] * hcol[i + 1]  # use ^ith hcol values
        hcol[i + 1] = -sn_vec[i] * hcol[i] + cs_vec[i] * hcol[i + 1]  # use h^i
        hcol[i] = temp

    # update cs and sn
    cs_k, sn_k = update_rotations(hcol[k], hcol[k + 1])
    hcol[k] = cs_k * hcol[k] + sn_k * hcol[k + 1]
    hcol[k + 1] = 0.0

    cs_vec[k] = cs_k
    sn_vec[k] = sn_k

    return hcol, cs_vec, sn_vec

def update_rotations(hcol_k, hcol_kp1):

    t = np.sqrt(hcol_k**2 + hcol_kp1**2)
    cs = hcol_k / t
    sn = hcol_kp1 / t

    return cs, sn

N = 8
eps = 0.5
u_exact, A, f = system_solver(N, eps)
D = np.diag(np.ones(N - 1) * A[0, 0])
B_jac = np.identity(N - 1) - np.matmul(np.linalg.inv(D), A)
ev, ef = np.linalg.eig(B_jac)
a = test_gmres_solver(N, eps)
a1, res_lst_gm = full_gmres(N, eps, np.zeros(A.shape[0]), max_iter=5)

fig, ax = plt.subplots(1, 1, dpi=100)

ax.plot(res_lst_gm)
ax.grid()
ax.set_yscale("log")
