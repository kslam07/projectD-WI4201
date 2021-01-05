import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import gmres
from scipy.linalg import solve_triangular


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.rk_lst = [1.0]

    def __call__(self, pr_norm=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(pr_norm)))
            self.rk_lst.append(pr_norm)


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


def test_gmres_solver1D(N, eps, rtol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Discretisation
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs

    counter = gmres_counter()
    return gmres(A, f, tol=rtol, callback=counter, M=np.linalg.inv(np.identity(N - 1) * (2 * eps / h))
                 , callback_type="pr_norm", restart=len(A), maxiter=1)[0], counter.rk_lst


def test_gmres_solver2D(N, eps, rtol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Discretisation
    A = A2dim(N)
    f = np.ones(len(A))

    counter = gmres_counter()
    return gmres(A, f, tol=rtol, callback=counter, M=np.linalg.inv(np.identity(len(A)) * A[0, 0])
                 , callback_type="pr_norm", restart=len(A), maxiter=1)[0], counter.rk_lst


def A2dim(n):
    # n=5 #number of outer points
    # P=12 # number of panels
    N = n ** 2  # number of points in u vector
    h = 1 / N  # spacing
    uijp1 = -1
    uim1j = -1 - h
    uij = 4 + h
    uip1j = -1
    uijm1 = -1
    e = 1

    # uijp1=1
    # uim1j=2
    # uij=3
    # uip1j=4
    # uijm1=5

    # A = scipy.sparse.diags([-e / h - 1, 2 * e / h + 1, -e / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    A = scipy.sparse.diags([uijp1, uim1j, uij, uip1j, uijm1], [-n, -1, 0, 1, n], shape=(N, N)).toarray()
    # Only from nth plus one th column matrix is needed, the rest in bc
    # print(np.matrix(A))

    # A = np.eye(N-1,N-1,k=-1)+np.eye(N-1,N-1,k=n)
    # # A=np.eye(N-1,N-1,k=(N-2))
    # print(A)

    r = np.concatenate((np.arange(n), np.arange(1, n + 1) * -1))
    r_col = []
    il = 0
    ir = n - 1
    for i in range(0, n - 2):
        r_col.extend((il + n * i, ir + n * i))

    # big fix for Sam
    r = np.concatenate((r[r >= 0], [ri + n**2 for ri in r if ri < 0]))

    A = np.delete(A, r, 1)
    A = np.delete(A, r_col, 1)
    A = np.delete(A, r, 0)
    A = np.delete(A, r_col, 0)
    return A / h / h


def full_gmres1D(N, eps, u0, max_iter=None, atol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Discretisation
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    M_inv = np.linalg.inv(np.identity(N - 1) * (2 * eps / h))  # Jacobi left preconditioner
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs

    # initialise arrays
    # res_scaled = np.linalg.norm(f - A @ un) / np.linalg.norm(f)
    un = u0.copy()
    m = min(max_iter, A.shape[0]) if max_iter is not None else A.shape[0]  # full GMRES or restarted GMRES
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
        if j != m - 1:  # check if it is the end of the Hessenberg matrix
            vmat[:, j + 1] = v_iter / hmat[j + 1, j]

        # triangulize Hessenberg matrix
        hcol, cs_vec, sn_vec = apply_rotation(hmat[:, j], cs_vec, sn_vec, j)

        # update residual vector
        beta[j + 1] = -sn_vec[j] * beta[j]
        beta[j] = cs_vec[j] * beta[j]
        res_lst.append(np.abs(beta[j + 1]) / np.linalg.norm(f))

        if res_lst[j] < atol:
            break

    y = solve_triangular(hmat[:j + 1, :j + 1], beta[:j + 1])  # solve the triangular system without inversion
    un[:j + 1] = u0[:j + 1] + vmat[:j + 1, :j + 1] @ y

    return un, res_lst


def full_gmres2D(N, u0, max_iter=None, atol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines
    # Discretisation
    A = A2dim(N)
    M_inv = np.linalg.inv(np.eye(len(A)) * A[0, 0])
    f = np.ones(len(A))

    # initialise arrays
    # res_scaled = np.linalg.norm(f - A @ un) / np.linalg.norm(f)
    un = u0.copy()
    m = min(max_iter, A.shape[0]) if max_iter is not None else A.shape[0]  # full GMRES or restarted GMRES
    vmat = np.zeros((len(A), m))  # basis functions in Krylov space
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
        if j != m - 1:  # check if it is the end of the Hessenberg matrix
            vmat[:, j + 1] = v_iter / hmat[j + 1, j]

        # triangulize Hessenberg matrix
        hcol, cs_vec, sn_vec = apply_rotation(hmat[:, j], cs_vec, sn_vec, j)

        # update residual vector
        beta[j + 1] = -sn_vec[j] * beta[j]
        beta[j] = cs_vec[j] * beta[j]
        res_lst.append(np.abs(beta[j + 1]) / np.linalg.norm(f))

        if res_lst[j] < atol:
            break

    y = solve_triangular(hmat[:j + 1, :j + 1], beta[:j + 1])  # solve the triangular system without inversion
    un[:j + 1] = u0[:j + 1] + vmat[:j + 1, :j + 1] @ y

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
    t = np.sqrt(hcol_k ** 2 + hcol_kp1 ** 2)
    cs = hcol_k / t
    sn = hcol_kp1 / t

    return cs, sn

def restart_GMRES(N, eps, atol=1e-6, max_iter=1):

    # first call
    u_re, res_lst_re = full_gmres1D(N, eps, u0=np.zeros(N - 1), max_iter=max_iter)
    # second call; use restarted sol. as initial sol. vector
    u_gm, res_lst_gm = full_gmres1D(N, eps, u0=u_re, atol=atol)

    return u_gm, np.concatenate((res_lst_re, res_lst_gm[1:]))

def test_restart_GMRES(N, eps, max_iter):
    # Initial values
    h = 1 / N  # nr of lines

    # Discretisation
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs

    counter = gmres_counter()
    u_re, res_lst_re = gmres(A, f, callback=counter, M=np.linalg.inv(np.identity(N - 1) * (2 * eps / h))
                 , callback_type="pr_norm", restart=max_iter, maxiter=1)[0], counter.rk_lst
    counter = gmres_counter()
    u_gm, res_lst_gm = gmres(A, f, x0=u_re, callback=counter, M=np.linalg.inv(np.identity(N - 1) * (2 * eps / h))
                 , callback_type="pr_norm", restart=len(A), maxiter=1)[0], counter.rk_lst
    return u_gm, np.concatenate((res_lst_re, res_lst_gm[1:]))

# =========================================== Q13 plotting code =================================================
n = 512
m_range = np.linspace(0.10, 0.5, 4) * n
fig, ax = plt.subplots(2, 2, sharey=True, dpi=200)
ax = ax.flatten()
for i, m in enumerate(m_range):
    u_re, res_re = restart_GMRES(n, 1, max_iter=int(m), atol=1e-9)
    u_py, res_py = test_restart_GMRES(n, 1, max_iter=int(m))
    u_full, res_full = full_gmres1D(n, 1, u0=np.zeros(n - 1), atol=1e-9)
    ax[i].plot(res_re, label="Restarted GMRES(m)")
    ax[i].plot(res_full, linestyle='dashdot', label="Full GMRES")
    ax[i].plot(res_py, linestyle='--', c='k', label="Scipy GMRES")
    ax[i].set_title("M={}".format(int(m)))
    ax[i].set_yscale("log")
    ax[i].grid()
    ax[i].set_xlabel("k [-]", fontsize=12)
    ax[i].set_ylabel(r"$\parallel{\mathbf{r}^{k}}\parallel/\parallel\mathbf{f}^{h}\parallel$", fontsize=12)
    ax[i].label_outer()
ax[0].legend()


# =========================================== Q12 plotting code =================================================
# n_range = 2 ** np.arange(4, 10, 1)
# eps_range = [0.1, 0.5, 1.0]
#
# fig, ax = plt.subplots(1, 3, dpi=200, sharex=False, sharey=True)
# ax = ax.flatten()
#
# for i, eps in enumerate(eps_range):
#
#     for j, N in enumerate(n_range):
#         # u_exact, A, f = system_solver(N, eps)
#         # A=A2dim(N)
#         # D = np.diag(np.ones(len(A)) * A[0, 0])
#         # B_jac = np.identity(len(A)) - np.matmul(np.linalg.inv(D), A)
#         # ev, ef = np.linalg.eig(B_jac)
#         u_test, res_lst_test = test_gmres_solver1D(N, eps, rtol=1e-6)
#         # a1, res_lst_gm = full_gmres2D(N, eps, np.zeros(A.shape[0]),atol=1e-10)
#         u_gm, res_lst_gm = full_gmres1D(N, eps, np.zeros(N - 1))
#
#         ax[i].plot(res_lst_test, label="Scipy's GMRES" if j == 0 else "", linestyle='--', c='k')
#         ax[i].plot(res_lst_gm, label=r"N={}".format(N))
#         ax[i].set_yscale("log")
#         ax[i].grid()
#         ax[i].set_xlabel("k [-]", fontsize=12)
#         ax[i].set_ylabel(r"$\parallel{\mathbf{r}^{k}}\parallel/\parallel\mathbf{f}^{h}\parallel$", fontsize=12)
#         ax[i].label_outer()
#     ax[i].axhline(1e-6, c='k', linestyle='dashdot', label=r"tol=$10^{-6}$" if i == 0 else "")
#     ax[i].set_title(r"$\epsilon$={}".format(eps))
#
# ax[0].legend(prop={"size": 8})



