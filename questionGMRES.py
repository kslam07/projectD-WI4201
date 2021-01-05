import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import gmres
from scipy.linalg import solve_triangular

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
        self.rk_lst = [1]
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
                 , callback_type="pr_norm", restart=len(A), maxiter=len(A))[0], counter.rk_lst

def test_gmres_solver2D(N, eps, rtol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Discretisation
    A = A2dim(N)
    f = np.ones(len(A))

    counter = gmres_counter()
    return gmres(A, f, tol=rtol, callback=counter, M=np.linalg.inv(np.identity(len(A)) * A[0,0])
                 , callback_type="pr_norm", restart=len(A), maxiter=len(A))[0], counter.rk_lst

def A2dim(n):
    # n=5 #number of outer points
    # P=12 # number of panels
    N=n**2 #number of points in u vector
    h=1/N # spacing
    uijp1=-1
    uim1j=-1-h
    uij=4+h
    uip1j=-1
    uijm1=-1
    e=1
    
    # uijp1=1
    # uim1j=2
    # uij=3
    # uip1j=4
    # uijm1=5
    
    # A = scipy.sparse.diags([-e / h - 1, 2 * e / h + 1, -e / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    A = scipy.sparse.diags([uijp1, uim1j, uij, uip1j, uijm1], [-n, -1, 0, 1, n],shape=(N, N)).toarray()
    # Only from nth plus one th column matrix is needed, the rest in bc
    # print(np.matrix(A))
    
    # A = np.eye(N-1,N-1,k=-1)+np.eye(N-1,N-1,k=n)
    # # A=np.eye(N-1,N-1,k=(N-2))
    # print(A)
    
    r=np.concatenate((np.arange(n),np.arange(1,n+1)*-1))
    r_col=[]
    il=0
    ir=n-1
    for i in range(0,n-2):
        r_col.extend((il+n*i,ir+n*i))

    A=np.delete(A,r,1)
    A=np.delete(A,r_col,1)
    A=np.delete(A,r,0)
    A=np.delete(A,r_col,0)
    return A/h/h

def full_gmres1D(N, eps, u0, max_iter=None, atol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines

    # Discretisation
    A = scipy.sparse.diags([-eps / h - 1, 2 * eps / h + 1, -eps / h], [-1, 0, 1], shape=(N - 1, N - 1)).toarray()
    M_inv = np.linalg.inv(np.identity(N - 1) * (2 * eps / h))  # Jacobi left preconditioner
    f = np.zeros(N - 1)
    f[0] = eps / h + 1  # bring bc to rhs

    # initialise arrays
    #res_scaled = np.linalg.norm(f - A @ un) / np.linalg.norm(f)
    un = u0.copy()
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
        if j != m - 1:  #  check if it is the end of the Hessenberg matrix
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

def full_gmres2D(N, eps, u0, max_iter=None, atol=1e-6):
    # Initial values
    h = 1 / N  # nr of lines
    # Discretisation
    A = A2dim(N)
    M_inv = np.linalg.inv(np.eye(len(A))*A[0,0])
    f = np.ones(len(A))

    # initialise arrays
    #res_scaled = np.linalg.norm(f - A @ un) / np.linalg.norm(f)
    un = u0.copy()
    m = min(max_iter, A.shape[0]) if max_iter is not None else A.shape[0] # full GMRES or restarted GMRES
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
        if j != m - 1:  #  check if it is the end of the Hessenberg matrix
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

    t = np.sqrt(hcol_k**2 + hcol_kp1**2)
    cs = hcol_k / t
    sn = hcol_kp1 / t

    return cs, sn

#when using h as an input use:N=h+1
N=5
eps = 0.5
u_exact, A, f = system_solver(N, eps)
A=A2dim(N)
D = np.diag(np.ones(len(A)) * A[0, 0])
B_jac = np.identity(len(A)) - np.matmul(np.linalg.inv(D), A)
ev, ef = np.linalg.eig(B_jac)
a, res_lst_ex = test_gmres_solver2D(N, eps, rtol=1e-6)
a1, res_lst_gm = full_gmres2D(N, eps, np.zeros(A.shape[0]),atol=1e-10)
y=np.zeros(len(ev))
plt.scatter(ev,y,marker='o')
a=(np.max(ev)-np.min(ev))/2
b=0.0
ev=np.sort(ev)
y=np.sqrt(b*(1-ev**2/a))
t=np.linspace(0,2*np.pi,100)
plt.plot(np.max(ev)*np.cos(t),b*np.sin(t))
# plt.axis('scaled')
plt.grid()
plt.xlabel('Re (-)')
plt.ylabel('Im (-)')
plt.savefig('ev_plot14',dpi=250)
# fig, ax = plt.subplots(1, 1, dpi=100)

# ax.plot(res_lst_gm)
# ax.plot(res_lst_ex)
# ax.grid()
# ax.set_yscale("log")