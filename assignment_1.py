import numpy as np
import numpy.linalg as nplg
import scipy.sparse as spsp
import scipy.linalg as splg
import time
import scipy.sparse.linalg as spsplg
from scipy.sparse.linalg import inv

def system_matrix(n):
    A = 1*np.diag(np.ones(n-1), 1) + 1 * \
        np.diag(np.ones(n-1), -1) + 4*np.diag(np.ones(n), 0)
    A = spsp.csr_matrix(A)
    return A


def jacobi_setup(A): 
    D = spsp.diags(A.diagonal())
    L = spsp.tril(A,k=-1)
    U = spsp.triu(A,k=1)
    Dinv = inv(D)

    return Dinv, L+U
    

def jacobi_solve(Dinv, L_plus_U, b, x0=None, tol=1e-6):

   # Dinv:     inv(D), where D is the diagonal part of A
   # L_plus_U: L+U (upper and lower triangular parts of A)
   # b:        right-hand side vector
   # x0:       Initial guess (if None, the zero vector is used)
   # tol:      Relative error tolerance

   # If no initial guess supplied, use the zero vector
    R = -Dinv*L_plus_U
    d = Dinv*b
    n_iter = 0
    if x0 is None:
        x_new = 0*b
    else:
        x_new = x0
    err = 2*tol
    while err > tol:
       n_iter+=1
       x_old = x_new
       
       x_new = R*x_new + d
       err = nplg.norm(x_new - x_old)/nplg.norm(x_new)
    #    print(err)

    return x_new, n_iter


def conjugate_gradient_solve(A, b, x0=None, tol=1e-6):

    # If no initial guess supplied, use the zero vector
    if x0 is None:
        x_new = 0*b
    else:
        x_new = x0

    # r: residual
    # p: search direction
    r = b - A@x_new
    rho = nplg.norm(r)**2   
    p = np.copy(r)
    err = 2*tol
    n_iter = 0
    while err > tol:
        x = x_new
        w = A@p
        Anorm_p_squared = np.dot(p, w)
        
        # If norm_A(p) is 0, we should have converged.
        if Anorm_p_squared == 0:
            break

        alpha = rho/Anorm_p_squared
        x_new = x + alpha*p
        r -= alpha*w
        rho_prev = rho
        rho = nplg.norm(r)**2
        p = r + (rho/rho_prev)*p
        err = nplg.norm(x_new - x)/nplg.norm(x_new)
        n_iter += 1

    return x_new, n_iter

def test_solver(n=1000, N=3, method='lu', tol=1e-6):
    """Solves N nxn systems using {method} and compares with
    a direct solver (spsplg.solve()). """
    A = system_matrix(n)
    B = [np.random.rand(n) for _ in range(N)]

    if method == 'lu':
        lu = spsplg.splu(A)
        
    elif method == 'jacobi':
        Dinv, L_plus_U = jacobi_setup(A)

    for b in B:
        x_true = spsplg.spsolve(A, b)

        if method == 'lu':
            x = lu.solve(b)
        elif method == 'jacobi':
            x, n_iter = jacobi_solve(Dinv, L_plus_U, b, tol=tol)
        elif method == 'cg':
            x, n_iter = conjugate_gradient_solve(A, b, tol=tol)
        print(nplg.norm(x - x_true))
        if nplg.norm(x - x_true)/nplg.norm(x_true) > 10*tol:
            print(f'Error! {method} yields an error larger than {10*tol:.2e}.')


def run_and_time(n=10000, N=100, methods=['gauss','lu','jacobi'], tol=1e-6):
    """Solves N nxn systems using the listed methods and prints the execution time"""
    A = system_matrix(n)
    B = [np.random.rand(n) for _ in range(N)]
    iterative_methods = ['jacobi', 'cg']

    for method in methods:

        if method in iterative_methods:
            n_iter_tot = 0
        else: 
            n_iter_tot = 'N/A'

        if method == 'lu':
            lu = spsplg.splu(A)
        elif method == 'jacobi':
            Dinv, L_plus_U = jacobi_setup(A)

        start_time = time.time()
        for b in B:
            if method == 'gauss':
                x = spsplg.spsolve(A, b)
            if method == 'lu':
                x = lu.solve(b)
            elif method == 'jacobi':
                x, n_iter = jacobi_solve(Dinv, L_plus_U, b, tol=tol)
            elif method == 'cg':
                x, n_iter = conjugate_gradient_solve(A, b, tol=tol)

            if method in iterative_methods:
                n_iter_tot += n_iter

        t = time.time() - start_time
        print(f'{method}: Time: {t:.2e} s. Total iterations: {n_iter_tot}')





def main():
    run_and_time(methods = ['cg', 'gauss', 'lu', 'jacobi'])


if __name__ == "__main__":
    main()