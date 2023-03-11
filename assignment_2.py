import numpy as np
import scipy.sparse as spsp
import matplotlib.pyplot as plt
import math
import rungekutta4 as rk4
from time import time 
from assignment_1 import jacobi_setup,jacobi_solve,conjugate_gradient_solve
import scipy.sparse.linalg as spsplg
import numpy.linalg as nplg


def my_mass_matrix_assembler(x):
    N = len(x) - 1 
    M = spsp.dok_matrix((N+1, N+1)) # initialize stiffness matrix
    
    for i in range(N):              
        h = x[i+1] - x[i]           
        M[i, i] += 2*h/6              
        M[i, i+1] += h/6
        M[i+1, i] += h/6
        M[i+1, i+1] += 2*h/6

    M[0, 0] = 1E6                # adjust for BC
    M[N, N] = 1E6
    return M.tocsr()


def my_stiffness_matrix_assembler(x):
    N = len(x) - 1                  # number of elements
    A = spsp.dok_matrix((N+1, N+1)) # initialize stiffness matrix
    for i in range(N):              # loop over elements
        h = x[i+1] - x[i]           # element length
        A[i, i] += 1/h              # assemble element stiffness
        A[i, i+1] += -1/h
        A[i+1, i] += -1/h
        A[i+1, i+1] += 1/h
    return A.tocsr()

k = 100*2*np.pi
xl = 0                                 # left end point of interval
xr = 1 
a = 1
T = 10**(-5)

def run_sim(N=2000, show_animation=False, solver = "jacobi"):
    t = 0
    h = (xr-xl)/(N-1)                           # mesh size
    x = np.arange(xl, xr+h, h)                # node coords
    A = my_stiffness_matrix_assembler(x)
    M = my_mass_matrix_assembler(x)
    # equation is M*xi_t = -A*xi

    if solver == "jacobi":
        Dinv, LplusU = jacobi_setup(M)

    elif solver == "lu":
        lu = spsplg.splu(M)

    def rhs(xi):
        b = -A@xi
        b[0] = 0
        b[-1] = 0 

        guess = xi
        if solver == "jacobi":
            xi_t, iters = jacobi_solve(Dinv=Dinv, L_plus_U=LplusU, b=b, x0=guess)
        elif solver == "cg":
            xi_t, iters = conjugate_gradient_solve(A=M, b=b, x0=guess)
        elif solver == "lu":
            xi_t = lu.solve(b)
        elif solver == "analytical":
            xi_t = spsplg.spsolve(M, b)
        # structure: M*xi_t = A*xi
        return xi_t

    ht_try = 0.1*math.sqrt(a)*h**2
    mt = int(np.ceil(T/ht_try)+1)
    tvec, ht = np.linspace(0, T, mt, retstep=True)
    xi = np.sin(k*x)
    if show_animation:
        fig, ax = plt.subplots()
        [line] = ax.plot(x, xi, label='Approximation')
        ax.set_xlim([xl, xr])
        ax.set_ylim([-1, 1.2])
        title = plt.title(f't = {0:.2f} microseconds')
        plt.draw()
        plt.pause(1)
    tstart = time()
    for tidx in range(mt-1):
        xi, t = rk4.step(rhs, xi, t, ht)
        
        if tidx % 25 == 0 and show_animation: 
            line.set_ydata(xi)
            title.set_text(f't = {t*1E6:.2f} microseconds')
            plt.draw()
            plt.pause(1e-8)
    print(f"N = {N}. Runtime for solver: {solver}: {time()-tstart} s")
    return xi,x


def plot_exact(xi,x):
    fig, ax = plt.subplots()
    ax.set_xlim([xl, xr])
    ax.set_ylim([np.min(xi)*2, np.max(xi)*2])
    plt.title(f't = {T*1E6:.2f} microseconds')
    exact = np.sin(k*x)*np.exp(-a*k**2*T)
    print(exact-xi)
    plt.plot(x,xi,'r')
    plt.plot(x,exact,'g--')
    plt.show()
    

def compute_error(xi,u_exact):
    error = nplg.norm(xi - u_exact)/nplg.norm(u_exact)
    return error

def runAll():
    meshs = np.array([2000,3000,4000,6000,7000,8000])
    solvers = np.array(["cg", "lu"])
    errors = np.empty((solvers.shape[0],meshs.shape[0]))
    
    for i,solver in enumerate(solvers):
        for j,N in enumerate(meshs):
            xi,x = run_sim(N, solver=solver)
            u_exact = np.sin(k*x)*np.exp(-a*k**2*T)
            error = compute_error(xi,u_exact)
            errors[i][j] = error

    return meshs, errors, solvers


def plot_convergence(meshs, errors, solvers):
    colors = ['r','g']
    markers = ["o","x"]
    for i,solver in enumerate(solvers):
        plt.loglog(meshs, errors[i], label=solver, marker=markers[i], color=colors[i])

    plt.loglog(meshs,meshs**2)
    plt.grid(visible=True)
    plt.xlabel('mesh element size')
    plt.ylabel('l2 error')
    plt.legend()
    plt.savefig("convergence.png")
    plt.show()

def main():
    np.random.seed(1)

    
    # xi,x = run_sim(8000, show_animation=True, solver="lu")
    # plot_exact(xi,x)
    # u_exact = np.sin(k*x)*np.exp(-a*k**2*T)
    # error = compute_error(xi, u_exact)
    # print(error)
    meshs, errors, solvers = runAll()
    print(errors)
    plot_convergence(meshs, errors,solvers)

if __name__ == '__main__':
    main()  