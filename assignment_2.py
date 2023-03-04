import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt
import math
import rungekutta4 as rk4
from scipy.sparse.linalg import inv
from time import time 


def my_mass_matrix_assembler(x):
    N = len(x) - 1 
    M = spsp.dok_matrix((N+1, N+1)) # initialize stiffness matrix
    # M = np.zeros((N+1,N+1))
    
    for i in range(N):              # loop over elements
        h = x[i+1] - x[i]           # element length
        M[i, i] += 2*h/6              # assemble element stiffness
        M[i, i+1] += h/6
        M[i+1, i] += h/6
        M[i+1, i+1] += 2*h/6

    M[0, 0] = 1E6                 # adjust for BC
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
    A[0, 0] = 1E6                 # adjust for BC
    A[N, N] = 1E6
    # A[0, 0] = 1                 # adjust for BC
    # A[N, N] =1
    return A.tocsr()

k = 100*2*np.pi
xl = 0                                 # left end point of interval
xr = 1 
a = 1
T = 10**(-5)

def run_sim(show_animation=False):
    t = 0
    N = 2000                       # number of intervals
    
    h = (xr-xl)/(N-1)                           # mesh size
    x = np.arange(xl, xr, h)                # node coords
    A = my_stiffness_matrix_assembler(x)
    M = my_mass_matrix_assembler(x)
    # equation is M*xi_t = -A*xi
    # Minv = np.linalg.inv(M)
    Minv = splg.inv(M)
    MA = -Minv@A

    def rhs(u):
        
        res = MA@u
        # structure: u_t = -M^-1*A*u
        return res

    # error = np.linalg.norm(exact_prime,2) - xi.T@A@xi
    ht_try = 0.1*math.sqrt(a)*h**2
    mt = int(np.ceil(T/ht_try)+1) # round up so that (mt-1)*ht = T
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

    for tidx in range(mt-1):
        xi, t = rk4.step(rhs, xi, t, ht)
        
        if tidx % 25 == 0 and show_animation: 
            line.set_ydata(xi)
            title.set_text(f't = {t*1E6:.2f} microseconds')
            plt.draw()
            plt.pause(1e-8)
    return xi,x


def plot_exact(xi,x):
    fig, ax = plt.subplots()
    [line] = ax.plot(x, xi, label='Approximation')
    ax.set_xlim([xl, xr])
    ax.set_ylim([np.min(xi)*2, np.max(xi)*2])
    title = plt.title(f't = {T*1E6:.2f} microseconds')
    exact = np.sin(k*x)*np.exp(-a*k**2*T)
    plt.plot(x,xi,'r')
    plt.plot(x,exact,'g--')
    plt.show()
    

def main():
    tstart = time()
    xi,x = run_sim(show_animation=True)
    print(f"runtime: {time()-tstart}s")
    plot_exact(xi,x)

if __name__ == '__main__':
    main()  