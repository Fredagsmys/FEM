import numpy as np
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
import matplotlib.pyplot as plt
import math


def my_mass_matrix_assembler(x):
    N = len(x) - 1 
    M = spsp.dok_matrix((N+1, N+1)) # initialize stiffness matrix
    for i in range(N):              # loop over elements
        h = x[i+1] - x[i]           # element length
        M[i, i] += 2*h/6              # assemble element stiffness
        M[i, i+1] += h
        M[i+1, i] += h
        M[i+1, i+1] += 2*h/6


def my_stiffness_matrix_assembler(x):
    N = len(x) - 1                  # number of elements
    A = spsp.dok_matrix((N+1, N+1)) # initialize stiffness matrix
    for i in range(N):              # loop over elements
        h = x[i+1] - x[i]           # element length
        A[i, i] += 1/h              # assemble element stiffness
        A[i, i+1] += -1/h
        A[i+1, i] += -1/h
        A[i+1, i+1] += 1/h
    # A[0, 0] = 10E6                 # adjust for BC
    # A[N, N] =10E6
    A[0, 0] = 1                 # adjust for BC
    A[N, N] =1
    return A.tocsr()

def main():
    a = 0                                 # left end point of interval
    b = 1                                 # right
    N = 1000                       # number of intervals
    T = 10**(-5)
    k = 100*2*np.pi
    h = (b-a)/N                           # mesh size
    x = np.arange(a, b, h)                # node coords
    A = my_stiffness_matrix_assembler(x)
    M = my_mass_matrix_assembler(x)
    # error = np.linalg.norm(exact_prime,2) - xi.T@A@xi

if __name__ == '__main__':
    main()  