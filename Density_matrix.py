#-------------------------------------------------------------------------------
# Name:        Density matrix
# Purpose:
#
# Author:      Divesh Mathur
#
# Created:     08/02/2025
# Copyright:   (c) Divesh Mathur 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import scipy as sp
import random
from numpy import array

rng = random.Random()
#rng.seed(5)

def vector(m): #Generate a random vector of dimension 'm' for single particle
    coeff = ()
    for i in range(m):
        coeff = coeff + (rng.random(),)

    coeff = array(coeff)
    coeff = coeff/np.sqrt(np.sum(coeff**2))

    return coeff

def composite_state(m, N):  # Generate a composite vector state for N particles
    vectors = ()
    for i in range(N):
        vectors = vectors + (vector(m),)

    state = np.kron(vectors[0],vectors[1])
    for i in range(2,N):
        state  = np.kron(state,vectors[i])

    return state

def pure_density_matrix(m,N):
    psi = composite_state(m,N)
    rho = np.outer(psi,psi)
    return rho

def weight_dist(n):
    weights = ()
    for i in range(n):
        weights = weights + (rng.random(),)

    weights = array(weights)
    weights = weights/np.sum(weights)

    return weights

def mixed_density_matrix(n,m,N): #n = number of states in the ensemble; N = number of spins in each system; m = dimension of single spin state space
    rho = np.zeros([m**N,m**N])
    weights = weight_dist(n)
    for i in range(n):
        rho = rho + weights[i]*pure_density_matrix(m, N)

    return rho

def I(N):
    I = np.identity(2**N)
    return I

def a_b(N): 
    a = np.kron([[1],[0]], I(N-1))  #0 or up spin
    b = np.kron([[0],[1]],I(N-1))   #1 or down spin
    return a,b

def trace_1(rho,N):
    a = a_b(N)[0]
    b = a_b(N)[1]
    tr = (a.T @ rho @ a) + (b.T @ rho @ b)
    return tr

def TRACE_1(A): #A is the matrix; Different way to evaluate the partial trace (Slightly slower than before)
    N = int(np.log2(len(A)))
    a = 2**(N-1)
    A_rest = np.zeros([a,a])
    for i in range(a):
        for j in range(a):
            A_rest[i,j] = A[i,j] + A[a+i,a+j]
    return A_rest

def getstate():
    state = rng.getstate()
    return state

def setstate(x):
    rng.setstate(x)

def basis(N):   #It gives the natural spin z eigenvector basis in which density matrices are first written
    basis = ()
    for i in range(2**N):
        state = np.zeros(2**N)
        for j in range(len(state)):
            if j == i:
                state[j] = 1
        basis = basis + (state,)
    return basis


#rho = pure_density_matrix(2,4)
#print(rho)
#tr = trace_1(rho,4)
#print(tr)

#print(np.trace(np.square(mixed_density_matrix(10, 3, 3))))
#print(np.trace(pure_density_matrix(3,3)**2))
#print(vector(3))

#rho = pure_density_matrix(2,2)
#print(np.trace(np.dot(rho,rho)))
#print(np.trace(rho))
#print(np.trace(mixed_density_matrix(10,2,2)))
#print(np.trace(np.dot(mixed_density_matrix(10,2,2), mixed_density_matrix(10,2,2))))

print("Module Density Matrix has been loaded!")