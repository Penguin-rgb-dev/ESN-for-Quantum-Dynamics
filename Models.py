#-------------------------------------------------------------------------------
# Name:        Fully connected transverse field Ising model (Density matrix formulation)
# Purpose:
#
# Author:      Divesh Mathur
#
# Created:     10/02/2025
# Copyright:   (c) Divesh Mathur 2025
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import random
rng = random.Random()
import numpy as np
import scipy as sp
from numpy import linalg, exp
from numpy.linalg import inv
from scipy.linalg import expm

#Hamiltonian matrix
#N - number of spins
#K - range of coupling constants
#T - time step size
#h - strength of the external magnetic field
#rng.seed(5)

def I(i) :  #Identity matrix of order 2^i
    I = np.identity(2**(i))
    return I

def J(N, K_min, K_max):    #Matrix of weights, symmetric; J in [-K,K]
    J = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            if not i==j:
                J[i][j] = J[j][i] = rng.random() * (K_max-K_min) + K_min
    return J

def X(N):   #Pauli X for the composite system
    x = np.array([[0,1],[1,0]])
    X = ()
    for i in range(N):
        X = X + (np.kron(np.kron(I(i),x),I(N-i-1)),)

    X = np.array(X)
    return X

def Y(N):   #Pauli X for the composite system
    y = np.array([[0,-1j],[1j,0]])
    Y = np.zeros(N, dtype = 'object')
    for i in range(N):
        Y[i] = np.kron(np.kron(I(i),y),I(N-i-1))

    return Y

def Z(N):   #Pauli Z for the composite system
    z = np.array([[1,0], [0,-1]])
    Z = ()
    for i in range(N):
        Z = Z + (np.kron(np.kron(I(i),z),I(N-i-1)),)
    Z = np.array(Z)
    return Z

def ZZ(N):
    z = np.array([[1,0],[0,-1]])
    ZZ = []
    for i in range(N):
        for j in range(i+1,N):
            part_1 = np.kron(I(i),z)
            part_1 = np.kron(part_1,I(j-i-1))
            part_2 = np.kron(z,I(N-j-1))
            ZZ.append(np.kron(part_1,part_2))
    return ZZ

def Heisenberg_NN(N,K,h):   #Weights are in the range (0,K); mag. field = h
    x = X(N)
    y = Y(N)
    z = Z(N)
    W = J(N,0,K)
    H = np.zeros([2**N,2**N])
    for i in range(N-1):
        H = H - W[i,i+1]*(x[i]@x[i+1] + y[i]@y[i+1] + z[i]@z[i+1])
    for i in range(N):
        H = H - h * z[i]
    return H, W


def Ferromagnetic_Heisenberg(N,K,h):   #Weights are in the range (0,K); mag. field = h
    x = X(N)
    y = Y(N)
    z = Z(N)
    W = J(N,0,K)

    H = np.zeros([2**N,2**N])
    for i in range(N):
        for j in range (N):
            if j > i:
                H = H - W[i][j]*(x[i]@x[j] + y[i]@y[j] + z[i]@z[j])
            else:
                continue

    for i in range(N):
        H = H - h*z[i]

    return H, W

def Anti_Ferromagnetic_Heisenberg(N,K,h):   #Weights are in the range (0,K); mag. field = h
    x = X(N)
    y = Y(N)
    z = Z(N)
    W = J(N,0,K)

    H = np.zeros([2**N,2**N])
    for i in range(N):
        for j in range (N):
            if j > i:
                H = H + W[i][j]*(x[i]@x[j] + y[i]@y[j] + z[i]@z[j])
            else:
                continue

    for i in range(N):
        H = H - h*z[i]

    return H, W

def Mixed_Heisenberg(N,K,h):   #Weights are in the range (0,K); mag. field = h
    x = X(N)
    y = Y(N)
    z = Z(N)
    W = J(N,-K/2,K/2)

    H = np.zeros([2**N,2**N])
    for i in range(N):
        for j in range (N):
            if j > i:
                H = H - W[i][j]*(x[i]@x[j] + y[i]@y[j] + z[i]@z[j])
            else:
                continue

    for i in range(N):
        H = H - h*z[i]

    return H, W


def Ising_1DNN(N,K,h):  #Weights are in the range (-K/2,K/2)
    x = X(N)
    z = Z(N)
    W = J(N,-K/2,K/2)
    H = np.zeros([2**N,2**N])
    for i in range(N - 1):
        H = H + W[i][i + 1] * x[i] @ x[i + 1]
    for i in range(N):
        H = H + h * z[i]
    return H, W

def Ising(N,K,h):   #Weights are in the range (-K/2,K/2)
    x = X(N)
    z = Z(N)
    W = J(N,-K/2,K/2)
    H = np.zeros([2**N,2**N])
    for i in range(N):
        for j in range (N):
            if j > i:
                H = H + W[i][j]*x[i]@x[j]
            else:
                continue

    for i in range(N):
        H = H + h*z[i]

    return H, W


def Time_evolution_operator(H, time_step):
    U = expm(-1.j*H*time_step)
    return U

def time_evolution(rho,H,T):
    U = Time_evolution_operator(H, 1/10)
    P = (rho,)
    for i in range(10*T):
        P = P + (U @ P[i] @ U.T.conj(),)

    return P

def getstate():
    state = rng.getstate()
    return state

def setstate(x):
    rng.setstate(x)

def ran():
    for i in range(5):
        print(rng.random())

print('Module Hamiltonian Loaded!')



