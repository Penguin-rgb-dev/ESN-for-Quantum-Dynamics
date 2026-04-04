## NARMA Task for delay=10, scanning over values of h.

import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression
from Models import X, Y, Z, ZZ, Ising
from Density_matrix import trace_1, mixed_density_matrix
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(seed=42)

## Defining input function
def inpt(rho, s, N):
    a = np.array([[0],[1]])
    b = np.array([[1],[0]])
    psi_s = np.sqrt(1-s)*a + np.sqrt(s)*b
    rho_s = np.outer(psi_s, psi_s)
    rho_rest = trace_1(rho,N)
    density_matrix = np.kron(rho_s, rho_rest)
    return density_matrix

## Dataset (linear memory y_(n) = s_(n-delay))
n=10    #delay or order of NARMA
washout = 1000
train = 2000
test = 2000
# input
s = rng.uniform(0.0,0.2,5100)
#output
y = np.zeros(5100)
for i in range(11,5100):
    y[i] = 0.1 + 1.5*s[i-n]*s[i-1] + 0.05*y[i-1]*np.sum(y[i-n:i]) + 0.3*y[i-1]

s = s[100:]/0.2 # Rescaling input to be in [0,1] for inputting into the quantum reservoir
y = y[100:]
s_washout = s[:washout]
s_train = s[washout:washout+train]
s_test = s[washout+train:]
y_washout = y[:washout]
y_train = y[washout:washout+train]
y_test = y[washout+train:]

# Loop over parameters
N=10
J=1
tau=10*J
Cov_mean = []
Cov_std = []
H = np.logspace(-2,2,num=20)
for h in H:
    Cov = []    # Cov of all 100 realizations
    for l in range(100):    #Looping over 100 different realizations of the hamiltonian 
        Hamiltonian, Jij = Ising(N, J, h)
        rho = mixed_density_matrix(10,2,N)  # Initial mixed state
        ## Diagonalizing the Hamiltonian
        E, U = eigh(Hamiltonian)
        energy_diffs = E[:, np.newaxis] - E[np.newaxis, :]
        phase_factors = np.exp(-1j * energy_diffs * tau)
        
        ## Defining Time evolution via ED
        def time_evolve(rho_0): #time evolve t=1 
            rho_energy = U.conj().T @ rho_0 @ U
            rho_energy_t = rho_energy * phase_factors
            rho_t = U @ rho_energy_t @ U.conj().T
            return rho_t

        ## Defing the list of observables
        x = X(N)
        y = Y(N)
        z = Z(N)
        zz = ZZ(N)  #list of all pairs
        obs  = list(x) + list(y) + list(z) + zz

        ## Washout, training, and testing
        # Washout
        for s in s_washout:
            rho = inpt(rho,s,N)
            rho = time_evolve(rho)
        # Training
        X = np.zeros([train,len(obs)])
        for k in range(len(s_train)):
            rho = inpt(rho,s_train[k],N)
            rho = time_evolve(rho)
            for j in range(len(obs)):
                X[k,j] = np.real(np.trace(rho@obs[j]))
        model = LinearRegression()
        model.fit(X,y_train)
        #Testing
        y_pred = []
        for s in s_test:
            rho = inpt(rho,s,N)
            rho = time_evolve(rho)
            x = []
            for i in range(len(obs)):
                x.append(np.real(np.trace(rho@obs[i])),)
            x = np.array(x)
            x = x[np.newaxis,:]
            y_pred.append(model.predict(x))

        y_pred = np.array(y_pred)
        y_pred = y_pred[:,0]

        cov = np.cov(y_test,y_pred)
        C = (cov[0,1]**2)/(cov[0,0]*cov[1,1])
        Cov.append(C)

    Cov = np.array(Cov)
    Cov_mean.append(np.mean(Cov))
    Cov_std.append(np.sqrt(np.var(Cov)))
H = np.array(H)
np.save('Cov_mean.npy',Cov_mean)
np.save('Cov_std.npy',Cov_std)
np.save('H.npy',H)