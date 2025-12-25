## 'for' loop on parameters
## Hyper-parameter optimization (|lambda|_max, C, and a).
## Defining the network and the update rule
import numpy as np
from numpy.linalg import eigvals
from numpy import typing as npt
from sklearn.linear_model import LinearRegression
from typing import Annotated, Literal
import time



start_time = time.perf_counter()
# Parameters
N = 400 #internal units
K = 1   #input units
L = 1   #output units


#Weights. N internal units. 1 input unit. 1 output unit.

W_original = np.zeros([N,N]) #internal weight matrix
for i in range(N):
    for j in range(N):
        a = np.random.uniform()
        if a < 0.9875: W_original[i,j]=0
        elif 0.9875<a<0.99375: W_original[i,j]=-0.4
        else: W_original[i,j]=0.4

eigval = eigvals(W_original)
max_eig = max(abs(eigval))
if max_eig == 0:
    raise RuntimeError("Maximum eigenvalue of W_original is zero; cannot rescale.")

W_in_original = np.zeros([N,K])  #input connections
for i in range(N):
    for j in range(K):
        a = np.random.uniform()
        if a<0.5: W_in_original[i,j]=0
        elif a==0.5 or 0.5<a<0.75: W_in_original[i,j]=0.14
        else: W_in_original[i,j]=-0.14

W_back_original = np.zeros([N,K])
for i in range(N):
    for j in range(K):
        W_back_original[i,j] = np.random.uniform(-0.56,0.56)

## Rescaling
factor=0.1
W_back = factor*W_back_original.copy()
W_in = factor*W_in_original.copy()


# Update rule
def update(x:Annotated[npt.NDArray[np.float64], Literal["N","M"]],u:npt.NDArray[np.float64],y:npt.NDArray[np.float64],C,a, W, W_back, W_in, delta=1) -> npt.NDArray[np.float64]:
    x = (1-delta*C*a)*x + delta*C*(np.tanh(W_in @ u + W @ x + W_back @ y))
    return x 

def update_training(x:Annotated[npt.NDArray[np.float64], Literal["N","M"]],u:npt.NDArray[np.float64],y:npt.NDArray[np.float64],C,a, W, W_back, W_in, delta=1) -> npt.NDArray[np.float64]:
    v = np.random.uniform(-1e-5,1e-5,np.shape(x))
    x = (1-delta*C*a)*x + delta*C*(np.tanh(W_in @ u + W @ x + W_back @ y + v))
    return x 

#Parameters
washout = 1000
train = 9000
test = 2000

FACTORS = np.linspace(0.1,1,10)

tasks = [(i,j,k) for i in FACTORS for j in FACTORS for k in FACTORS]
RESULTS = []

np.random.seed(42)  #reproducible random noise
FLUSH=0
with open(f'output_{factor}.txt','w') as f:
    for lambda_max,C,a in tasks:
        FLUSH += 1
        ## Rescaling the internal weights matrix
        
        W = (lambda_max/max(abs(eigval)))*W_original.copy()

        MSE_TEST = []
        for Y in range(100):
            ## Training and testing data
            y = np.load(f'New/Data_2/y_{Y+1}.npy')[:washout+train+test]
            y_washout = y[:washout]
            y_train = y[washout:washout+train]
            y_test = y[washout+train:]

            ## Algorithm
            x = np.random.uniform(-1,1,[N,1])   # randomly chosen initial state
            y_train = np.reshape(y_train,(len(y_train),-1)) # Reshaping y_train into a column vector of shape (len(y_train), 1)
            u = np.array([[0.2]])   # Setting a constatnt input bias to increase the variability of the internal signals

            #Washout
            for i in range(washout):
                x = update_training(x,u.reshape(-1,1),y_washout[i].reshape(-1,1),C,a, W, W_back, W_in)
                # Now we are at x(washout+1)

            #Training
            X = np.zeros([train,N])  #Collects x(washout+2) to x(train)
            X[0,:] = x[:,0]
            for i in range(0,train-1):   #total train-1 steps
                x = update_training(x,u.reshape(-1,1),y_train[i].reshape(-1,1),C,a, W, W_back, W_in)
                X[i+1,:] = x[:,0]

            
            model = LinearRegression(fit_intercept=False)
            model.fit(X,y_train)
            #print(f'Training for y_{Y+1} is done!')

            ## Testing
            y_pred = np.zeros(test)
            x = update(x,u.reshape(-1,1),y_train[train-1].reshape(-1,1),C,a, W, W_back, W_in)
            y_pred[0] = model.predict(x.reshape(1,-1))[0][0]
            for i in range(test-1):
                x = update(x,u.reshape(-1,1),y_pred[i].reshape(-1,1),C,a, W, W_back, W_in)
                y_pred[i+1] = model.predict(x.reshape(1,-1))[0][0]

            ## Normalized Mean Squared Error
            y_fit = model.predict(X)
            mse_test = np.sum((y_test - y_pred)**2)/np.sum(y_test**2)
            MSE_TEST.append(mse_test)

        ## Analysis of results(MSE_TEST)
        minimum_mse = np.min(MSE_TEST)
        N1 = {} #values less than 1
        for l in range(len(MSE_TEST)):
            if MSE_TEST[l] < 1:
                N1[f'{l+1}'] = MSE_TEST[l]
        N2 = {} #values less than 0.1
        for l in range(len(MSE_TEST)):
            if MSE_TEST[l] < 0.1:
                N2[f'{l+1}'] = MSE_TEST[l]
        n1 = len(N1)
        n2 = len(N2)
        RESULTS.append([round(lambda_max,2), round(C,2),round(a,2),n1,n2,minimum_mse])
        flush_freq = 100
        if FLUSH%flush_freq==0:
            for I in range(flush_freq):
                print(np.array(RESULTS[-(flush_freq-I)]),file=f,flush=True)

np.save(f'New/ESN_PARAMETER_SEARCH_{factor}.npy', np.array(RESULTS))

end_time = time.perf_counter()
print(end_time - start_time)