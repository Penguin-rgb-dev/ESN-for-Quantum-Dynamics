## For one set of parameter values.
## learning and predicting 100 different time-series, one at a time.
## Defining the network and the update rule
import numpy as np
from numpy.linalg import eigvals
from numpy import typing as npt
from sklearn.linear_model import LinearRegression
from typing import Annotated, Literal
import time

start_time = time.perf_counter()
N = 400 #internal units
K = 1   #input units
L = 1   #output units
lambda_max = 0.6
time_constant = 0.1 
decay_constant = 0.6

#Weights. N internal units. 1 input unit. 1 output unit.

W = np.zeros([N,N]) #internal weight matrix
for i in range(N):
    for j in range(N):
        a = np.random.uniform()
        if a < 0.9875: W[i,j]=0
        elif 0.9875<a<0.99375: W[i,j]=-0.4
        else: W[i,j]=0.4

W_in = np.zeros([N,K])  #input connections
for i in range(N):
    for j in range(K):
        a = np.random.uniform()
        if a<0.5: W_in[i,j]=0
        elif a==0.5 or 0.5<a<0.75: W_in[i,j]=0.14
        else: W_in[i,j]=-0.14

W_back = np.zeros([N,K])
for i in range(N):
    for j in range(K):
        W_back[i,j] = np.random.uniform(-0.56,0.56)

W_back = 0.1*W_back
W_in = 0.1*W_in

eigval = eigvals(W)
W = (lambda_max/max(abs(eigval)))*W
eigval = eigvals(W)
print('|lambda_max|(W) =',max(abs(eigval)))
print('|lambda_max|(tilde_W) =', (0.44*lambda_max)+(1-(0.44*0.9)))



# Update rule
def update(x:Annotated[npt.NDArray[np.float64], Literal["N","M"]],u:npt.NDArray[np.float64],y:npt.NDArray[np.float64],delta=1,C=time_constant,a=decay_constant) -> npt.NDArray[np.float64]:
    x = (1-delta*C*a)*x + delta*C*(np.tanh(W_in @ u + W @ x + W_back @ y))
    return x 

def update_training(x:Annotated[npt.NDArray[np.float64], Literal["N","M"]],u:npt.NDArray[np.float64],y:npt.NDArray[np.float64],delta=1,C=time_constant,a=decay_constant) -> npt.NDArray[np.float64]:
    v = np.random.uniform(-1e-5,1e-5,np.shape(x))
    x = (1-delta*C*a)*x + delta*C*(np.tanh(W_in @ u + W @ x + W_back @ y + v))
    return x 

washout = 2000
train = 20000
test = 10000
MSE_TRAIN = []
MSE_TEST = []
SEEDS = np.random.randint(0,1000,size=100)
np.save('New/Results/Seeds.npy', SEEDS)

for Y in range(100):
    np.random.seed(SEEDS[Y])
    #np.random.seed(42)
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
        x = update_training(x,u.reshape(-1,1),y_washout[i].reshape(-1,1))
        # Now we are at x(washout+1)

    #Training
    X = np.zeros([train,N])  #Collects x(washout+2) to x(train)
    X[0,:] = x[:,0]
    for i in range(0,train-1):   #total train-1 steps
        x = update_training(x,u.reshape(-1,1),y_train[i].reshape(-1,1))
        X[i+1,:] = x[:,0]

    
    model = LinearRegression(fit_intercept=False)
    model.fit(X,y_train)
    print(f'Training for y_{Y+1} is done!')

    ## Testing
    y_pred = np.zeros(test)
    x = update(x,u.reshape(-1,1),y_train[train-1].reshape(-1,1))
    y_pred[0] = model.predict(x.reshape(1,-1))[0][0]
    for i in range(test-1):
        x = update(x,u.reshape(-1,1),y_pred[i].reshape(-1,1))
        y_pred[i+1] = model.predict(x.reshape(1,-1))[0][0]
    np.save(f'New/Results/y_pred_{Y+1}.npy',y_pred)

    ## mse
    y_fit = model.predict(X)
    #mse_train = np.sum((y_train-y_fit)**2)/np.sum(y_train**2)
    mse_test = np.sum((y_test - y_pred)**2)/np.sum(y_test**2)

    #MSE_TRAIN.append(mse_train)
    MSE_TEST.append(mse_test)


#np.save('New/Results/MSE_TRAIN.npy', np.array(MSE_TRAIN))
np.save('New/Results/MSE_TEST.npy', np.array(MSE_TEST))   
end_time = time.perf_counter()
print(f'{end_time-start_time:.4f}')
#with open('New/Results/output','a') as f:
#    print(f'Time elapsed: {end_time-start_time:.4f} seconds',file=f)

