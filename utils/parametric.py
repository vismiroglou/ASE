import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from signal_1 import periodogram

def least_squares(x, N1, N2, p, plot=False):
    def build_X():
        X = np.zeros((N2 - N1, p))
        for i, n in enumerate(range(N1, N2)):
            X[i, :] = x[n-1:n-p-1:-1]
        return X
    
    X =  build_X()
    pred_coeffs = - np.linalg.inv(X.T @ X) @ X.T @ x[N1:N2+1]

    # print('Predicted coefficients:', pred_coeffs)

    x_pred = np.zeros(N)
    for n in range(p, N):
        x_pred[n] = -np.dot(pred_coeffs, x[n-p:n][::-1])
    
    e = x - x_pred
    e_var = np.sqrt(np.var(e))
    # print('Noise variance: ', e_var)

    if plot:
        plt.figure()
        plt.plot(np.arange(N), x, label='real')
        plt.plot(np.arange(N), x_pred, label='predicted')
        plt.title('Real vs. predicted signal') 
        plt.legend()   
    
    return pred_coeffs, x_pred, e_var


def param_psd(x_pred, e_var, plot=False):
    X = np.fft.fft(x_pred)
    pxx = e_var**2 / (np.abs(X) ** 2)

    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(np.arange(-len(X)/2, len(X)/2), pxx, label="Parametric PSD")
        plt.xlabel("Frequency")
        plt.ylabel("PSD")
        plt.legend()
        plt.grid(True)
    
    return pxx
    

# Load signal.mat
x = scipy.io.loadmat('signal.mat')['x']
x = x.reshape(1000)
N = len(x)
p = 2 

# N1, N2  of the covariance method
N1 = p + 1
N2 = N
prediction, x_pred, e_var = least_squares(x, N1, N2, p, True)
plt.show()

param_psd(x_pred, e_var, True)
plt.show(block=False)
periodogram(x, N, plot=True)
plt.show()

e_var_list = []
for p in range(1, 50):
    N1 = p + 1
    N2 = N
    _, _, e_var = least_squares(x,N1, N2, p, False)
    e_var_list.append(e_var)

plt.plot(range(len(e_var_list)), e_var_list)
plt.xlabel('Model order')
plt.ylabel('Noise variance')
plt.show()