'''
- Implement the MUSIC method for estimating the PSD. Test it on the test
  signal from an earlier exercises.
- Show (with pen & paper) how the MUSIC method can be implemented with
  FFTs and modify your function to use it.
- Implement the method for model order selection using the  average of 
  the principal angles between subspaces  and try it on the test signal.
- Modify your MUSIC implement to make use of the signal subspace eigenvectors 
  instead of the noise subspace eigenvectors.
'''
from matplotlib import pyplot as plt
import numpy as np 

def generate_signal(N, fs, k, plot=False):
    np.random.seed(42)
    t = np.arange(N) / fs

    a = np.random.normal(0, 10, k)
    f = np.random.normal(0, fs/2, k)
    phi = np.random.uniform(-np.pi, np.pi, k)

    # Reshape for broadcasting: (k, 1) for a, f, phi; (1, N) for t
    a = a[:, np.newaxis]
    f = f[:, np.newaxis]
    phi = phi[:, np.newaxis]
    t = t[np.newaxis, :]
    x = np.sum(a * np.exp(1j * 2 * np.pi * f * t + 1j * phi), axis=0)

    # Add complex Gaussian noise
    noise_real = np.random.normal(0, 5, size=N)
    noise_imag = np.random.normal(0, 5, size=N)
    noise = noise_real + 1j * noise_imag
    
    x = x + noise

    if plot:
        plt.plot(range(N), x.real, label="Real part")
        plt.plot(range(N), x.imag, label="Imag part")
        # plt.plot(range(N), noise.real, label="Noise Real part")
        # plt.plot(range(N), noise.imag, label="Noise Imag part")
        plt.legend()
        plt.title(f"Complex Signal (N={N}, k={k})")
        plt.xlabel("Time [s]")
    
    return x, fs


def music(x, M, plot=False):
    # Get table R
    print(len(x))
    # R = (x[:, None] @ x.conj()[None, :]) / len(x)
    # eigenvalues, eigenvectors = np.linalg.eig(R)
    # print("Eigenvalues:", eigenvalues)
    # plt.hist(eigenvalues.real, bins=100)
    # plt.yscale('log')
    # plt.show()


if __name__ == '__main__':
    N = 1000
    fs = 8000
    k = 3
    x, fs = generate_signal(N, fs, k, plot=False)

    music(x, 4)