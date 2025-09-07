'''
Day 1
    - Make a script for generating a test signal of length N for use in later exercises. 
      The test signal should contain two sinusoids with amplitudes 2 and 4 and  frequencies 
      500 and 2500 Hz, respectively, for a sampling frequency of 8 kHz, and WGN noise of variance 1.
    - Implement the periodogram and correlogram spectral estimators and compare their PSD estimates 
      using the above test signal for different number of samples N.
    - Show (with pen and paper) that a Fourier transform of a signal of length N can be broken into
      k smaller Fourier transforms of length N/k.
    - Implement (in MATLAB/Python) the Bartlett and Welch methods.
    - Implement a function for computing the down-sampled discrete-time analytic signal.
    - Implement, in a function, the least squares method for AR parameter estimation. The starting 
      and end points should be selectable.
    - Load the signal x in signal.mat and find the AR(2) parameters using the implemented least-squares
      method with starting and end points corresponding to the covariance method. Find also the variance 
      of the noise.
    - Use the parameters of the AR model to estimate the PSD of the signal and compare the so-obtained 
      estimate to those of the correlogram and periodogram methods. What do you observe? 
    - Plot the noise variance estimate as a function of the model order using the implemented estimator. 
      How can the model order be estimated?
'''

import numpy as np
from matplotlib import pyplot as plt

def generate_signal(N, plot=False):
    fs = 8000
    A1, f1 = 2, 500
    A2, f2 = 4, 2500
    t = np.arange(N) / fs   # time vector in seconds
     
    # Complex exponentials (e^{j2πft})
    x1 = A1 * np.exp(1j * 2 * np.pi * f1 * t)
    x2 = A2 * np.exp(1j * 2 * np.pi * f2 * t)

    # Combined complex signal
    x = x1 + x2

    # Add complex Gaussian noise
    np.random.seed(42)
    noise = (np.random.normal(0, 1, N) +
            1j * np.random.normal(0, 1, N)) / np.sqrt(2)  # proper complex noise
    x = x + noise

    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(t, x.real, label="Real part")
        plt.plot(t, x.imag, label="Imag part")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title("Complex Signal (time domain)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'Signal_N_{N}.png')

    return x, fs


def periodogram(x, fs, plot=False):
    N = len(x)
    freqs = np.linspace(-fs/2, fs/2, N, endpoint=False)
    t = np.arange(N)/fs

    # DFT manually 
    X = np.zeros(N, dtype=complex)
    for k, f in enumerate(freqs):
        X[k] = np.sum(x * np.exp(-1j * 2 * np.pi * f * t))

    # Periodogram
    Pxx = (1/N) * np.abs(X)**2

    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(freqs, Pxx.real, label="real")
        plt.plot(freqs, Pxx.imag, label="imaginary")
        plt.xlabel("Frequency")
        plt.ylabel("PSD")
        plt.title(f"Periodogram, N: {N}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'Periodogram_N_{N}.png')
    return Pxx


def correlogram(x, fs, plot=False):
    def calc_ace():
        ace = []
        for k in range(N):
            ace.append(np.sum(x[k+1:N] * np.conj(x[np.arange(k+1,N)- k])) / (N - k))
        ace = np.array(ace)
        return ace

    N = len(x)
    
    t = np.arange(N)/fs
    freqs = np.linspace(-fs/2, fs/2, N, endpoint=False)

    X = np.zeros(len(t), dtype=complex)
    ace = calc_ace()
    for i, f in enumerate(freqs):
        X[i] = np.sum(ace * np.exp(-1j * 2 * np.pi * f * t))

    # Correlogram
    Pxx = X

    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(freqs, Pxx.real, label="real")
        plt.plot(freqs, Pxx.imag, label="imaginary")
        plt.xlabel("Frequency")
        plt.ylabel("PSD")
        plt.title(f"Correlogram, N: {N}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'Correlogram_N_{N}.png')
    return Pxx


def bartlett(x, fs, L, plot=False):
    """
    Bartlett method for PSD estimation.
    x: input signal
    fs: sampling frequency
    L: number of segments
    plot: whether to plot the PSD
    """
    N = len(x)
    M = N // L  # Length of each segment
    x = x[:L * M]  # Truncate to fit segments
    segments = x.reshape(L, M)

    pxx = 0
    for seg in segments:
        pxx += periodogram(seg, fs)
    pxx /= L

    freqs = np.linspace(-fs/2, fs/2, M, endpoint=False)
    
    if plot:

        # plt.figure(figsize=(10,4))
        plt.plot(freqs, pxx, label=f"Bartlett PSD | L={L}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD")
        plt.legend()
        plt.grid(True)

    return pxx

def welch(x, fs, L, overlap=0.5, window='hanning', plot=False):
    """
    Welch method for PSD estimation.
    x: input signal
    fs: sampling frequency
    L: number of segments
    overlap: fraction of overlap between segments (0 to <1)
    plot: whether to plot the PSD
    """
    N = len(x)
    M = N // L  # Length of each segment
    step = int(M * (1 - overlap))
    if step < 1:
        raise ValueError("Overlap too high, step size < 1")
    segments = []
    for start in range(0, N - M + 1, step):
        segments.append(x[start:start+M])
    segments = np.array(segments)
    if window == 'hanning':
        window = np.hanning(M)

    pxx = 0
    for seg in segments:
        seg_win = seg * window
        pxx += periodogram(seg_win, fs)
    pxx /= len(segments)
    freqs = np.linspace(-fs/2, fs/2, M, endpoint=False)
    if plot:
        # plt.figure(figsize=(10,4))
        plt.plot(freqs, pxx, linestyle='--', label=f"Welch PSD | L={L}")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD")
        plt.legend()
        plt.grid(True)
    return pxx

def ds_analytic(x, fs, plot=False):
    x_r = x.real
    N = len(x_r)

    X = np.fft.fft(x_r)

    H = np.zeros(N)
    if N % 2 == 0:  # even length
        H[0] = 1
        H[1:N//2] = 2
        H[N//2] = 1
    else:           # odd length
        H[0] = 1
        H[1:(N+1)//2] = 2
    
    X_a = X * H
    X_a = X_a[:np.ceil(N/2).astype(int)]
    x_a = np.fft.ifft(X_a)

    plt.plot(np.arange(N)/fs, x_r, label="real signal")
    plt.plot(np.arange(len(x_a))/fs * 2, x_a, label="analytical")
    plt.plot(np.arange(len(x_a))/fs * 2, x_a.imag, label="analytical_imag")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.legend()
    plt.grid(True)

  
if __name__ == '__main__':
    x, fs = generate_signal(100,False)
    # plt.show()
    # ds_analytic(x, fs, True)
    # pxx = periodogram(x, fs, True)
    # pxx = correlogram(x, fs, True)
    # plt.figure(figsize=(10,4))
    # for L in (5, 10, 50):
    #     pxx = bartlett(x, fs, L, True)
    #     pxx = welch(x, fs, L, 0.5, 'hanning', True)
    # plt.savefig('barlet_welch_1000.png')
    plt.figure(figsize=(10,4))
    ds_analytic(x, fs, plot=True)
    plt.savefig('discrete_analytic.png')
    plt.show()