import numpy as np
from matplotlib import pyplot as plt

def generate_signal(N, plot=False):
    fs = 8000
    A1, f1 = 2, 500
    A2, f2 = 4, 2500
    # Time vector
    t = np.arange(N) / fs   # time indices in seconds

    # Generate sinusoids
    x1 = A1 * np.sin(2 * np.pi * f1 * t)
    x2 = A2 * np.sin(2 * np.pi * f2 * t)

    # Combined signal
    x = x1 + x2

    np.random.seed(42)
    noise = np.random.normal(0, 1, N)
    x = x + noise

    if plot:
        # Plot
        plt.figure(figsize=(10,4))
        plt.plot(t, x, label="Combined Signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return x, fs

def periodogram(x, plot=False):
    N = len(x)
    freqs = np.linspace(0, fs/2, N, endpoint=False)
    t = np.arange(N) / fs

    # DFT manually 
    X = np.zeros(N, dtype=complex)
    for k, f in enumerate(freqs):
        X[k] = np.sum(x * np.exp(-1j*2*np.pi*f*t))

    # Periodogram
    Pxx = (1/N) * np.abs(X)**2

    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(freqs, Pxx, label="Periodogram")
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude?")
        plt.legend()
        plt.grid(True)
        plt.show()

    return Pxx

def correlogram(x, plot=False):
    N = len(x)
    # --- Step 1: Estimate autocorrelation ---
    lags = np.arange(-(N-1), N)   # from -(N-1) to (N-1)
    rxx = np.zeros(len(lags))

    for i, m in enumerate(lags):
        if m >= 0:
            rxx[i] = np.sum(x[:N-m] * x[m:]) / N
        else:
            rxx[i] = np.sum(x[-m:] * x[:N+m]) / N
    

    # --- Step 2: Compute PSD via direct DFT ---
    freqs = np.linspace(0, fs/2, N, endpoint=False)
    Pxx = np.zeros(N, dtype=complex)

    for k, f in enumerate(freqs):
        Pxx[k] = np.sum(rxx * np.exp(-1j * 2 * np.pi * f * lags / fs))

    # Keep real part (should be real-valued)
    Pxx = np.real(Pxx)

    if plot:
        plt.figure(figsize=(10,4))
        plt.plot(freqs, Pxx, label="Periodogram")
        plt.xlabel("Frequency")
        plt.ylabel("Amplitude?")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return Pxx


if __name__ == '__main__':
    x, fs = generate_signal(50)
    pxx = periodogram(x, True)
    pxx = correlogram(x, True)