import numpy as np
from scipy.signal.windows import kaiser
from matplotlib import pyplot as plt
from scipy.signal import chirp

def generate_signal(N, fs, K, plot=False):
    t = np.arange(N) / fs

    a = np.random.normal(0, 10, K)
    f = np.random.uniform(0, fs/2, 2*K)
    
    x = 0
    for k in range(K):
        x += a[k] * chirp(t, f0=f[2*k], f1=f[2*k+1], t1=t[-1], method='linear')

    noise = np.random.normal(0, 5, size=N)

    # x = x + noise

    if plot:
        plt.figure(figsize=(15,3))
        plt.plot(t, x.real, label="Real part")
        plt.legend()
        plt.title(f"Dynamic signal (N={N}, order={K})")
        plt.xlabel("Time [s]")
        plt.tight_layout()

    return x, fs, f

def get_kaiser_shape(A):
    """
    Calculate the Kaiser window shape parameter (a) based on the desired attenuation.

    Parameters:
        A (float): Desired stopband attenuation in dB.

    Returns:
        float: Kaiser window shape parameter (a).
    """
    if A < 13.36:
        a = 0
    elif 13.26 <= A < 60:
        a = 0.76609 * (A - 13.26)**0.4 + 0.09834 * (A - 13.26)
    elif A >= 60:
        a = 0.12438 * (A + 6.3)
    
    # print(f'Kaiser window shape: {a}')
    return a


def get_kaiser_length(Dfw, A, fs):
    """
    Calculate the required length of the Kaiser window.

    Parameters:
        Dfw (float): Transition width in Hz.
        A (float): Desired stopband attenuation in dB.
        fs (float): Sampling frequency in Hz.

    Returns:
        float: Kaiser window length (N).
    """
    N = np.ceil((fs * 6 *(A + 12) / (Dfw * 155)) + 1).astype(int)
    # print(f'Kaiser window length: {N}')
    return N


def stft(x, fs=1, win=None, noverlap=0):
    assert win is not None, "Window function must be provided"
    x = np.asarray(x)
    win = np.asarray(win)
    win_len = len(win)
    hop = win_len - noverlap
    n_frames = 1 + (len(x) - win_len) // hop
    nfft = win_len
    stft_matrix = np.zeros((nfft, n_frames), dtype=np.complex64)

    for i in range(n_frames):
        start = i * hop
        frame = x[start:start + win_len] * win
        stft_matrix[:, i] = np.fft.fft(frame)

    t = np.arange(n_frames) * hop / fs
    f = np.fft.fftfreq(nfft)
    return stft_matrix, f, t


def compute_normalisation_factor(nfft, fs, win=None, mode="psd"):
    if win is None:
        win = np.ones(nfft)

    win_energy = np.sum(win ** 2)

    if mode == "psd":
        return fs * win_energy
    elif mode == "ps":
        return win_energy
    else:
        raise ValueError("mode must be either 'psd' or 'ps'")
    

if __name__ == '__main__':

    x, fs = generate_signal(50)
    A = 40
    Dfw = 100
    a = get_kaiser_shape(A)
    N = get_kaiser_length(Dfw, A, fs)
    win = kaiser(N, a)
    stft_matrix, f, t = stft(x, fs, win, noverlap=0, nfft=N)