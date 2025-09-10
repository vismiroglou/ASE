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
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import numpy as np 

def generate_signal(N, fs, k, plot=False):
  np.random.seed(42)
  t = np.arange(N) / fs

  a = np.random.normal(0, 10, k)
  f = np.random.uniform(0, fs/2, k)
  phi = np.random.uniform(-np.pi, np.pi, k)

  a = a[:, np.newaxis]
  f = f[:, np.newaxis]
  phi = phi[:, np.newaxis]
  t = t[np.newaxis, :]
  x = np.sum(a * np.exp(1j * 2 * np.pi * f * t + 1j * phi), axis=0)

  noise_real = np.random.normal(0, 5, size=N)
  noise_imag = np.random.normal(0, 5, size=N)
  noise = noise_real + 1j * noise_imag

  x = x + noise

  if plot:
      plt.figure()
      plt.plot(range(N), x.real, label="Real part")
      plt.plot(range(N), x.imag, label="Imag part")
      plt.legend()
      plt.title(f"Complex Signal (N={N}, order={k})")
      plt.xlabel("Time [s]")
      plt.tight_layout()

  return x, fs, f


def music_psd(x, m, k, true_freqs=None, fourier=False, plot=False):
  N = len(x)
  freqs = np.linspace(0, fs/2, N)

  Y = np.zeros((m, N-m + 1), dtype=complex)
  for i in range(N - m + 1):
      Y[:, i] = x[i + m - 1 : i - 1 : -1] if i > 0 else x[m-1::-1]

  R = (Y @ Y.conj().T) / (N - m + 1)
  eigenvalues, eigenvectors = np.linalg.eig(R)
  idx = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:, idx]

  if plot:
    plt.figure()
    plt.plot(range(len(eigenvalues)), np.sort(eigenvalues.real)[::-1], 'o-')
    plt.yscale("log")
    plt.title(f"Eigenvalues of the Autocorrelation Matrix. Model order gt: {k}")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.grid()
    plt.tight_layout()

  noise_sub = eigenvectors[:, k:]

  if fourier:
    noise_fft = np.fft.fft(noise_sub, n=N)  # shape (N, n_noise)
    Pxx_music = 1 / np.sum(np.abs(noise_fft) ** 2, axis=1)
    Pxx_music = Pxx_music[:N//2]  # Take positive frequencies
    freqs = np.fft.fftfreq(N, d=1/fs)[:N//2]

  else:
    Pxx_music = np.zeros_like(freqs)
    for i, f in enumerate(freqs):
      a = np.exp(-1j * 2 * np.pi * f * np.arange(m) / fs)
      denom = np.conj(a) @ noise_sub @ noise_sub.conj().T @ a
    Pxx_music[i] = 1 / np.abs(denom)

  if plot:
    plt.figure()
    plt.plot(freqs, Pxx_music)
    # if true_freqs is not None:
    #     for tf in true_freqs:
    #         plt.axvline(tf, color='r', linestyle='--', alpha=0.2, label='True Frequency' if 'True Frequency' not in plt.gca().get_legend_handles_labels()[1] else "")
    #     if 'True Frequency' in plt.gca().get_legend_handles_labels()[1]:
    #         plt.legend()
    plt.title("MUSIC PSD Estimate")
    plt.xlabel("Frequency")
    plt.grid()
    plt.tight_layout()
  return Pxx_music


def build_rhat(x, N, m):
	L = m + 1
	Nsnap = N - m
	Y = np.zeros((L, Nsnap), dtype=complex)
	for tt in range(Nsnap):
		Y[:, tt] = x[tt:tt+L][::-1]   # y(t), y(t-1), ... y(t-m) ; reverse so index 0 is y(t)

	Rhat = (Y @ Y.conj().T) / Nsnap

	Rinv = np.linalg.inv(Rhat)
	return Rhat, Rinv


def capon(m, K):
    L = m + 1
    freqs = np.linspace(0, 0.5, K, endpoint=True)
    a_all = np.exp(-1j * 2*np.pi * np.outer(np.arange(L), freqs))
    P_capon = np.zeros(K)
    for k in range(K):
        a = a_all[:, k].reshape(-1,1)
        denom = (a.conj().T @ Rinv @ a).item()
        P_capon[k] = 1.0 / np.real(denom)   # MVDR/Capon power estimate (up to scaling)
    return P_capon


if __name__ == '__main__':
    # Set parameters
    fs = 8000 
    N = 1024 
    t = np.arange(N) / fs
	
	
    x, fs, true_freqs = generate_signal(N=N, fs=fs, k=2, plot=False)
    Rhat, Rinv = build_rhat(x, N, m=31)
    

    # Pxx = music_psd(x, 50, k, abs(true_freqs), fourier=True, plot=True)
    # plt.show()