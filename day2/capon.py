import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from music import generate_signal

# np.random.seed(1)

# # --- Parameters ---
fs = 8000                # sample rate (normalized)
N = 1024                # total samples
t = np.arange(N) / fs


x, fs, f = generate_signal(N=N, fs=fs, k=2, plot=False)

# --- Build data matrix of sliding windows y~(t) = [y(t), y(t-1), ..., y(t-m)]^T ---
m = 31                   # filter length m -> vector length L = m+1
L = m + 1
# Form matrix Ytilde of size (L x Nsnap) where Nsnap = N - m
Nsnap = N - m
Y = np.zeros((L, Nsnap), dtype=complex)
for tt in range(Nsnap):
    Y[:, tt] = x[tt:tt+L][::-1]   # y(t), y(t-1), ... y(t-m) ; reverse so index 0 is y(t)

# Sample covariance Rhat (LxL)
Rhat = (Y @ Y.conj().T) / Nsnap

# Regularize Rhat for numerical stability (diagonal loading)
# delta = 1e-6 * np.trace(Rhat) / L
# Rhat += delta * np.eye(L)

# Precompute inverse
Rinv = np.linalg.inv(Rhat)

# Frequency grid for spectral estimate
K = 2048
freqs = np.linspace(0, 0.5, K, endpoint=True)   # only 0..0.5 (Nyquist) since real-valued signal

# steering vector a(omega) = [1, e^{-i omega}, e^{-i 2 omega}, ..., e^{-i m omega}]^T (backward filtering convention from Stoica)
# and g(omega) = (1/(N-m)) sum_{t=m+1..N} y~(t) e^{-i omega t}  (see Stoica eq (5.6.45))
a_all = np.exp(-1j * 2*np.pi * np.outer(np.arange(L), freqs))  # shape (L, K)

# compute g(omega)
# note: t used in g definition is the absolute time index corresponding to each snapshot (we used snapshot tt -> original time index = tt + m)
time_indices = np.arange(m, N)  # t = m .. N-1  (N-m snapshots)
exponent = np.exp(-1j * 2*np.pi * np.outer(time_indices, freqs))  # shape (Nsnap, K)
# Y columns correspond to snapshots at t = m .. N-1 in the same order, where Y[:,tt] corresponds to time index tt + m
g_all = (Y @ exponent) / Nsnap   # shape (L, K)  (matrix multiplication sums over snapshots)

# --- Periodogram (Welch-like average using Hamming window) ---
# Use a simple averaged periodogram via FFT on overlapping segments for comparison
from scipy.signal import welch
f_welch, Pxx = welch(x, fs=fs, window='hamming', nperseg=256, noverlap=128, nfft=K*2, return_onesided=True)
# keep only 0..0.5 (normalized)
f_welch = f_welch / fs
# --- Capon PSD ---
P_capon = np.zeros(K)
for k in range(K):
    a = a_all[:, k].reshape(-1,1)
    denom = (a.conj().T @ Rinv @ a).item()
    P_capon[k] = 1.0 / np.real(denom)   # MVDR/Capon power estimate (up to scaling)

# --- APES amplitude estimate and PSD = |beta|^2 ---
P_apes = np.zeros(K)
beta = np.zeros(K, dtype=complex)
for k in range(K):
    a = a_all[:, k].reshape(-1,1)            # (L,1)
    g = g_all[:, k].reshape(-1,1)            # (L,1)
    aRinv = a.conj().T @ Rinv                # (1,L)
    gRinv = g.conj().T @ Rinv                # (1,L)
    num = (a.conj().T @ Rinv @ g).item()     # scalar (complex) = a^H R^{-1} g
    term = (g.conj().T @ Rinv @ g).item()    # g^H R^{-1} g (scalar, possibly complex but should be real)
    denom = (1 - term) * (a.conj().T @ Rinv @ a).item() + np.abs(num)**2
    # avoid tiny denom
    if np.abs(denom) < 1e-12:
        beta_k = 0.0 + 0.0j
    else:
        beta_k = num / denom
    beta[k] = beta_k
    P_apes[k] = np.abs(beta_k)**2

# Frequency axis normalization: periodogram has units of power per Hz; our Capon/APES are relative - we'll normalize for plotting
# Normalize all to max=1 for easier comparison (relative shapes)
P_capon_norm = P_capon / np.max(P_capon)
P_apes_norm = P_apes / np.max(P_apes) if np.max(P_apes)>0 else P_apes
Pxx_interp = np.interp(freqs, f_welch, Pxx / np.max(Pxx))

# --- Plot results ---
plt.figure(figsize=(10,5))
plt.plot(freqs, Pxx_interp, label='Welch Periodogram (norm)')
plt.plot(freqs, P_capon_norm, label='Capon (MVDR) (norm)')
plt.plot(freqs, P_apes_norm, label='APES |beta|^2 (norm)')

for tf in f:
    plt.axvline(tf/fs, color='r', linestyle='--', alpha=0.2, label='True Frequency' if 'True Frequency' not in plt.gca().get_legend_handles_labels()[1] else "")
if 'True Frequency' in plt.gca().get_legend_handles_labels()[1]:
    plt.legend()
plt.xlabel('Frequency (normalized)')
plt.ylabel('Normalized power (relative)')
plt.title('Comparison: Periodogram vs Capon vs APES (normalized)')
plt.legend()
plt.grid(True)
plt.show()

# --- Also show estimated complex amplitude (APES) magnitude near peaks ---
# print peak frequencies found by Capon and APES (simple peak pick)
pk_idx_capon = np.argsort(P_capon)[-4:][::-1]
pk_idx_apes = np.argsort(P_apes)[-4:][::-1]
print("Top Capon freq estimates (normalized):", freqs[pk_idx_capon])
print("Top APES freq estimates (normalized):", freqs[pk_idx_apes])

# Display sample of beta (complex amplitude) around the true frequencies
def nearest_idx(array, values):
    return [np.argmin(np.abs(array - v)) for v in values]

near_idx = nearest_idx(freqs, f)
for i, idx in enumerate(near_idx):
    print(f"APES beta at f{ i+1 } ~ {freqs[idx]:.5f}: magnitude={np.abs(beta[idx]):.4f}, phase={np.angle(beta[idx]):.3f} rad")
