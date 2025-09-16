# Python script to simulate the described array, compute delays, generate snapshots,
# compute covariance, implement Bartlett (classic) and Capon beamformers, and plot spectra.
# This will run in the notebook and display results/plots.
import numpy as np
import matplotlib.pyplot as plt


def generate_signal(N, fs:int | float, A:list, f:list, phase:list, noise:bool=False):
    '''
    Parameters:
        L: number of snapshots
    '''
    A = np.array(A)
    f = np.array(f)
    phase = np.array(phase)

    assert A.shape == f.shape == phase.shape, 'Signal amplitudes A, frequencies f and phases phase must have the same shape.'

    n = np.arange(N)
    t = n / fs

    x = np.sum(A[:, np.newaxis] * np.exp(1j * (2 * np.pi * f[:, np.newaxis] * t + phase[:, np.newaxis])), axis=0)

    if noise:
        noise = (np.sqrt(1.0/2) * (np.random.randn(M, N) + 1j*np.random.randn(M, N)))
        x = x + noise
    return x


def get_delays(M, d:float, doas:list):
    '''
    Parameters:
        M: number of sensors
        d: sensor spacing
        doas (array): DOAs of sources
    '''
    doas = np.array(doas)
    c = 343.0
    m_idx = np.arange(M)
    positions = m_idx * d

    tau = np.zeros((positions.shape[0], doas.shape[0]))

    for i, doa in enumerate(doas):
        tau[:, i] = positions * np.sin(doa)  / c
   
    return tau


if __name__ == '__main__':
    M = 16
    d = 0.05 
    f = [500, 500]
    fs = 8000
    doas = [-np.pi/4, np.pi/5]
    N = 64 # number of snapshots
    A = [5.0, 3.0]
    phase = [0.3, 0.7]

    get_delays(M, d, doas)

    x = generate_signal(N, fs, A, f, phase)
# # Parameters
# M = 16                      # number of sensors
# d = 0.05                    # spacing in meters (5 cm)
# Fs = 8000                   # sampling rate in Hz
# f = 500                     # narrowband frequency in Hz for both sources
# A1 = 5.0                    # amplitude source 1
# A2 = 3.0                    # amplitude source 2
# theta1 = -np.pi/4           # DOA source 1 (radians)
# theta2 = np.pi/5            # DOA source 2 (radians)
# c = 343.0                   # speed of propagation (m/s) -- assumed sound speed
# N = 64                      # number of snapshots
# sigma2 = 1.0                # noise variance (complex noise, E[|n|^2] = sigma2)



# # Steering vector function (narrowband complex baseband)
# def steering_vector(theta):
#     # returns Mx1 complex steering vector with reference at sensor 0
#     phase_shifts = -1j * k * positions * np.sin(theta)
#     a = np.exp(phase_shifts)
#     return a.reshape(M, 1)

# a1 = steering_vector(theta1)
# a2 = steering_vector(theta2)

# # Generate narrowband complex signals for L snapshots
# # Use complex exponentials at frequency f sampled at Fs, with arbitrary initial phases
# n = np.arange(L)
# t = n / Fs
# phase1 = 0.3  # arbitrary fixed phase for source 1
# phase2 = -0.7 # arbitrary fixed phase for source 2

# s1 = A1 * np.exp(1j * (2*np.pi*f*t + phase1))  # length L (1D)
# s2 = A2 * np.exp(1j * (2*np.pi*f*t + phase2))

# # Build data matrix X (M x L)
# # X = a1 * s1 + a2 * s2 + noise
# S = np.vstack([s1, s2])  # 2 x L (not necessary but for clarity)
# X_signal = a1 @ s1.reshape(1, L) + a2 @ s2.reshape(1, L)

# # Complex Gaussian noise: real and imag each with variance sigma2/2
# noise = (np.sqrt(sigma2/2) * (np.random.randn(M, L) + 1j*np.random.randn(M, L)))

# X = X_signal + noise

# # Sample spatial covariance matrix (normalized)
# R = (1.0 / L) * (X @ X.conj().T)

# print("\nData matrix X shape:", X.shape)
# print("Spatial covariance matrix R shape:", R.shape)
# print("First 3x3 block of R (complex):\n", R[:3, :3])

# # Angle grid for scanning (degrees)
# angles_deg = np.linspace(-90, 90, 721)
# angles_rad = np.deg2rad(angles_deg)

# # Precompute steering vectors over grid
# A_grid = np.hstack([steering_vector(theta) for theta in angles_rad])  # M x Ntheta

# # Bartlett (classic) beamformer (spatial periodogram)
# P_bartlett = np.real(np.sum(np.conj(A_grid).transpose(1,0) @ R @ A_grid, axis=1))
# # The line above computes a^H R a for each steering vector; rearranged for speed.
# # Normalize by M so the scale is comparable
# P_bartlett = P_bartlett / M

# # Capon (MVDR) beamformer
# R_inv = np.linalg.inv(R + 1e-6 * np.eye(M))  # small diagonal loading for stability
# P_capon = np.zeros(angles_rad.size, dtype=float)
# for idx in range(angles_rad.size):
#     a = A_grid[:, idx].reshape(M,1)
#     denom = np.real((a.conj().T @ R_inv @ a)[0,0])
#     P_capon[idx] = 1.0 / denom

# # Normalize Capon to have comparable peak levels
# P_capon = P_capon / np.max(P_capon)
# P_bartlett = P_bartlett / np.max(P_bartlett)

# # Plot Bartlett spectrum
# plt.figure(figsize=(8,4))
# plt.plot(angles_deg, 10*np.log10(P_bartlett + 1e-12))  # dB scale
# plt.title("Bartlett (Classic) Spatial Spectrum")
# plt.xlabel("Angle (degrees)")
# plt.ylabel("Power (dB, normalized)")
# plt.grid(True)
# plt.xlim(-90, 90)
# plt.ylim(-60, 5)
# plt.show()

# # Plot Capon spectrum
# plt.figure(figsize=(8,4))
# plt.plot(angles_deg, 10*np.log10(P_capon + 1e-12))
# plt.title("Capon (MVDR) Spatial Spectrum")
# plt.xlabel("Angle (degrees)")
# plt.ylabel("Power (dB, normalized)")
# plt.grid(True)
# plt.xlim(-90, 90)
# plt.ylim(-60, 5)
# plt.show()

# # Also plot both on same figure for direct visual comparison (separate plot but combined lines)
# plt.figure(figsize=(8,4))
# plt.plot(angles_deg, 10*np.log10(P_bartlett + 1e-12), label='Bartlett')
# plt.plot(angles_deg, 10*np.log10(P_capon + 1e-12), label='Capon', linestyle='--')
# plt.title("Comparison: Bartlett vs Capon Spatial Spectrum")
# plt.xlabel("Angle (degrees)")
# plt.ylabel("Power (dB, normalized)")
# plt.legend()
# plt.grid(True)
# plt.xlim(-90, 90)
# plt.ylim(-60, 5)
# plt.show()

# # Return key variables for inspection
# print("\nPeak locations (degrees) from grids (rough estimate):")
# peak_bart = angles_deg[np.argmax(P_bartlett)]
# peak_capon = angles_deg[np.argmax(P_capon)]
# print(f" Bartlett peak at {peak_bart:.2f} deg (true: {np.rad2deg(theta1):.2f} and {np.rad2deg(theta2):.2f})")
# print(f" Capon   peak at {peak_capon:.2f} deg (true: {np.rad2deg(theta1):.2f} and {np.rad2deg(theta2):.2f})")
