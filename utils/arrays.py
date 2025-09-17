import numpy as np

class CaponBeamformer:
    def __init__(self, x, M):
        self.x = x
        self.M = M

        self.R, self.R_inv = self._get_spatial_covariance()

    def _get_spatial_covariance(self):
        _, N = self.x.shape
        R = (1.0 / N) * (self.x @ self.x.conj().T)
        R_inv = np.linalg.inv(R + 1e-6 * np.eye(self.M))
        return R, R_inv

    def _steering_vectors(self, theta, d, f, c=343.0):
        thetas = np.atleast_1d(theta)
        freqs = np.atleast_1d(f)
        m_idx = np.arange(self.M) * d

        A = np.zeros((self.M, len(thetas)), dtype=complex)
        for i, (theta, f) in enumerate(zip(thetas, freqs)):
            k = 2 * np.pi * f / c
            phase_shifts = -1j * k * m_idx * np.sin(theta)
            A[:, i] = np.exp(phase_shifts)
        return A

    def _get_grid(self, d, f, c=343):
        angles_deg = np.linspace(-90, 90, 721)
        angles_rad = np.deg2rad(angles_deg)

        A_grid = self._steering_vectors(angles_rad, d, np.full_like(angles_rad, np.mean(f)))

        return A_grid, angles_deg, angles_rad

    def compute_spectrum(self, d, f):
        A_grid, angles_deg, angles_rad = self._get_grid(d, f)

        P_capon = np.zeros(len(angles_rad))
        for i in range(len(angles_rad)):
            a = A_grid[:, i].reshape(self.M,1)
            denom = np.real((a.conj().T @ self.R_inv @ a)[0,0])
            P_capon[i] = 1.0 / denom
        return P_capon


class ClassicBeamformer:
    def __init__(self, x, M):
        self.x = x
        self.M = M

        self.R, self.R_inv = self._get_spatial_covariance()

    def _get_spatial_covariance(self):
        _, N = self.x.shape
        R = (1.0 / N) * (self.x @ self.x.conj().T)
        R_inv = np.linalg.inv(R + 1e-6 * np.eye(self.M))
        return R, R_inv

    def _steering_vectors(self, theta, d, f, c=343.0):
        thetas = np.atleast_1d(theta)
        freqs = np.atleast_1d(f)
        m_idx = np.arange(self.M) * d

        A = np.zeros((self.M, len(thetas)), dtype=complex)
        for i, (theta, f) in enumerate(zip(thetas, freqs)):
            k = 2 * np.pi * f / c
            phase_shifts = -1j * k * m_idx * np.sin(theta)
            A[:, i] = np.exp(phase_shifts)
        return A

    def _get_grid(self, d, f, c=343):
        angles_deg = np.linspace(-90, 90, 721)
        angles_rad = np.deg2rad(angles_deg)

        A_grid = self._steering_vectors(angles_rad, d, np.full_like(angles_rad, np.mean(f)))

        return A_grid, angles_deg, angles_rad

    def compute_spectrum(self, d, f):
        A_grid, angles_deg, angles_rad = self._get_grid(d, f)

        P_classic = np.zeros(len(angles_rad))
        for i in range(len(angles_rad)):
            a = A_grid[:, i].reshape(self.M,1)
            P_classic[i] = np.real((a.conj().T @ self.R @ a)[0,0])
        P_classic /= (self.M ** 2)

        return P_classic


class ArraySignal:
    def __init__(self, M, d):
        self.M = M
        self.d = d
        np.random.seed(42)

    def _steering_vectors(self, thetas, freqs, c=343.0):
        thetas = np.atleast_1d(thetas)
        freqs = np.atleast_1d(freqs)
        m_idx = np.arange(self.M) * self.d

        A = np.zeros((self.M, len(thetas)), dtype=complex)
        for i, (theta, f) in enumerate(zip(thetas, freqs)):
            k = 2 * np.pi * f / c
            phase_shifts = -1j * k * m_idx * np.sin(theta)
            A[:, i] = np.exp(phase_shifts)
        return A


    def generate_signal(self, N, fs, A, freqs, theta, sigma=1.0):

        n = np.arange(N)
        t = n / fs

  
        phases = np.random.uniform(0, 2*np.pi, size=len(A))
        S = np.zeros((len(A), N), dtype=complex)
        
        for i, (amp, f, phi) in enumerate(zip(A, freqs, phases)):
            S[i, :] = amp * np.exp(1j * (2*np.pi*f*t + phi))
 
        A_true = self._steering_vectors(theta, freqs)
        
        X_signal = A_true @ S
        noise = np.sqrt(sigma/2) * (np.random.randn(self.M, N) + 1j*np.random.randn(self.M, N))
        X = X_signal + noise
        return X


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