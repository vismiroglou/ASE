import numpy as np
import matplotlib.pyplot as plt

def compute_theta(rpm, fs):
    """
    Compute cumulative shaft rotation in revolutions at each sample.
    rpm: scalar or array (length N) of rpm values.
    fs: sampling frequency.
    Returns theta array (length N) in revolutions (i.e. number of turns).
    """
    if np.isscalar(rpm):
        # constant rpm
        omega_rev_per_s = rpm / 60.0
        return omega_rev_per_s
    else:
        revs_per_s = np.asarray(rpm) / 60.0
        theta = np.cumsum(revs_per_s) / fs
        return theta


def vsdft_frame(x_frame, theta_frame, orders, window=None):
    """
    Compute VSDFT for a single window/frame.
    x_frame: 1D array of samples in frame
    theta_frame: same-length array of cumulative revolutions for each sample (absolute or relative, only differences matter)
    orders: 1D array of orders to compute (can be floats)
    window: window array (same length as x_frame) or None
    Returns complex spectrum of shape (len(orders),)
    """
    x = np.asarray(x_frame)
    if window is not None:
        w = np.asarray(window)
        if w.shape != x.shape:
            raise ValueError("window must match x_frame length")
        x = x * w
    # theta_frame should be in revolutions. Compute kernel for each order:
    # kernel shape (len(orders), len(frame))
    # exp(-j*2*pi*order*theta)
    orders = np.asarray(orders)
    phase = np.exp(-1j * 2.0 * np.pi * np.outer(orders, theta_frame))
    # multiply and sum along time axis
    S = phase.dot(x)   # shape (len(orders),)
    return S


def vsdft(x, fs, orders, rpm, win, overlap=0.5):
    """
    Compute short-time Velocity Synchronous DFT (order spectrogram).

    Parameters:
        x (array-like): 1D input signal (length N).
        fs (float): Sampling frequency in Hz.
        orders (array-like): Orders to evaluate (e.g., np.arange(1, 20)).
        rpm (float or array-like): Scalar or 1D array of instantaneous rpm sampled at fs (length N).
        win (array-like): Window array (length of each frame).
        overlap (float, optional): Fractional overlap between frames [0, <1]. Default is 0.5.

    Returns:
        times (np.ndarray): Center times of frames in seconds (shape: n_frames).
        orders (np.ndarray): Array of requested orders (shape: len(orders)).
        S (np.ndarray): Complex spectrogram (shape: [len(orders), n_frames]).
    """
    N = len(x)
    
    M = len(win)
    if M < 1:
        raise ValueError("frame_len too small")
    step = int(np.round(M * (1.0 - overlap)))
    if step < 1:
        raise ValueError("overlap too large -> step < 1")

    # prepare theta (revolutions per sample)
    if np.isscalar(rpm):
        revs_per_s = rpm / 60.0
        theta_full = (np.arange(N) * (revs_per_s / fs))
    else:
        rpm = np.asarray(rpm)
        if len(rpm) != N:
            raise ValueError("rpm array must have same length as x")
        theta_full = compute_theta(rpm, fs)

    # number of frames
    frames = np.arange(0, N - M + 1, step)
    times = (frames + M/2.0) / fs  # center time of each frame

    orders = np.asarray(orders)
    S = np.zeros((len(orders), len(frames)), dtype=complex)

    for i, start in enumerate(frames):
        end = start + M
        x_frame = x[start:end]
        theta_frame = theta_full[start:end]
        S[:, i] = vsdft_frame(x_frame, theta_frame, orders, window=win)

    S /= np.sum(win)

    return S, times, orders


def generate_order_rpm(N, fs, rpm_start, rpm_end, ord_true):
    t = np.arange(N) / fs
    rpm = rpm_start + (rpm_end - rpm_start) * (t / t.max())

    theta = compute_theta(rpm, fs)
    if np.isscalar(theta):
        theta = (np.arange(N) * (theta / fs))

    x = 0.8 * np.sin(2*np.pi*ord_true*theta)

    return x, fs, rpm

  
if __name__ == "__main__":
    from scipy.signal.windows import kaiser
    fs = 4000
    ord_true = 3
    win = kaiser(500, 0.8)
    x, _, rpm = generate_order_rpm(8000, fs, 2000, 3000, ord_true)

    S, times, orders = vsdft(x, fs, np.arange(1,20), rpm, win)

    # Plot the order spectrogram
    plt.figure(figsize=(10,6))
    extent = [times[0], times[-1], orders[0], orders[-1]]
    plt.pcolormesh(times, orders, np.abs(S), shading="gouraud")
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel("Time [s]")
    plt.ylabel("Order")
    plt.title("VSDFT Order Spectrogram (short-time)")
    plt.axhline(ord_true, color='r', linestyle='--', linewidth=1.0, label=f"true order {ord_true:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Single-frame VSDFT at mid-time
    mid_idx = len(times)//2
    S_mid = S[:, mid_idx]
    plt.figure(figsize=(6,4))
    plt.plot(orders, 20*np.log10(np.abs(S_mid) + 1e-12))
    plt.xlabel("Order")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"VSDFT at t = {times[mid_idx]:.3f} s")
    plt.grid(True)
    plt.show()