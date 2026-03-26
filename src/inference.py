from scipy.signal import fftconvolve
import numpy as np

def _normalize_pdf(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    area = np.trapezoid(y, x)
    if area <= 0:
        return y
    return y / area


def _estimate_advective_kernel(t: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Estimate advective travel-time distribution g(t)
    using a smooth normalized BTC (early-time dominant).
    """
    c_norm = _normalize_pdf(c, t)

    # Emphasize early-time behavior (less retention influence)
    cutoff = int(0.4 * len(t))
    g = np.zeros_like(c_norm)
    g[:cutoff] = c_norm[:cutoff]

    return _normalize_pdf(g, t)


def _deconvolve_retention(t: np.ndarray, f: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Estimate retention kernel h(t) such that:
        f(t) ≈ g * h
    using FFT-based deconvolution.
    """
    eps = 1e-8

    F = np.fft.rfft(f)
    G = np.fft.rfft(g)

    H = F / (G + eps)

    h = np.fft.irfft(H, n=len(t))
    h = np.maximum(h, 0.0)

    return _normalize_pdf(h, t)


def build_effective_signals(
    btc_smoothed: SmoothedSeries,
    snapshot_moments: SnapshotMoments,
) -> EffectiveSignals:

    t = btc_smoothed.x_uniform
    f = np.maximum(btc_smoothed.y_smooth, 0.0)

    # -------------------------------
    # 1. Estimate advective kernel g(t)
    # -------------------------------
    g = _estimate_advective_kernel(t, f)

    # -------------------------------
    # 2. Estimate retention kernel h(t)
    # -------------------------------
    h = _deconvolve_retention(t, f, g)

    # -------------------------------
    # 3. Velocity from travel-time kernel
    # -------------------------------
    # Mean travel time
    mean_t = np.trapezoid(t * g, t)

    # Effective velocity (assuming unit length or normalized)
    velocity_signal = np.full_like(t, 1.0 / (mean_t + 1e-8))

    # -------------------------------
    # 4. Retention signal = h(t)
    # -------------------------------
    retention_signal = h

    # -------------------------------
    # 5. Direct kernel (optional)
    # -------------------------------
    direct_kernel_signal = np.maximum(f - g, 0.0)

    return EffectiveSignals(
        time=t,
        btc_smooth=f,
        velocity_signal=velocity_signal,
        retention_signal=retention_signal,
        direct_kernel_signal=direct_kernel_signal,
    )
