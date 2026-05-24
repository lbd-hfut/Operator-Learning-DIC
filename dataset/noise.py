"""Camera noise models for synthetic DIC images."""
import numpy as np


def add_gaussian_noise(image: np.ndarray, std: float, seed: int = None) -> np.ndarray:
    """Add Gaussian noise to image.

    Args:
        image: [H, W] or [B, H, W] image array
        std: noise standard deviation
        seed: random seed

    Returns:
        noisy image, same shape as input
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(*image.shape).astype(np.float32) * std
    noisy = image + noise
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def add_poisson_noise(image: np.ndarray, scale: float = 100.0, seed: int = None) -> np.ndarray:
    """Add Poisson (shot) noise.

    Args:
        image: [H, W] array in [0, 1]
        scale: photon count scale factor
        seed: random seed

    Returns:
        noisy image
    """
    rng = np.random.RandomState(seed)
    photons = image * scale
    noisy_photons = rng.poisson(np.maximum(photons, 0))
    return np.clip(noisy_photons / scale, 0.0, 1.0).astype(np.float32)


def add_salt_pepper_noise(image: np.ndarray, prob: float = 0.01, seed: int = None) -> np.ndarray:
    """Add salt-and-pepper noise.

    Args:
        image: [H, W] array
        prob: probability of a pixel being corrupted
        seed: random seed

    Returns:
        noisy image
    """
    rng = np.random.RandomState(seed)
    noisy = image.copy()
    mask = rng.rand(*image.shape) < prob
    salt = rng.rand(*image.shape) < 0.5
    noisy[mask & salt] = 1.0
    noisy[mask & ~salt] = 0.0
    return noisy.astype(np.float32)
