"""Quick smoke test for speckle generation."""
import numpy as np
from dataset.speckle_generator import SpeckleGenerator


def test_speckle_basic():
    gen = SpeckleGenerator(image_size=(128, 128), density=0.4, contrast=0.6, seed=42)
    img = gen.generate()
    assert img.shape == (128, 128)
    assert img.dtype == np.float32
    assert img.min() >= 0.0
    assert img.max() <= 1.0
    assert img.std() > 0.01  # Not a blank image
    print("[PASS] test_speckle_basic")


def test_speckle_batch():
    gen = SpeckleGenerator(image_size=(64, 64), density=0.3, seed=0)
    batch = gen.generate_batch(4)
    assert batch.shape == (4, 64, 64)
    # Each image should be different
    assert not np.allclose(batch[0], batch[1])
    print("[PASS] test_speckle_batch")


def test_speckle_density():
    """Higher density = more particles = higher variance."""
    gen_low = SpeckleGenerator(image_size=(128, 128), density=0.1, contrast=0.6, seed=42)
    gen_high = SpeckleGenerator(image_size=(128, 128), density=0.7, contrast=0.6, seed=42)
    img_low = gen_low.generate()
    img_high = gen_high.generate()
    # Denser speckle = more intensity variation
    assert img_high.std() > img_low.std() * 0.5
    print("[PASS] test_speckle_density")


if __name__ == "__main__":
    test_speckle_basic()
    test_speckle_batch()
    test_speckle_density()
    print("All speckle tests passed.")
