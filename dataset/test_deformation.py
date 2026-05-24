"""Quick smoke test for deformation generation."""
import numpy as np
from dataset.deformation_generator import DeformationGenerator


def test_all_modes():
    gen = DeformationGenerator(image_size=(128, 128), displacement_range=(1.0, 10.0), seed=42)
    for mode in ["tension", "compression", "shear", "rotation", "composite"]:
        u = gen.generate(mode=mode, amplitude=5.0)
        assert u.shape == (128, 128, 2)
        assert u.dtype == np.float32
        # Displacement should have some variation
        assert u.std() > 0.0
        print(f"  [PASS] test_mode_{mode}: max(|u|) = {np.abs(u).max():.2f}")


def test_random_mode():
    gen = DeformationGenerator(image_size=(64, 64), seed=0)
    for _ in range(10):
        u = gen.generate()
        assert u.shape == (64, 64, 2)
    print("[PASS] test_random_mode")


def test_amplitude_range():
    gen = DeformationGenerator(image_size=(64, 64), displacement_range=(0.1, 20.0), seed=42)
    u_small = gen.generate(mode="tension", amplitude=0.5)
    u_large = gen.generate(mode="tension", amplitude=15.0)
    assert u_large.max() > u_small.max() * 2
    print("[PASS] test_amplitude_range")


if __name__ == "__main__":
    test_all_modes()
    test_random_mode()
    test_amplitude_range()
    print("All deformation tests passed.")
