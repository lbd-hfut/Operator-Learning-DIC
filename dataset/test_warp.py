"""Quick smoke test for image warping."""
import numpy as np
from dataset.speckle_generator import SpeckleGenerator
from dataset.deformation_generator import DeformationGenerator
from dataset.warp import warp_image


def test_warp_identity():
    """Zero displacement should produce identical image."""
    gen = SpeckleGenerator(image_size=(128, 128), seed=42)
    ref = gen.generate()
    u_zero = np.zeros((128, 128, 2), dtype=np.float32)
    tar = warp_image(ref, u_zero)
    assert np.allclose(ref, tar, atol=1e-5)
    print("[PASS] test_warp_identity")


def test_warp_translation():
    """Pure translation: rigidly shifts the image."""
    gen = SpeckleGenerator(image_size=(128, 128), density=0.5, contrast=0.8, seed=42)
    ref = gen.generate()
    shift = 10.0
    u_field = np.zeros((128, 128, 2), dtype=np.float32)
    u_field[..., 0] = shift  # shift right by 10px
    tar = warp_image(ref, u_field)
    # After shifting right, the left edge should be padded with 0
    assert tar[:, 0].sum() == 0.0  # leftmost column is out of bounds
    # Interior should match
    assert np.allclose(ref[:, 0], tar[:, 10], atol=1e-4)
    print("[PASS] test_warp_translation")


def test_warp_output_shape():
    gen = SpeckleGenerator(image_size=(64, 64), seed=0)
    ref = gen.generate()
    deform = DeformationGenerator(image_size=(64, 64), seed=0)
    u_field = deform.generate(mode="tension", amplitude=5.0)
    tar = warp_image(ref, u_field)
    assert tar.shape == (64, 64)
    print("[PASS] test_warp_output_shape")


if __name__ == "__main__":
    test_warp_identity()
    test_warp_translation()
    test_warp_output_shape()
    print("All warp tests passed.")
