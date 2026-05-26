"""Real speckle image pool for DIC dataset generation.

Loads real speckle pattern images from a directory and provides
random patches for use as reference images in DIC data generation.
"""
import numpy as np
from pathlib import Path
from PIL import Image


class RealImagePool:
    """Load and manage a pool of real speckle pattern images.

    On initialization, scans a directory for image files, loads them
    as grayscale float32 normalized to [0,1], and caches them in memory.

    Args:
        image_dir: path to directory containing speckle images
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"}

    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {image_dir}")

        self._images: list[np.ndarray] = []
        self._load_all()

    def _load_all(self):
        paths = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        )
        if not paths:
            raise FileNotFoundError(f"No supported images found in {self.image_dir}")

        for p in paths:
            img = Image.open(p).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0
            self._images.append(arr)

        print(f"RealImagePool: loaded {len(self._images)} images from {self.image_dir}")

    @property
    def n_images(self) -> int:
        return len(self._images)

    def load_random_patch(self, image_size: tuple[int, int]) -> np.ndarray:
        """Load a random H x W crop from a randomly selected image.

        Args:
            image_size: (H, W) desired patch dimensions

        Returns:
            np.ndarray [H, W] float32, values in [0, 1]
        """
        H, W = image_size
        idx = np.random.randint(0, self.n_images)
        img = self._images[idx]

        h, w = img.shape

        # If image is smaller than target, resize it up
        if h < H or w < W:
            scale = max(H / h, W / w)
            new_h = max(int(h * scale), H)
            new_w = max(int(w * scale), W)
            img = self._resize_bilinear(img, new_h, new_w)
            h, w = img.shape

        # Random crop
        y0 = np.random.randint(0, h - H + 1)
        x0 = np.random.randint(0, w - W + 1)
        patch = img[y0:y0 + H, x0:x0 + W].copy()

        return patch

    def load_full(self, image_size: tuple[int, int], index: int) -> np.ndarray:
        """Load and resize a full image to target size (deterministic).

        Used by HDF5 generation for reproducibility.

        Args:
            image_size: (H, W) desired dimensions
            index: which image to load (mod n_images)

        Returns:
            np.ndarray [H, W] float32
        """
        img = self._images[index % self.n_images]
        return self._resize_bilinear(img, image_size[0], image_size[1])

    @staticmethod
    def _resize_bilinear(img: np.ndarray, h: int, w: int) -> np.ndarray:
        """Simple bilinear resize for numpy arrays."""
        from scipy.ndimage import zoom
        zoom_h = h / img.shape[0]
        zoom_w = w / img.shape[1]
        return zoom(img, (zoom_h, zoom_w), order=1)
