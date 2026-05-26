"""Dataset configuration dataclass."""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import numpy as np


@dataclass
class DatasetConfig:
    """Configuration for DIC dataset generation.

    Controls speckle pattern, deformation field, image properties,
    and query point sampling behavior.
    """

    # --- Speckle parameters ---
    speckle_size_range: Tuple[float, float] = (1.5, 6.0)
    speckle_density_range: Tuple[float, float] = (0.3, 0.6)
    speckle_contrast_range: Tuple[float, float] = (0.3, 0.9)

    # --- Deformation parameters ---
    deformation_modes: Dict[str, float] = field(default_factory=lambda: {
        "tension": 1.0,
        "compression": 1.0,
        "shear": 1.0,
        "rotation": 1.0,
        "composite": 1.0,
        "multiscale_random": 3.0,
    })
    displacement_range: Tuple[float, float] = (0.1, 20.0)

    # --- Image parameters ---
    image_size: Tuple[int, int] = (256, 256)
    noise_std_range: Tuple[float, float] = (0.0, 0.03)

    # --- Query point sampling ---
    n_query_min: int = 128
    n_query_max: int = 2048
    query_dropout_prob: float = 0.0
    gradient_weighted_sampling: bool = False

    # --- Dataset size ---
    n_samples: int = 10000
    seed: int = 42

    # --- Generation mode ---
    mode: str = "synthetic"              # "synthetic" | "real"
    real_image_dir: Optional[str] = None # path to real speckle image directory

    # --- Cache (not used for on-the-fly) ---
    cache_path: Optional[str] = None

    def sample_mode(self, rng: np.random.RandomState) -> str:
        """Sample a deformation mode according to configured weights.

        Args:
            rng: numpy RandomState for reproducible sampling

        Returns:
            mode name string
        """
        modes = list(self.deformation_modes.keys())
        weights = np.array([self.deformation_modes[m] for m in modes], dtype=float)
        weights = weights / weights.sum()
        return rng.choice(modes, p=weights)

    @property
    def mode_list(self) -> List[str]:
        """Active mode names (weight > 0)."""
        return [m for m, w in self.deformation_modes.items() if w > 0]
