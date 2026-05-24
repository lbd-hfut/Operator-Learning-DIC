"""Dataset configuration dataclass."""
from dataclasses import dataclass, field
from typing import Tuple, List, Optional


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
    deformation_modes: Tuple[str, ...] = (
        "tension",
        "compression",
        "shear",
        "rotation",
        "composite",
    )
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
    mode: str = "on_the_fly"
    cache_path: Optional[str] = None
