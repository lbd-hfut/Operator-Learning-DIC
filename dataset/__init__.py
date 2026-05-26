from .config import DatasetConfig
from .speckle_generator import SpeckleGenerator
from .deformation_generator import DeformationGenerator
from .warp import warp_image
from .noise import add_gaussian_noise
from .sampler import QueryPointSampler
from .image_pool import RealImagePool

# Lazy imports that require torch (only needed for training)
try:
    from .dic_dataset import DICDataset
    from .folder_dataset import FolderDICDataset
    from .hdf5_dataset import HDF5DICDataset
    from .collate import collate_fn
except ImportError:
    DICDataset = None
    FolderDICDataset = None
    HDF5DICDataset = None
    collate_fn = None
