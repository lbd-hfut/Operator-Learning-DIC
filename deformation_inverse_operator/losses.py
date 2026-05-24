"""Route B-specific losses.

Includes latent space regularization: encourages the latent code z
to be smooth and well-structured in the deformation latent space.
"""
import torch
import torch.nn as nn


class LatentConsistencyLoss(nn.Module):
    """Encourages consistent z for similar deformations.

    For two different speckle patterns with the same underlying displacement
    field, the latent codes should be similar.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """MSE between two latent codes."""
        return torch.nn.functional.mse_loss(z1, z2)


class LatentDiversityLoss(nn.Module):
    """Encourages diverse latent token usage.

    Penalizes tokens that are too similar to each other within a batch.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Penalize cosine similarity between latent tokens.

        Args:
            z: [B, M, d] latent code
        """
        B, M, d = z.shape
        z_norm = torch.nn.functional.normalize(z, dim=-1)
        # Pairwise cosine similarity matrix [B, M, M]
        sim = torch.matmul(z_norm, z_norm.transpose(-1, -2))
        # Exclude diagonal
        mask = torch.eye(M, device=z.device).unsqueeze(0)
        sim = sim * (1 - mask)
        return sim.abs().mean()
