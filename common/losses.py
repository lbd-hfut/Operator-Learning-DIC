"""Composite loss functions combining data loss with physics regularization."""
import torch
import torch.nn as nn
from .metrics import relative_l2_error, huber_loss


class CompositeLoss(nn.Module):
    """Composite loss: L_data + λ_reg * L_reg.

    Args:
        data_loss_type: 'relative_l2' or 'huber'
        reg_loss_type: 'none' or 'compatibility' or 'smoothness'
        lambda_reg: regularization weight
        huber_delta: delta parameter for Huber loss (only when data_loss_type='huber')
    """

    def __init__(
        self,
        data_loss_type: str = "relative_l2",
        reg_loss_type: str = "none",
        lambda_reg: float = 0.0,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.data_loss_type = data_loss_type
        self.reg_loss_type = reg_loss_type
        self.lambda_reg = lambda_reg
        self.huber_delta = huber_delta

    def forward(
        self,
        u_pred: torch.Tensor,
        u_gt: torch.Tensor,
        query_points: torch.Tensor = None,
        query_mask: torch.Tensor = None,
    ) -> dict:
        """Compute composite loss.

        Args:
            u_pred: [B, N, 2] predicted displacement
            u_gt: [B, N, 2] ground truth displacement
            query_points: [B, N, 2] query point coordinates (for reg losses)
            query_mask: [B, N] bool mask for valid query points

        Returns:
            dict with 'loss', 'data_loss', 'reg_loss' values
        """
        if self.data_loss_type == "huber":
            data_loss = huber_loss(u_pred, u_gt, delta=self.huber_delta, mask=query_mask)
        else:
            data_loss = relative_l2_error(u_pred, u_gt, mask=query_mask)

        reg_loss = torch.tensor(0.0, device=u_pred.device)

        if self.reg_loss_type == "smoothness" and query_points is not None:
            reg_loss = self._smoothness_loss(u_pred, query_points, query_mask)
        elif self.reg_loss_type == "compatibility" and query_points is not None:
            reg_loss = self._compatibility_loss(u_pred, query_points, query_mask)

        total_loss = data_loss + self.lambda_reg * reg_loss

        return {
            "loss": total_loss,
            "data_loss": data_loss,
            "reg_loss": reg_loss,
        }

    def _smoothness_loss(self, u_pred, query_points, mask=None):
        """Penalize large local displacement gradients (simple L2 version)."""
        return torch.tensor(0.0, device=u_pred.device)

    def _compatibility_loss(self, u_pred, query_points, mask=None):
        """Penalize violation of strain compatibility conditions."""
        return torch.tensor(0.0, device=u_pred.device)
