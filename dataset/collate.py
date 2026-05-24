"""Custom collate function for variable-length query point batches.

Pads query_points and u_gt to the maximum length in the batch,
and creates a boolean mask to identify valid entries.
"""
import torch
import numpy as np
from typing import List, Dict


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate a list of sample dicts into a batch.

    Pads variable-length query points to max_N across the batch.
    Creates query_mask to distinguish valid entries from padding.

    Args:
        batch: list of sample dicts, each with:
            'ref_img': [1, H, W] tensor
            'tar_img': [1, H, W] tensor
            'query_points': [N_i, 2] tensor
            'u_gt': [N_i, 2] tensor
            optional: 'roi_mask', 'deformation_mode', etc.

    Returns:
        batched dict as specified in the data interface.
    """
    # Stack images
    ref_imgs = torch.stack([s["ref_img"] for s in batch])
    tar_imgs = torch.stack([s["tar_img"] for s in batch])

    # Pad query points and displacements
    query_counts = [s["query_points"].shape[0] for s in batch]
    max_n = max(query_counts)

    B = len(batch)
    padded_query_pts = torch.zeros(B, max_n, 2)
    padded_u_gt = torch.zeros(B, max_n, 2)
    query_mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i, (sample, n) in enumerate(zip(batch, query_counts)):
        padded_query_pts[i, :n] = sample["query_points"]
        padded_u_gt[i, :n] = sample["u_gt"]
        query_mask[i, :n] = True

    result = {
        "ref_img": ref_imgs,
        "tar_img": tar_imgs,
        "query_points": padded_query_pts,
        "u_gt": padded_u_gt,
        "query_mask": query_mask,
    }

    # Pass through optional fields
    for key in ["roi_mask", "deformation_mode", "speckle_params", "sample_id"]:
        if key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([s[key] for s in batch])
            else:
                result[key] = [s[key] for s in batch]

    return result
