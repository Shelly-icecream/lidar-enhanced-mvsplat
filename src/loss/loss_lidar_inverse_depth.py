"""Sparse LiDAR supervision of inverse depth before clamp and hard replacement."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .loss import Loss


def sparse_inverse_depth_loss(raw, lidar_depth, near, far, beta):
    if raw.shape[:-1] != lidar_depth.shape:
        raise ValueError("Raw inverse depth and LiDAR image shapes must agree.")
    raw = raw.float()
    lidar = lidar_depth.detach().to(device=raw.device, dtype=torch.float32)
    valid = torch.isfinite(lidar) & (lidar > 0)
    valid &= (lidar >= near[..., None, None]) & (lidar <= far[..., None, None])
    # Avoid reciprocals of zero/NaN even at masked-out locations.
    target = torch.where(valid, lidar, torch.ones_like(lidar)).reciprocal()[..., None]
    valid = valid[..., None].expand_as(raw) & torch.isfinite(raw)
    prediction = torch.where(valid, raw, torch.zeros_like(raw))
    target = torch.where(valid, target, torch.zeros_like(raw))
    penalty = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    return penalty.sum() / valid.sum().clamp_min(1)


@dataclass
class LossLidarInverseDepthCfg:
    weight: float = 1.0
    beta: float = 0.01


@dataclass
class LossLidarInverseDepthCfgWrapper:
    lidar_inverse_depth: LossLidarInverseDepthCfg


class LossLidarInverseDepth(
    Loss[LossLidarInverseDepthCfg, LossLidarInverseDepthCfgWrapper]
):
    def forward(self, prediction, batch, gaussians, global_step):
        raw = gaussians.raw_context_inverse_depth
        context = batch["context"]
        if raw is None or "lidar_depth" not in context:
            raise ValueError(
                "lidar_inverse_depth requires raw inverse depth and dataset.load_lidar_depth=true."
            )
        return self.cfg.weight * sparse_inverse_depth_loss(
            raw, context["lidar_depth"], context["near"], context["far"], self.cfg.beta
        )
