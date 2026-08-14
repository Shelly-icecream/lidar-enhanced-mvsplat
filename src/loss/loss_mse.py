from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    target_lidar_center_weight: float = 1.0
    target_lidar_neighbor_weight: float = 1.0


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    diagnostics: dict[str, Tensor]

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        delta = prediction.color - batch["target"]["image"]
        squared_error = delta.square()
        dynamic_mask = batch["target"].get("dynamic_mask")
        valid_mask = torch.ones_like(squared_error[:, :, :1])
        if dynamic_mask is not None:
            valid_mask = 1.0 - dynamic_mask.to(
                device=squared_error.device, dtype=squared_error.dtype
            )

        lidar_mask = batch["target"].get("lidar_mask")
        center_weight = float(self.cfg.target_lidar_center_weight)
        neighbor_weight = float(self.cfg.target_lidar_neighbor_weight)
        if center_weight < neighbor_weight or neighbor_weight < 1.0:
            raise ValueError(
                "Target LiDAR MSE weights must satisfy "
                "center_weight >= neighbor_weight >= 1."
            )

        pixel_weight = torch.ones_like(valid_mask)
        center_mask = None
        neighbor_mask = None
        if lidar_mask is not None and center_weight > 1.0:
            center_mask = (lidar_mask > 0.5).to(
                device=squared_error.device, dtype=squared_error.dtype
            ) * valid_mask
            b, v, _, h, w = center_mask.shape
            dilated_mask = F.max_pool2d(
                center_mask.reshape(b * v, 1, h, w),
                kernel_size=3,
                stride=1,
                padding=1,
            ).reshape(b, v, 1, h, w)
            neighbor_mask = (
                (dilated_mask - center_mask).clamp_min(0.0) * valid_mask
            )
            pixel_weight = (
                pixel_weight
                + (neighbor_weight - 1.0) * neighbor_mask
                + (center_weight - 1.0) * center_mask
            )

        weighted_valid_mask = pixel_weight * valid_mask
        denominator = (
            weighted_valid_mask.sum() * prediction.color.shape[2]
        ).clamp_min(1.0)
        loss = (squared_error * weighted_valid_mask).sum() / denominator

        self.diagnostics = {}
        if center_mask is not None and neighbor_mask is not None:
            channel_count = prediction.color.shape[2]

            def region_mse(mask: Tensor) -> Tensor:
                region_denominator = (mask.sum() * channel_count).clamp_min(1.0)
                return (squared_error * mask).sum() / region_denominator

            non_lidar_mask = valid_mask * (1.0 - center_mask)
            self.diagnostics = {
                "target_lidar_coverage": (
                    center_mask.sum() / valid_mask.sum().clamp_min(1.0)
                ).detach(),
                "mse_target_lidar": region_mse(center_mask).detach(),
                "mse_target_lidar_neighbor": region_mse(neighbor_mask).detach(),
                "mse_target_non_lidar": region_mse(non_lidar_mask).detach(),
            }

        return self.cfg.weight * loss
