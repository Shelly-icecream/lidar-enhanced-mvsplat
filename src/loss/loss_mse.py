from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..geometry.projection import project
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    target_lidar_center_weight: float = 1.0


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    diagnostics: dict[str, Tensor]

    @staticmethod
    @torch.no_grad()
    def _project_context_lidar_gaussians(
        batch: BatchedExample,
        gaussians: Gaussians,
        target_height: int,
        target_width: int,
    ) -> Tensor | None:
        """Project t-1/t+1 LiDAR Gaussians into every target camera."""
        context_lidar_mask = batch["context"].get("lidar_mask")
        if context_lidar_mask is None:
            return None

        batch_size, context_views, _, context_height, context_width = (
            context_lidar_mask.shape
        )
        gaussians_per_context_pixel, remainder = divmod(
            gaussians.means.shape[1],
            context_views * context_height * context_width,
        )
        if remainder or gaussians_per_context_pixel == 0:
            raise ValueError(
                "Cannot map flattened Gaussians back to context pixels: "
                f"num_gaussians={gaussians.means.shape[1]}, "
                f"context_shape={(context_views, context_height, context_width)}."
            )

        selected = rearrange(
            context_lidar_mask > 0.5,
            "b v 1 h w -> b (v h w)",
        ).repeat_interleave(gaussians_per_context_pixel, dim=1)
        target_views = batch["target"]["extrinsics"].shape[1]
        projected_mask = torch.zeros(
            batch_size,
            target_views,
            1,
            target_height,
            target_width,
            device=gaussians.means.device,
            dtype=torch.bool,
        )

        for batch_index in range(batch_size):
            selected_means = gaussians.means[batch_index, selected[batch_index]]
            if selected_means.numel() == 0:
                continue
            for target_index in range(target_views):
                xy, in_front = project(
                    selected_means,
                    batch["target"]["extrinsics"][batch_index, target_index],
                    batch["target"]["intrinsics"][batch_index, target_index],
                )
                in_frame = (
                    in_front
                    & torch.isfinite(xy).all(dim=-1)
                    & (xy[:, 0] >= 0.0)
                    & (xy[:, 0] < 1.0)
                    & (xy[:, 1] >= 0.0)
                    & (xy[:, 1] < 1.0)
                )
                if not in_frame.any():
                    continue
                pixel_x = (xy[in_frame, 0] * target_width).long().clamp(
                    0, target_width - 1
                )
                pixel_y = (xy[in_frame, 1] * target_height).long().clamp(
                    0, target_height - 1
                )
                projected_mask[
                    batch_index, target_index, 0, pixel_y, pixel_x
                ] = True
        return projected_mask

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

        center_weight = float(self.cfg.target_lidar_center_weight)
        if center_weight < 1.0:
            raise ValueError("Projected LiDAR Gaussian weight must be at least 1.")

        pixel_weight = torch.ones_like(valid_mask)
        center_mask = None
        projected_context_lidar = None
        if center_weight > 1.0:
            projected_context_lidar = self._project_context_lidar_gaussians(
                batch,
                gaussians,
                squared_error.shape[-2],
                squared_error.shape[-1],
            )
        if projected_context_lidar is not None:
            center_mask = projected_context_lidar.to(
                device=squared_error.device, dtype=squared_error.dtype
            ) * valid_mask
            pixel_weight = (
                pixel_weight
                + (center_weight - 1.0) * center_mask
            )

        weighted_valid_mask = pixel_weight * valid_mask
        denominator = (
            weighted_valid_mask.sum() * prediction.color.shape[2]
        ).clamp_min(1.0)
        loss = (squared_error * weighted_valid_mask).sum() / denominator

        self.diagnostics = {}
        if center_mask is not None:
            channel_count = prediction.color.shape[2]

            def region_mse(mask: Tensor) -> Tensor:
                region_denominator = (mask.sum() * channel_count).clamp_min(1.0)
                return (squared_error * mask).sum() / region_denominator

            non_lidar_mask = valid_mask * (1.0 - center_mask)
            self.diagnostics = {
                "projected_context_lidar_coverage": (
                    center_mask.sum() / valid_mask.sum().clamp_min(1.0)
                ).detach(),
                "mse_projected_context_lidar": region_mse(center_mask).detach(),
                "mse_outside_projected_context_lidar": region_mse(
                    non_lidar_mask
                ).detach(),
            }

        return self.cfg.weight * loss
