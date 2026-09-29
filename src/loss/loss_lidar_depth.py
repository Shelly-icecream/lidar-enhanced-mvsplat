from dataclasses import dataclass

import torch

from .loss import Loss


@dataclass
class LossLidarDepthCfg:
    weight: float = 1.0


@dataclass
class LossLidarDepthCfgWrapper:
    lidar_depth: LossLidarDepthCfg


class LossLidarDepth(Loss[LossLidarDepthCfg, LossLidarDepthCfgWrapper]):
    def forward(self, prediction, batch, gaussians, global_step):
        # Visual disparity before clamping or LiDAR replacement; keep its gradient.
        # Shape: [batch, view, depth_sample, height, width].
        visual = gaussians.visual_disparity
        context = batch["context"]
        depth = context["lidar_depth"].to(visual)
        mask = context["lidar_mask"].to(device=visual.device) > 0.5
        lidar = depth.clamp_min(1e-6).reciprocal()
        minimum = context["far"].to(visual).reciprocal()[..., None, None, None]
        maximum = context["near"].to(visual).reciprocal()[..., None, None, None]
        valid = mask & torch.isfinite(depth) & (depth > 1e-6)
        valid = valid & (lidar >= minimum) & (lidar <= maximum)
        # Select first so invalid NaN/Inf LiDAR values cannot pollute the loss.
        selected = valid.expand_as(visual)
        error = (visual[selected] - lidar.expand_as(visual)[selected]).abs()
        # Equal weight per valid pixel, averaging depth samples within a pixel.
        denominator = (valid.sum() * visual.shape[2]).clamp_min(1)
        return self.cfg.weight * error.sum() / denominator
