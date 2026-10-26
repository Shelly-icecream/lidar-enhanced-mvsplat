from dataclasses import dataclass
from itertools import combinations

import torch
import torch.nn.functional as F

from .loss import Loss


@dataclass
class LossCrossVisualDepthCfg:
    weight: float = 1.0
    beta: float = 0.01
    color_threshold: float = 0.1
    reprojection_threshold_px: float = 1.0
    relative_depth_threshold: float = 0.05
    print_every_n_steps: int = 50
    epsilon: float = 1e-6


@dataclass
class LossCrossVisualDepthCfgWrapper:
    cross_visual_depth: LossCrossVisualDepthCfg


class LossCrossVisualDepth(Loss[LossCrossVisualDepthCfg, LossCrossVisualDepthCfgWrapper]):
    """Symmetric consistency of context camera z-depths (normalized intrinsics)."""

    def direction(self, depth_a, depth_b, image_a, image_b, k_a, k_b, pose_a, pose_b,
                  dynamic_mask_a=None, dynamic_mask_b=None):
        # Float32 geometry also keeps inverses safe under mixed precision.
        depth_a, depth_b = depth_a.float(), depth_b.float()
        image_a, image_b = image_a.float(), image_b.float()
        k_a, k_b = k_a.float(), k_b.float()
        pose_a, pose_b = pose_a.float(), pose_b.float()
        batch, height, width = depth_a.shape
        eps = self.cfg.epsilon
        y, x = torch.meshgrid(
            (torch.arange(height, device=depth_a.device) + 0.5) / height,
            (torch.arange(width, device=depth_a.device) + 0.5) / width,
            indexing="ij",
        )
        xy = torch.stack((x, y), -1).reshape(-1, 2).expand(batch, -1, -1)
        raw_a = depth_a.flatten(1)
        safe_a = torch.where(torch.isfinite(raw_a) & (raw_a > eps), raw_a, torch.ones_like(raw_a))
        rays_a = torch.linalg.solve(k_a, torch.cat((xy, torch.ones_like(xy[..., :1])), -1).transpose(1, 2))
        points_a = rays_a * safe_a[:, None]
        a_to_b = torch.linalg.solve(pose_b, pose_a)
        points_b = a_to_b[:, :3, :3] @ points_a + a_to_b[:, :3, 3:4]
        z_b = points_b[:, 2]
        projected = k_b @ points_b
        xy_b = (projected[:, :2] / projected[:, 2:3].clamp_min(eps)).transpose(1, 2)
        # Invalid projections must never enter grid_sample as NaN/Inf.
        grid = (xy_b.nan_to_num(nan=-2., posinf=2., neginf=-2.) * 2 - 1).unsqueeze(2)
        valid_b_map = torch.isfinite(depth_b) & (depth_b > eps)
        safe_b_map = torch.where(valid_b_map, depth_b, torch.zeros_like(depth_b))
        sampled_b = F.grid_sample(safe_b_map[:, None], grid, align_corners=False)[:, 0, :, 0]
        sampled_color = F.grid_sample(image_b, grid, align_corners=False)[..., 0]
        color_a = image_a.flatten(2)
        residual = (z_b - sampled_b) / torch.maximum(z_b, sampled_b).clamp_min(eps).detach()

        with torch.no_grad():
            # Reconstruct using B's depth, then project back into A.
            rays_b = torch.linalg.solve(k_b, torch.cat((xy_b, torch.ones_like(xy_b[..., :1])), -1).transpose(1, 2))
            b_to_a = torch.linalg.solve(pose_a, pose_b)
            returned = b_to_a[:, :3, :3] @ (rays_b * sampled_b[:, None]) + b_to_a[:, :3, 3:4]
            back = k_a @ returned
            xy_back = (back[:, :2] / back[:, 2:3].clamp_min(eps)).transpose(1, 2)
            pixel_scale = xy.new_tensor((width, height))
            roundtrip = ((xy_back - xy) * pixel_scale).norm(dim=-1)
            # Require all contributing depth samples to be valid.
            support = F.grid_sample(valid_b_map[:, None].float(), grid, align_corners=False)[:, 0, :, 0]
            inside = ((xy_b >= xy.new_tensor((0.5 / width, 0.5 / height))) &
                      (xy_b <= xy.new_tensor((1 - 0.5 / width, 1 - 0.5 / height)))).all(-1)
            mask = (torch.isfinite(raw_a) & (raw_a > eps) & torch.isfinite(z_b) & (z_b > eps)
                    & inside & (support >= 1 - 1e-6) & (sampled_b > eps)
                    & (returned[:, 2] > eps) & torch.isfinite(residual)
                    & (residual.abs() <= self.cfg.relative_depth_threshold)
                    & (roundtrip <= self.cfg.reprojection_threshold_px)
                    & ((color_a - sampled_color).abs().mean(1) <= self.cfg.color_threshold))
            if dynamic_mask_a is not None:
                mask &= (dynamic_mask_a.to(depth_a.device) < 0.5).reshape(batch, -1)
            if dynamic_mask_b is not None:
                # Use the depth sampling grid and require every bilinear
                # contributor in the other context to be static.
                static_b = (dynamic_mask_b.to(depth_a.device) < 0.5).float()
                static_support = F.grid_sample(
                    static_b.reshape(batch, 1, height, width), grid,
                    align_corners=False,
                )[:, 0, :, 0]
                mask &= static_support >= 1 - 1e-6
        # Select before evaluating Smooth L1: invalid residuals cannot poison backward.
        selected = torch.where(mask, residual, torch.zeros_like(residual))
        penalty = F.smooth_l1_loss(selected, torch.zeros_like(selected), beta=self.cfg.beta, reduction="none")
        loss = penalty.sum() / mask.sum().clamp_min(1)
        return loss, mask.float().mean().detach()

    def forward(self, prediction, batch, gaussians, global_step):
        depth = gaussians.context_depth
        if depth is None:
            raise ValueError("cross_visual_depth requires encoder context_depth")
        context = batch["context"]
        dynamic_mask = context.get("dynamic_mask")
        losses, ratios = [], []
        with torch.autocast(device_type=depth.device.type, enabled=False):
            for a, b in combinations(range(depth.shape[1]), 2):
                for source, target in ((a, b), (b, a)):
                    loss, ratio = self.direction(
                        depth[:, source], depth[:, target],
                        context["image"][:, source], context["image"][:, target],
                        context["intrinsics"][:, source], context["intrinsics"][:, target],
                        context["extrinsics"][:, source], context["extrinsics"][:, target],
                        dynamic_mask[:, source] if dynamic_mask is not None else None,
                        dynamic_mask[:, target] if dynamic_mask is not None else None,
                    )
                    losses.append(loss)
                    ratios.append(ratio)
        result = self.cfg.weight * (torch.stack(losses).mean() if losses else depth.nan_to_num().sum() * 0)
        self.diagnostics = {"valid_ratio": torch.stack(ratios).mean() if ratios else depth.new_zeros(())}
        return result
