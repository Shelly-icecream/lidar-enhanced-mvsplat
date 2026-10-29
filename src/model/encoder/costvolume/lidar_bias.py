"""Hard replacement of final camera Z-depth at valid sparse LiDAR pixels."""

import torch


def replace_depth_with_lidar(depths, lidar_depth, near, far):
    # depths: [B,V,H*W,surface,sample]; LiDAR: [B,V,H,W].
    if (
        lidar_depth.shape[:2] != depths.shape[:2]
        or lidar_depth.shape[-2] * lidar_depth.shape[-1] != depths.shape[2]
    ):
        raise ValueError("LiDAR depth must match context views and image resolution.")
    lidar = lidar_depth.to(device=depths.device, dtype=depths.dtype).detach()
    valid = torch.isfinite(lidar) & (lidar > 0)
    valid &= (lidar >= near[..., None, None]) & (lidar <= far[..., None, None])
    lidar = lidar.flatten(2)[..., None, None]
    valid = valid.flatten(2)[..., None, None]
    return torch.where(valid, lidar, depths)
