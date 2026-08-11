import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel
class LidarTokenCrossAttention(nn.Module):
    """Predict local LiDAR-conditioned residuals for visual depth logits."""

    def __init__(
        self,
        num_depth_candidates: int,
        attention_dim: int,
        num_heads: int,
        radius: int,
        max_delta_logit: float,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError(
                "LiDAR attention_dim must be divisible by num_heads."
            )
        if radius < 0:
            raise ValueError("LiDAR cross-attention radius must be non-negative.")
        if max_delta_logit <= 0:
            raise ValueError("LiDAR max delta logit must be positive.")

        self.num_depth_candidates = num_depth_candidates
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.radius = radius
        self.kernel_size = 2 * radius + 1
        self.max_delta_logit = max_delta_logit

        # Raw normalized xy plus sin/cos at pi and 2*pi for each axis.
        position_dim = 10
        lidar_attribute_dim = 5
        self.query_norm = nn.LayerNorm(num_depth_candidates)
        self.query_proj = nn.Linear(num_depth_candidates, attention_dim)
        self.query_position_proj = nn.Linear(position_dim, attention_dim)
        self.query_stat_proj = nn.Linear(2, attention_dim)
        self.lidar_attribute_encoder = nn.Sequential(
            nn.Linear(lidar_attribute_dim, attention_dim),
            nn.SiLU(),
            nn.Linear(attention_dim, attention_dim),
        )
        self.lidar_position_proj = nn.Linear(position_dim, attention_dim)
        self.key_proj = nn.Linear(attention_dim, attention_dim)
        self.value_proj = nn.Linear(attention_dim, attention_dim)
        self.relative_position_bias = nn.Linear(2, num_heads, bias=False)
        self.delta_head = nn.Sequential(
            nn.Linear(attention_dim, attention_dim, bias=False),
            nn.GELU(),
            nn.Linear(
                attention_dim,
                num_depth_candidates,
                bias=False,
            ),
        )

        # The new branch starts as an exact identity for pretrained checkpoints.
        nn.init.zeros_(self.delta_head[-1].weight)


    @staticmethod
    def _position_encoding(
        xy: torch.Tensor,
    ) -> torch.Tensor:
        x, y = xy.unbind(dim=-1)
        fourier = torch.stack(
            (
                torch.sin(math.pi * x),
                torch.cos(math.pi * x),
                torch.sin(2.0 * math.pi * x),
                torch.cos(2.0 * math.pi * x),
                torch.sin(math.pi * y),
                torch.cos(math.pi * y),
                torch.sin(2.0 * math.pi * y),
                torch.cos(2.0 * math.pi * y),
            ),
            dim=-1,
        )
        return torch.cat((xy, fourier), dim=-1)

    @staticmethod
    def _normalized_grid(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(height * width, 2)

    def forward(
        self,
        visual_logits: torch.Tensor,
        lidar_disp: torch.Tensor,
        lidar_mask: torch.Tensor,
        visual_disp: torch.Tensor,
        visual_entropy: torch.Tensor,
        lidar_density: torch.Tensor,
        disp_min: torch.Tensor,
        disp_max: torch.Tensor,
    ) -> torch.Tensor:
        vb, candidates, height, width = visual_logits.shape
        if candidates != self.num_depth_candidates:
            raise ValueError(
                "Visual depth-logit channels do not match num_depth_candidates."
            )
        if lidar_mask.shape[-2:] != (height, width):
            raise ValueError(
                "LiDAR token grid must match the visual depth-logit grid."
            )

        valid = lidar_mask[:, 0] > 0.5
        has_lidar = valid.flatten(1).any(dim=1)
        if not valid.any():
            return visual_logits

        grid = self._normalized_grid(
            height,
            width,
            visual_logits.device,
            visual_logits.dtype,
        )
        position_encoding = self._position_encoding(grid)
        logit_tokens = visual_logits.flatten(2).transpose(1, 2)

        disp_scale = (disp_max - disp_min).clamp_min(1e-6)
        lidar_disp_norm = (lidar_disp - disp_min) / disp_scale
        visual_disp_norm = (visual_disp - disp_min) / disp_scale
        residual_norm = (lidar_disp - visual_disp) / disp_scale

        # Q_i = E_logit(L_i) + E_pos(x_i, y_i)
        #       + E_stat(expected_disparity_i, entropy_i).
        query_stats = torch.cat(
            (visual_disp_norm, visual_entropy),
            dim=1,
        ).flatten(2).transpose(1, 2)
        query_logit = self.query_proj(self.query_norm(logit_tokens))
        query_position = self.query_position_proj(position_encoding)[None]
        query_stats_encoded = self.query_stat_proj(query_stats)
        query = query_logit + query_position + query_stats_encoded
        with torch.no_grad():
            logit_norm = query_logit.norm(dim=-1).mean()
            position_norm = query_position.norm(dim=-1).mean()
            stat_norm = query_stats_encoded.norm(dim=-1).mean()
            combined_norm = query.norm(dim=-1).mean()
            visual_logits_detached = visual_logits.detach()
            visual_candidate_std = visual_logits_detached.std(
                dim=1,
                unbiased=False,
            ).mean()
            print(
                "[LiDAR Attention Query Diagnostics] "
                f"E_logit_norm={logit_norm.item():.6f}, "
                f"E_pos_norm={position_norm.item():.6f}, "
                f"E_stat_norm={stat_norm.item():.6f}, "
                f"Q_norm={combined_norm.item():.6f}, "
                f"visual_logits_min={visual_logits_detached.min().item():.6f}, "
                f"visual_logits_mean={visual_logits_detached.mean().item():.6f}, "
                f"visual_logits_max={visual_logits_detached.max().item():.6f}, "
                f"visual_logits_std={visual_logits_detached.std(unbiased=False).item():.6f}, "
                f"visual_candidate_std_mean={visual_candidate_std.item():.6f}, "
                f"entropy_min={visual_entropy.min().item():.6f}, "
                f"entropy_mean={visual_entropy.mean().item():.6f}, "
                f"entropy_max={visual_entropy.max().item():.6f}"
            )

        lidar_attributes = torch.cat(
            (
                lidar_disp_norm,
                visual_disp_norm,
                residual_norm,
                visual_entropy,
                lidar_density,
            ),
            dim=1,
        )
        window_area = self.kernel_size**2
        num_pixels = height * width
        local_attributes = F.unfold(
            lidar_attributes,
            kernel_size=self.kernel_size,
            padding=self.radius,
        ).reshape(vb, 5, window_area, num_pixels).permute(0, 3, 2, 1)
        local_valid = F.unfold(
            valid[:, None].to(dtype=visual_logits.dtype),
            kernel_size=self.kernel_size,
            padding=self.radius,
        ).transpose(1, 2) > 0.5

        position_map = position_encoding.transpose(0, 1).reshape(
            1,
            position_encoding.shape[-1],
            height,
            width,
        ).expand(vb, -1, -1, -1)
        local_positions = F.unfold(
            position_map,
            kernel_size=self.kernel_size,
            padding=self.radius,
        ).reshape(
            vb,
            position_encoding.shape[-1],
            window_area,
            num_pixels,
        ).permute(0, 3, 2, 1)

        lidar_tokens = self.lidar_attribute_encoder(local_attributes)
        lidar_tokens = lidar_tokens + self.lidar_position_proj(local_positions)
        key = self.key_proj(lidar_tokens).reshape(
            vb, num_pixels, window_area, self.num_heads, self.head_dim
        )
        value = self.value_proj(lidar_tokens).reshape(
            vb, num_pixels, window_area, self.num_heads, self.head_dim
        )
        query_heads = query.reshape(
            vb, num_pixels, self.num_heads, self.head_dim
        )


        offsets = torch.arange(
            -self.radius,
            self.radius + 1,
            device=visual_logits.device,
            dtype=visual_logits.dtype,
        )
        offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing="ij")
        relative_xy = torch.stack((offset_x, offset_y), dim=-1).reshape(
            window_area, 2
        ) / max(self.radius, 1)
        relative_bias = self.relative_position_bias(relative_xy).transpose(0, 1)

        attention_scores = torch.einsum(
            "bnhd,bnkhd->bnhk",
            query_heads,
            key,
        ) / math.sqrt(self.head_dim)
        attention_scores = attention_scores + relative_bias[None, None, :, :]
        attention_scores = attention_scores.masked_fill(
            ~local_valid[:, :, None, :],
            -1e4,
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights * local_valid[:, :, None, :].to(
            dtype=attention_weights.dtype
        )
        attention_weights = attention_weights / attention_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        attended = torch.einsum(
            "bnhk,bnkhd->bnhd",
            attention_weights,
            value,
        ).reshape(vb, num_pixels, self.attention_dim)

        delta_logits_raw = self.delta_head(attended)
        delta_logits_centered = (
            delta_logits_raw - delta_logits_raw.mean(dim=-1, keepdim=True)
        )
        delta_logits = self.max_delta_logit * torch.tanh(delta_logits_centered)
        has_local_lidar = local_valid.any(dim=-1, keepdim=True)
        delta_logits = delta_logits * has_local_lidar.to(delta_logits.dtype)
        with torch.no_grad():
            active_delta = delta_logits[has_local_lidar.expand_as(delta_logits)]
            active_centered = delta_logits_centered[
                has_local_lidar.expand_as(delta_logits_centered)
            ]
            if active_delta.numel() > 0:
                saturation_ratio = (
                    active_delta.abs() >= 0.9 * self.max_delta_logit
                ).float().mean()
                print(
                    "[LiDAR Attention Delta Diagnostics] "
                    f"pre_tanh_min={active_centered.min().item():.6f}, "
                    f"pre_tanh_mean={active_centered.mean().item():.6f}, "
                    f"pre_tanh_max={active_centered.max().item():.6f}, "
                    f"delta_min={active_delta.min().item():.6f}, "
                    f"delta_abs_mean={active_delta.abs().mean().item():.6f}, "
                    f"delta_max={active_delta.max().item():.6f}, "
                    f"max_delta_logit={self.max_delta_logit:.6f}, "
                    f"saturation_ratio={saturation_ratio.item():.6f}"
                )
        delta_logits = delta_logits.transpose(1, 2).reshape(
            vb, candidates, height, width
        )
        return visual_logits + delta_logits

def build_lidar_surface_prior(
    disp_candi_curr,
    lidar_disp_low,
    lidar_mask_low,
    sigma_disp,
    eps=1e-6,
):
    """Build the shared LiDAR response over inverse-depth candidates."""
    if sigma_disp <= 0:
        raise ValueError(
            f"sigma_disp must be positive, got {sigma_disp}."
        )

    surface_prior = torch.exp(
        -0.5
        * (
            (disp_candi_curr - lidar_disp_low)
            / max(float(sigma_disp), eps)
        ).square()
    )
    return surface_prior * lidar_mask_low


def build_lidar_visibility_prior(
    lidar_depth,
    lidar_mask,
    disp_candi_curr,
    target_hw,
    visual_depth_logits=None,
    lambda_surface=10.0,
    lambda_free=2.0,
    sigma_disp=0.32,
    free_margin=0.5,
    eps=1e-6,
):
    """
    Args:
        lidar_depth:
            [B, V, 1, H, W]，LiDAR 物理深度，单位为米。
        lidar_mask:
            [B, V, 1, H, W]，LiDAR 有效掩码。
        disp_candi_curr:
            [V*B, D, 1, 1]，逆深度候选。
        target_hw:
            (Hf, Wf)，与低分辨率 depth logits 相同。

    Returns:
        lidar_bias:
            [V*B, D, Hf, Wf]
        lidar_mask_low:
            [V*B, 1, Hf, Wf]
        lidar_disp_low:
            [V*B, 1, Hf, Wf]
    """

    lidar_depth = lidar_depth.to(
        device=disp_candi_curr.device,
        dtype=disp_candi_curr.dtype,
    )

    lidar_mask = lidar_mask.to(
        device=disp_candi_curr.device,
        dtype=disp_candi_curr.dtype,
    )

    Hf, Wf = target_hw

    # [B,V,1,H,W] -> [V*B,1,H,W]
    lidar_depth = rearrange(
        lidar_depth,
        "b v c h w -> (v b) c h w",
    )

    lidar_mask = rearrange(
        lidar_mask,
        "b v c h w -> (v b) c h w",
    )

    H, W = lidar_depth.shape[-2:]

    assert H % Hf == 0, (
        f"LiDAR height {H} cannot be evenly downsampled to {Hf}"
    )
    assert W % Wf == 0, (
        f"LiDAR width {W} cannot be evenly downsampled to {Wf}"
    )

    scale_h = H // Hf
    scale_w = W // Wf

    kernel_size = (scale_h, scale_w)
    stride = (scale_h, scale_w)

    # 有效 LiDAR 点
    valid_mask = (
        (lidar_mask > 0.5)
        & torch.isfinite(lidar_depth)
        & (lidar_depth > eps)
    )

    valid_mask_float = valid_mask.to(
        dtype=lidar_depth.dtype
    )

    # 原分辨率逆深度，无效像素置零
    lidar_disp = torch.where(
        valid_mask,
        1.0 / lidar_depth.clamp(min=eps),
        torch.zeros_like(lidar_depth),
    )

    # 区块内只要有点，低分辨率位置就有效
    lidar_mask_low = F.max_pool2d(
        valid_mask_float,
        kernel_size=kernel_size,
        stride=stride,
    )

    lidar_mask_low = (
        lidar_mask_low > 0.5
    ).to(dtype=lidar_depth.dtype)

    # 最大逆深度对应最近点
    lidar_disp_low = F.max_pool2d(
        lidar_disp,
        kernel_size=kernel_size,
        stride=stride,
    )
    # Fraction of valid LiDAR pixels in each low-resolution cell.
    lidar_cell_density = F.avg_pool2d(
        valid_mask_float,
        kernel_size=kernel_size,
        stride=stride,
    )

    # 还原物理深度，供自由空间项使用
    lidar_depth_low = torch.where(
        lidar_mask_low > 0.5,
        1.0 / lidar_disp_low.clamp(min=eps),
        torch.zeros_like(lidar_disp_low),
    )
    # Condition the two analytic-prior strengths on inverse depth. Normalizing
    # against the current candidate range keeps the input stable across near/far
    # settings. The predicted log offsets are bounded to a 1/4x--4x multiplier.
    disp_min = disp_candi_curr.amin(dim=1, keepdim=True)
    disp_max = disp_candi_curr.amax(dim=1, keepdim=True)
    normalized_lidar_disp = (
        (lidar_disp_low - disp_min)
        / (disp_max - disp_min).clamp(min=eps)
    ).clamp(0.0, 1.0)
    
    
    disp_range = (disp_max - disp_min).clamp(min=eps)

    normalized_entropy = torch.zeros_like(lidar_disp_low)
    normalized_residual = torch.zeros_like(lidar_disp_low)
    local_density = torch.zeros_like(lidar_disp_low)

    if visual_depth_logits is not None:
        # Visual statistics describe the actual coarse distribution (T=1).
        # Detaching prevents auxiliary statistics from changing the visual
        # predictor merely to manipulate uncertainty. Analytic-bias
        # temperature calibration is applied separately at the fusion site.
        visual_pdf = F.softmax(
            visual_depth_logits.detach(),
            dim=1,
        )
        visual_expected_disp = (
            visual_pdf * disp_candi_curr
        ).sum(dim=1, keepdim=True)
        entropy = -(
            visual_pdf * visual_pdf.clamp_min(eps).log()
        ).sum(dim=1, keepdim=True)
        normalized_entropy = entropy / math.log(visual_pdf.shape[1])

        top2 = visual_pdf.topk(k=2, dim=1).values
        inverse_margin = 1.0 - (top2[:, :1] - top2[:, 1:2])

        visual_variance = (
            visual_pdf
            * (disp_candi_curr - visual_expected_disp).square()
        ).sum(dim=1, keepdim=True)
        normalized_visual_std = (
            visual_variance.clamp_min(0.0).sqrt() / disp_range
        ).clamp(0.0, 1.0)

        local_density = F.avg_pool2d(
            lidar_cell_density,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        neighbor_disp_sum = F.avg_pool2d(
            lidar_disp_low,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        neighbor_count = F.avg_pool2d(
            lidar_mask_low,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        neighbor_disp_mean = (
            neighbor_disp_sum / neighbor_count.clamp_min(eps)
        )
        local_disp_disagreement = (
            (lidar_disp_low - neighbor_disp_mean).abs() / disp_range
        ).clamp(0.0, 1.0)
        normalized_residual = (
            (visual_expected_disp - lidar_disp_low).abs() / disp_range
        ).clamp(0.0, 1.0)
    lambda_surface_map = torch.full_like(
        lidar_disp_low, float(lambda_surface)
    )
    lambda_free_map = torch.full_like(
        lidar_disp_low, float(lambda_free)
    )
        
    # 可选：第一次调用时打印统计量
    if not getattr(
        build_lidar_visibility_prior,
        "_printed_downsample_statistics",
        False,
    ):
        with torch.no_grad():
            original_count = (
                valid_mask_float.flatten(1).sum(dim=1)
            )

            lowres_count = (
                lidar_mask_low.flatten(1).sum(dim=1)
            )

            point_to_cell_ratio = (
                lowres_count
                / original_count.clamp(min=1)
            )

            coverage_ratio = (
                lowres_count
                / float(Hf * Wf)
            )

            print(
                "[LiDAR Aggregation] "
                f"input_shape={tuple(lidar_depth.shape)}, "
                f"lowres_shape={tuple(lidar_mask_low.shape)}, "
                f"scale=({scale_h},{scale_w}), "
                f"original_per_view="
                f"{original_count.detach().cpu().tolist()}, "
                f"lowres_per_view="
                f"{lowres_count.detach().cpu().tolist()}, "
                f"point_to_cell_ratio="
                f"{point_to_cell_ratio.detach().cpu().tolist()}, "
                f"coverage_per_view="
                f"{coverage_ratio.detach().cpu().tolist()}"
            )

        build_lidar_visibility_prior._printed_downsample_statistics = True

    # 候选逆深度 [V*B,D,1,1]
    disp_candi = disp_candi_curr

    # 候选物理深度 [V*B,D,1,1]
    depth_candi = 1.0 / disp_candi.clamp(min=eps)

    # LiDAR 表面吸引项
    surface_prior = build_lidar_surface_prior(
        disp_candi_curr=disp_candi,
        lidar_disp_low=lidar_disp_low,
        lidar_mask_low=lidar_mask_low,
        sigma_disp=sigma_disp,
        eps=eps,
    )

    # LiDAR 表面前方的自由空间抑制项
    free_prior = (
        depth_candi
        < (lidar_depth_low - free_margin)
    ).to(dtype=lidar_depth.dtype)

    free_prior = (
        free_prior * lidar_mask_low
    )

    lidar_bias = (
        lambda_surface_map * surface_prior
        - lambda_free_map * free_prior
    )

    return (
        lidar_bias,
        lidar_mask_low,
        lidar_disp_low,
        lambda_surface_map,
        lambda_free_map,
        normalized_entropy,
        normalized_residual,
        local_density,
        surface_prior,
    )

def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    warp_padding_mode="zeros",
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    warped_feature = F.grid_sample(
        feature1,
        grid.view(b, d * h, w, 2),
        mode="bilinear",
        padding_mode=warp_padding_mode,
        align_corners=True,
    ).view(
        b, c, d, h, w
    )  # [B, C, D, H, W]

    return warped_feature


def prepare_feat_proj_data_lists(
    features, intrinsics, extrinsics, near, far, num_samples
):
    # prepare features
    b, v, _, h, w = features.shape

    feat_lists = []
    pose_curr_lists = []
    init_view_order = list(range(v))
    feat_lists.append(rearrange(features, "b v ... -> (v b) ..."))  # (vxb c h w)
    for idx in range(1, v):
        cur_view_order = init_view_order[idx:] + init_view_order[:idx]
        cur_feat = features[:, cur_view_order]
        feat_lists.append(rearrange(cur_feat, "b v ... -> (v b) ..."))  # (vxb c h w)

        # calculate reference pose
        # NOTE: not efficient, but clearer for now
        if v > 2:
            cur_ref_pose_to_v0_list = []
            for v0, v1 in zip(init_view_order, cur_view_order):
                cur_ref_pose_to_v0_list.append(
                    extrinsics[:, v1].clone().detach().inverse()
                    @ extrinsics[:, v0].clone().detach()
                )
            cur_ref_pose_to_v0s = torch.cat(cur_ref_pose_to_v0_list, dim=0)  # (vxb c h w)
            pose_curr_lists.append(cur_ref_pose_to_v0s)
    
    # get 2 views reference pose
    # NOTE: do it in such a way to reproduce the exact same value as reported in paper
    if v == 2:
        pose_ref = extrinsics[:, 0].clone().detach()
        pose_tgt = extrinsics[:, 1].clone().detach()
        pose = pose_tgt.inverse() @ pose_ref
        pose_curr_lists = [torch.cat((pose, pose.inverse()), dim=0),]

    # unnormalized camera intrinsic
    intr_curr = intrinsics[:, :, :3, :3].clone().detach()  # [b, v, 3, 3]
    intr_curr[:, :, 0, :] *= float(w)
    intr_curr[:, :, 1, :] *= float(h)
    intr_curr = rearrange(intr_curr, "b v ... -> (v b) ...", b=b, v=v)  # [vxb 3 3]

    # prepare depth bound (inverse depth) [v*b, d]
    min_depth = rearrange(1.0 / far.clone().detach(), "b v -> (v b) 1")
    max_depth = rearrange(1.0 / near.clone().detach(), "b v -> (v b) 1")
    depth_candi_curr = (
        min_depth
        + torch.linspace(0.0, 1.0, num_samples).unsqueeze(0).to(min_depth.device)
        * (max_depth - min_depth)
    ).type_as(features)
    depth_candi_curr = repeat(depth_candi_curr, "vb d -> vb d () ()")  # [vxb, d, 1, 1]
    return feat_lists, intr_curr, pose_curr_lists, depth_candi_curr


class DepthPredictorMultiView(nn.Module):
    """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
    keep this in mind when performing any operation related to the view dim"""

    def set_lidar_cross_attention_enabled(self, enabled: bool) -> None:
        self.use_lidar_cross_attention = bool(enabled)
        if enabled and self.lidar_cross_attention is None:
            self.lidar_cross_attention = LidarTokenCrossAttention(
                num_depth_candidates=self.num_depth_candidates,
                attention_dim=self.lidar_cross_attention_dim,
                num_heads=self.lidar_cross_attention_heads,
                radius=self.lidar_cross_attention_radius,
                max_delta_logit=self.lidar_cross_attention_max_delta_logit,
            )
        elif not enabled:
            self.lidar_cross_attention = None


    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        wo_depth_refine=False,
        wo_cost_volume=False,
        wo_cost_volume_refine=False,
        
        use_lidar_bias=False,
        use_lidar_coarse_loss=False,
        use_lidar_refine_loss=False,
        use_lidar_cross_attention=False,
        lidar_cross_attention_dim=128,
        lidar_cross_attention_heads=4,
        lidar_cross_attention_radius=4,
        lidar_cross_attention_max_delta_logit=15.0,
        lidar_cross_attention_inference_mode="auto",
        lidar_lambda_surface=10.0,
        lidar_lambda_free=2.0,
        lidar_sigma_disp=0.12,
        lidar_free_margin=0.5,
        lidar_temperature=5.0,
        **kwargs,
    ):
        super(DepthPredictorMultiView, self).__init__()
        self.num_depth_candidates = num_depth_candidates
        self.regressor_feat_dim = costvolume_unet_feat_dim
        self.upscale_factor = upscale_factor
        # ablation settings
        # Table 3: base
        self.wo_depth_refine = wo_depth_refine
        # Table 3: w/o cost volume
        self.wo_cost_volume = wo_cost_volume
        # Table 3: w/o U-Net
        self.wo_cost_volume_refine = wo_cost_volume_refine
        self.use_lidar_bias = use_lidar_bias
        self.use_lidar_coarse_loss = use_lidar_coarse_loss
        self.use_lidar_refine_loss = use_lidar_refine_loss
        self.use_lidar_cross_attention = use_lidar_cross_attention
        self.lidar_cross_attention_dim = lidar_cross_attention_dim
        self.lidar_cross_attention_heads = lidar_cross_attention_heads
        self.lidar_cross_attention_radius = lidar_cross_attention_radius
        self.lidar_cross_attention_max_delta_logit = (
            lidar_cross_attention_max_delta_logit
        )
        if lidar_cross_attention_inference_mode not in {"auto", "on", "off"}:
            raise ValueError(
                "lidar_cross_attention_inference_mode must be one of "
                "{'auto', 'on', 'off'}."
            )
        self.lidar_cross_attention_inference_mode = (
            lidar_cross_attention_inference_mode
        )
        if self.use_lidar_cross_attention and self.wo_cost_volume_refine:
            raise ValueError(
                "LiDAR cross-attention requires cost-volume U-Net refinement."
            )
        self.lidar_lambda_surface = lidar_lambda_surface
        self.lidar_lambda_free = lidar_lambda_free
        self.lidar_sigma_disp = lidar_sigma_disp
        self.lidar_free_margin = lidar_free_margin
        self.lidar_temperature = lidar_temperature
        if self.lidar_temperature <= 0:
            raise ValueError(
                "lidar_temperature must be positive, "
                f"got {self.lidar_temperature}."
            )
        self.lidar_cross_attention = None
        if self.use_lidar_cross_attention:
            self.set_lidar_cross_attention_enabled(True)
        self.lidar_parameter_diagnostics = {}
        
        # 用于统计整个测试集上的 LiDAR 三阶段误差
        self.lidar_diag = {
            "num_points": 0,
            "vis_error_sum": 0.0,
            "bias_error_sum": 0.0,
            "refine_delta_sum": 0.0,
            "refine_improve_count": 0,
            "refine_worsen_count": 0,
            "final_depth_num_points": 0,
            "final_depth_abs_error_sum": 0.0,
            "final_depth_abs_rel_sum": 0.0,
            "final_depth_sq_error_sum": 0.0,
        }

        # Cost volume refinement: 2D U-Net
        input_channels = feature_channels if wo_cost_volume else (num_depth_candidates + feature_channels)
        channels = self.regressor_feat_dim
        if wo_cost_volume_refine:
            self.corr_project = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            modules = [
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1,
                    attention_resolutions=costvolume_unet_attn_res,
                    channel_mult=costvolume_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=True,
                ),
                nn.Conv2d(channels, num_depth_candidates, 3, 1, 1)
            ]
            self.corr_refine_net = nn.Sequential(*modules)
            # cost volume u-net skip connection
            self.regressor_residual = nn.Conv2d(
                input_channels, num_depth_candidates, 1, 1, 0
            )

        # Depth estimation: project features to get softmax based coarse depth
        self.depth_head_lowres = nn.Sequential(
            nn.Conv2d(num_depth_candidates, num_depth_candidates * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_depth_candidates * 2, num_depth_candidates, 3, 1, 1),
        )

        # CNN-based feature upsampler
        proj_in_channels = feature_channels + feature_channels
        upsample_out_channels = feature_channels
        self.upsampler = nn.Sequential(
            nn.Conv2d(proj_in_channels, upsample_out_channels, 3, 1, 1),
            nn.Upsample(
                scale_factor=upscale_factor,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
        self.proj_feature = nn.Conv2d(
            upsample_out_channels, depth_unet_feat_dim, 3, 1, 1
        )

        # Depth refinement: 2D U-Net
        input_channels = 3 + depth_unet_feat_dim + 1 + 1
        channels = depth_unet_feat_dim
        if wo_depth_refine:  # for ablations
            self.refine_unet = nn.Conv2d(input_channels, channels, 3, 1, 1)
        else:
            self.refine_unet = nn.Sequential(
                nn.Conv2d(input_channels, channels, 3, 1, 1),
                nn.GroupNorm(4, channels),
                nn.GELU(),
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=1, 
                    attention_resolutions=depth_unet_attn_res,
                    channel_mult=depth_unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=True,
                    num_frames=num_views,
                    use_cross_view_self_attn=True,
                ),
            )

        # Gaussians prediction: covariance, color
        gau_in = depth_unet_feat_dim + 3 + feature_channels
        self.to_gaussians = nn.Sequential(
            nn.Conv2d(gau_in, gaussian_raw_channels * 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(
                gaussian_raw_channels * 2, gaussian_raw_channels, 3, 1, 1
            ),
        )

        # Gaussians prediction: centers, opacity
        if not wo_depth_refine:
            channels = depth_unet_feat_dim
            disps_models = [
                nn.Conv2d(channels, channels * 2, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channels * 2, gaussians_per_pixel * 2, 3, 1, 1),
            ]
            self.to_disparity = nn.Sequential(*disps_models)

    def forward(
        self,
        features,
        intrinsics,
        extrinsics,
        near,
        far,
        gaussians_per_pixel=1,
        deterministic=True,
        extra_info=None,
        cnn_features=None,
        lidar_depth=None,
        lidar_mask=None,
    ):
        """IMPORTANT: this model is in (v b), NOT (b v), due to some historical issues.
        keep this in mind when performing any operation related to the view dim"""
        extra_info = {} if extra_info is None else extra_info
        
        # format the input
        b, v, c, h, w = features.shape
        feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = (
            prepare_feat_proj_data_lists(
                features,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=self.num_depth_candidates,
            )
        )
        if cnn_features is not None:
            cnn_features = rearrange(cnn_features, "b v ... -> (v b) ...")

        # cost volume constructions
        feat01 = feat_comb_lists[0]
        if self.wo_cost_volume:
            raw_correlation_in = feat01
        else:
            raw_correlation_in_lists = []
            for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
                # sample feat01 from feat10 via camera projection
                feat01_warped = warp_with_pose_depth_candidates(
                    feat10,
                    intr_curr,
                    pose_curr,
                    1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                    warp_padding_mode="zeros",
                )  # [B, C, D, H, W]
                # calculate similarity
                raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(
                    1
                ) / (
                    c**0.5
                )  # [vB, D, H, W]
                raw_correlation_in_lists.append(raw_correlation_in)
            # average all cost volumes
            raw_correlation_in = torch.mean(
                torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False
            )  # [vxb d, h, w]
            raw_correlation_in = torch.cat((raw_correlation_in, feat01), dim=1)

        # refine cost volume via 2D u-net
        corr_feature = None
        if self.wo_cost_volume_refine:
            raw_correlation = self.corr_project(raw_correlation_in)
        else:
            # Keep the pretrained Sequential parameter names intact while
            # exposing the latent U-Net feature for optional LiDAR attention.
            corr_feature = raw_correlation_in
            for module in list(self.corr_refine_net.children())[:-1]:
                corr_feature = module(corr_feature)
            raw_correlation = self.corr_refine_net[-1](corr_feature)
            # apply skip connection
            raw_correlation = raw_correlation + self.regressor_residual(
                raw_correlation_in
            )
        depth_logits_vis = self.depth_head_lowres(raw_correlation)  
        pdf_vis = F.softmax(depth_logits_vis, dim=1)
        coarse_disps_vis = (disp_candi_curr * pdf_vis).sum(dim=1, keepdim=True)
        coarse_before_attention = coarse_disps_vis

        lidar_coarse_loss = None
        lidar_refine_loss = None
        lidar_mask_low = None
        lidar_disp_low = None
        
        has_lidar = lidar_depth is not None and lidar_mask is not None
        need_lidar = has_lidar and (
            self.use_lidar_bias
            or self.use_lidar_coarse_loss
            or self.use_lidar_refine_loss
            or self.use_lidar_cross_attention
        )

        if need_lidar:
            (
                lidar_bias,
                lidar_mask_low,
                lidar_disp_low,
                lambda_surface_map,
                lambda_free_map,
                visual_entropy,
                visual_lidar_residual,
                lidar_local_density,
                lidar_surface_prior,
            ) = build_lidar_visibility_prior(
                lidar_depth=lidar_depth,              # [B,V,1,H,W]
                lidar_mask=lidar_mask,                # [B,V,1,H,W]
                disp_candi_curr=disp_candi_curr,      # [v*b,D,1,1]
                target_hw=depth_logits_vis.shape[-2:],    # (h,w)
                visual_depth_logits=depth_logits_vis,
                lambda_surface=self.lidar_lambda_surface,
                lambda_free=self.lidar_lambda_free,
                sigma_disp=self.lidar_sigma_disp,
                free_margin=self.lidar_free_margin,
            )
            if self.use_lidar_cross_attention:
                if self.lidar_cross_attention is None:
                    raise RuntimeError(
                        "LiDAR cross-attention module is not initialized."
                    )
                disp_min_low = disp_candi_curr.amin(
                    dim=1,
                    keepdim=True,
                )
                disp_max_low = disp_candi_curr.amax(
                    dim=1,
                    keepdim=True,
                )

                depth_logits_vis = self.lidar_cross_attention(
                    visual_logits=depth_logits_vis,
                    lidar_disp=lidar_disp_low,
                    lidar_mask=lidar_mask_low,
                    visual_disp=coarse_disps_vis,
                    visual_entropy=visual_entropy,
                    lidar_density=lidar_local_density,
                    disp_min=disp_min_low,
                    disp_max=disp_max_low,
                )
                pdf_vis = F.softmax(depth_logits_vis, dim=1)
                coarse_disps_vis = (
                    disp_candi_curr * pdf_vis
                ).sum(dim=1, keepdim=True)
            mask = lidar_mask_low.bool()

            error_before = (
                coarse_before_attention.detach() - lidar_disp_low
            ).abs()
            error_after = (
                coarse_disps_vis - lidar_disp_low
            ).abs()
            if (
                self.use_lidar_cross_attention
                and mask.any()
            ):
                with torch.no_grad():
                    valid_error_before = error_before[mask]
                    valid_error_after = error_after[mask]
                    error_improvement = (
                        valid_error_before - valid_error_after
                    )
                    print(
                        "[LiDAR Cross-Attention Diagnostics] "
                        f"E_before="
                        f"{valid_error_before.mean().item():.8f}, "
                        f"E_after="
                        f"{valid_error_after.mean().item():.8f}, "
                        f"gain="
                        f"{error_improvement.mean().item():.8f}, "
                        f"improve_ratio="
                        f"{(error_improvement > 0).float().mean().item():.4f}, "
                        f"worsen_ratio="
                        f"{(error_improvement < 0).float().mean().item():.4f}"
                    )
            # bias only changes the forward depth logits.
            if self.use_lidar_bias:
                # Soften visual logits only at valid LiDAR cells, then add the
                # analytic surface/free-space bias. Non-LiDAR cells retain the
                # original visual logits exactly.
                depth_logits_calibrated = (
                    depth_logits_vis * (1.0 - lidar_mask_low) + (depth_logits_vis / self.lidar_temperature) * lidar_mask_low
             )
                depth_logits_lidar = depth_logits_calibrated + lidar_bias
            else:
                depth_logits_lidar = depth_logits_vis
            # loss only uses pure visual prediction before LiDAR bias.
            if self.use_lidar_coarse_loss and mask.any():
                if self.use_lidar_cross_attention:
                    # Supervise only the expected coarse disparity. Candidate
                    # range checks remain because out-of-range LiDAR cannot be
                    # represented by the current plane-sweep hypotheses.
                    candidate_disp_min = disp_candi_curr.amin(
                        dim=1,
                        keepdim=True,
                    )
                    candidate_disp_max = disp_candi_curr.amax(
                        dim=1,
                        keepdim=True,
                    )
                    candidate_mask = (
                        mask
                        & (lidar_disp_low >= candidate_disp_min)
                        & (lidar_disp_low <= candidate_disp_max)
                    )
                    candidate_keep_ratio = (
                        candidate_mask.sum().float()
                        / mask.sum().float().clamp_min(1.0)
                    )
                    if candidate_mask.any():
                        lidar_disparity_loss = error_after[
                            candidate_mask
                        ].mean()
                    else:
                        lidar_disparity_loss = error_after.new_zeros(())
                    print(
                        "[LiDAR Coarse Loss] "
                        f"candidate_keep_ratio="
                        f"{candidate_keep_ratio.detach().item():.4f} "
                        f"({candidate_mask.sum().item()}/"
                        f"{mask.sum().item()}), "
                        f"L_disp={lidar_disparity_loss.detach().item():.8f}"
                    )
                    lidar_coarse_loss = lidar_disparity_loss 
                else:
                    # Preserve the legacy expected-disparity loss when
                    # cross-attention is disabled.
                    lidar_coarse_loss = error_after[mask].mean()
        else:
            depth_logits_lidar = depth_logits_vis 

        # softmax to get coarse depth and density
        pdf = F.softmax(depth_logits_lidar, dim=1)  # [v*b, D, h, w]
        
        coarse_disps = (disp_candi_curr * pdf).sum(
            dim=1, keepdim=True
        )  # (vb, 1, h, w)
        # ============================================================
        # 低分辨率 LiDAR Bias 诊断
        # 比较：
        # 1. 纯视觉 coarse disparity
        # 2. 加入 LiDAR bias 后的 coarse disparity
        # ============================================================
        if (
            need_lidar
            and self.use_lidar_bias
            and lidar_mask_low is not None
            and lidar_disp_low is not None
        ):
            with torch.no_grad():
                valid_low = lidar_mask_low > 0.5

                if valid_low.any():
                    valid_parameter_cells = valid_low[:, :1]
                    surface_values = lambda_surface_map[valid_parameter_cells]
                    free_values = lambda_free_map[valid_parameter_cells]
                    entropy_values = visual_entropy[
                        valid_parameter_cells
                    ]
                    residual_values = visual_lidar_residual[
                        valid_parameter_cells
                    ]
                    density_values = lidar_local_density[
                        valid_parameter_cells
                    ]
                    bias_std_d = lidar_bias.std(dim=1, unbiased=False)
                    visual_logits_std_d = depth_logits_vis.std(
                        dim=1, unbiased=False
                    )
                    valid_2d = valid_low[:, 0]
                    mean_bias_std = bias_std_d[valid_2d].mean()
                    mean_visual_logits_std = visual_logits_std_d[
                        valid_2d
                    ].mean()
                    bias_to_visual_std_ratio = (
                        mean_bias_std
                        / mean_visual_logits_std.clamp(min=1e-6)
                    )
                    tempered_visual_logits = (
                        depth_logits_vis / self.lidar_temperature
                    )
                    top2_logits = tempered_visual_logits.topk(
                        k=2,
                        dim=1,
                    ).values
                    top2_margin = top2_logits[:, :1] - top2_logits[:, 1:2]

                    visual_candidate_index = tempered_visual_logits.argmax(
                        dim=1,
                        keepdim=True,
                    )
                    lidar_candidate_index = (
                        disp_candi_curr - lidar_disp_low
                    ).abs().argmin(dim=1, keepdim=True)
                    bias_at_visual_candidate = lidar_bias.gather(
                        1,
                        visual_candidate_index,
                    )
                    bias_at_lidar_candidate = lidar_bias.gather(
                        1,
                        lidar_candidate_index,
                    )
                    bias_advantage = (
                        bias_at_lidar_candidate - bias_at_visual_candidate
                    )
                    self.lidar_parameter_diagnostics = {
                        "lambda_surface_mean": surface_values.mean().detach(),
                        "lambda_surface_min": surface_values.min().detach(),
                        "lambda_surface_max": surface_values.max().detach(),
                        "lambda_free_mean": free_values.mean().detach(),
                        "lambda_free_min": free_values.min().detach(),
                        "lambda_free_max": free_values.max().detach(),
                        "visual_entropy_mean": (
                            entropy_values.mean().detach()
                        ),
                        "visual_lidar_residual_mean": (
                            residual_values.mean().detach()
                        ),
                        "lidar_local_density_mean": (
                            density_values.mean().detach()
                        ),
                        "bias_std_d": mean_bias_std.detach(),
                        "visual_logits_std_d": mean_visual_logits_std.detach(),
                        "bias_to_visual_std_ratio": (
                            bias_to_visual_std_ratio.detach()
                        ),
                    }
                    # 纯视觉 coarse disparity 与 LiDAR 的误差
                    err_vis_low = (
                        coarse_disps_vis - lidar_disp_low
                    ).abs()[valid_low]

                    # 加 bias 后 coarse disparity 与 LiDAR 的误差
                    err_bias_low = (
                        coarse_disps - lidar_disp_low
                    ).abs()[valid_low]

                    # 误差得到了多大改善
                    gain=(err_vis_low.mean() - err_bias_low.mean()).item()
                    
                    improve_ratio_low = (
                        err_bias_low < err_vis_low
                    ).float().mean()

                    worsen_ratio_low = (
                        err_bias_low > err_vis_low
                    ).float().mean()

                    print(
                        "[LiDAR Lowres Bias Diagnostics] "
                        f"valid_cells={int(valid_low.sum().item())}, "
                        f"E_visual={err_vis_low.mean().item():.8f}, "
                        f"E_bias={err_bias_low.mean().item():.8f}, "
                        f"gain={gain:.8f}, "
                        f"improve_ratio={improve_ratio_low.item():.4f}, "
                        f"worsen_ratio={worsen_ratio_low.item():.4f}, "
                        )
        pdf_max = torch.max(pdf, dim=1, keepdim=True)[0]  # argmax
        pdf_max = F.interpolate(pdf_max, scale_factor=self.upscale_factor)
        fullres_disps = F.interpolate(
            coarse_disps,
            scale_factor=self.upscale_factor,
            mode="bilinear",
            align_corners=True,
        )

        # depth refinement
        proj_feat_in_fullres = self.upsampler(torch.cat((feat01, cnn_features), dim=1))
        proj_feature = self.proj_feature(proj_feat_in_fullres)
        refine_out = self.refine_unet(torch.cat(
            (
                extra_info["images"],
                proj_feature,
                fullres_disps,
                pdf_max,
            ),
            dim=1,
        ))

        # gaussians head
        raw_gaussians_in = [refine_out,
                            extra_info["images"], proj_feat_in_fullres]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
        raw_gaussians = rearrange(
            raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
        )

        if self.wo_depth_refine:
            densities = repeat(
                pdf_max,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )
            depths = 1.0 / fullres_disps
            depths = repeat(
                depths,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )
        else:
            # delta fine depth and density
            delta_disps_density = self.to_disparity(refine_out)
            delta_disps, raw_densities = delta_disps_density.split(
                gaussians_per_pixel, dim=1
            )

            # combine coarse and fine info and match shape
            densities = repeat(
                F.sigmoid(raw_densities),
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )

            disp_min = 1.0 / rearrange(
                far, "b v -> (v b) () () ()"
            )
            disp_max = 1.0 / rearrange(
                near, "b v -> (v b) () () ()"
            )
            
            # LiDAR-biased coarse disparities act as hard anchors.  Let the
            # refinement head update pixels without LiDAR support, but do not
            # let its residual overwrite disparities in LiDAR-supported cells.
            if self.use_lidar_bias and lidar_mask_low is not None:
                lidar_support_full = F.interpolate(
                    lidar_mask_low.float(),
                    size=delta_disps.shape[-2:],
                    mode="nearest",
                ).to(device=delta_disps.device, dtype=delta_disps.dtype)
                raw_fine_disps = (
                    fullres_disps
                    + (1.0 - lidar_support_full) * delta_disps
                )
            else:
                raw_fine_disps = fullres_disps + delta_disps
            fine_disps = raw_fine_disps.clamp(
                min=disp_min,
                max=disp_max,
            )

            # Always report refinement health when LiDAR supervision is
            # available.  Keep the loss itself gated below so diagnostics do
            # not change the training objective.
            if lidar_depth is not None and lidar_mask is not None:
                lidar_depth_full = rearrange(
                    lidar_depth,
                    "b v c h w -> (v b) c h w",
                ).to(
                    device=fine_disps.device,
                    dtype=fine_disps.dtype,
                )

                lidar_mask_full = rearrange(
                    lidar_mask,
                    "b v c h w -> (v b) c h w",
                ).to(device=fine_disps.device)

                valid = (
                    (lidar_mask_full > 0.5)
                    & torch.isfinite(lidar_depth_full)
                    & (lidar_depth_full > 1e-6)
                )

                if valid.any():
                    with torch.no_grad():
                        raw_final_disp = raw_fine_disps[:, :1]
                        final_disp = fine_disps[:, :1]
                        coarse_disp = fullres_disps[:, :1]
                        delta_disp = delta_disps[:, :1]
                        final_depth = 1.0 / final_disp.clamp_min(1e-6)

                        lower_saturation_ratio = (
                            (raw_final_disp <= disp_min)[valid]
                            .float()
                            .mean()
                        )
                        upper_saturation_ratio = (
                            (raw_final_disp >= disp_max)[valid]
                            .float()
                            .mean()
                        )

                        print(
                            "[Refine diagnostics] "
                            f"lower_saturation_ratio="
                            f"{lower_saturation_ratio.item():.6f}, "
                            f"upper_saturation_ratio="
                            f"{upper_saturation_ratio.item():.6f}, "
                            f"coarse_mean="
                            f"{coarse_disp[valid].mean().item():.6f}, "
                            f"delta_mean="
                            f"{delta_disp[valid].mean().item():.6f}, "
                            f"delta_abs_mean="
                            f"{delta_disp[valid].abs().mean().item():.6f}, "
                            f"delta_min="
                            f"{delta_disp[valid].min().item():.6f}, "
                            f"delta_max="
                            f"{delta_disp[valid].max().item():.6f}, "
                            f"raw_min="
                            f"{raw_final_disp[valid].min().item():.6f}, "
                            f"raw_mean="
                            f"{raw_final_disp[valid].mean().item():.6f}, "
                            f"raw_max="
                            f"{raw_final_disp[valid].max().item():.6f}, "
                            f"final_depth_min="
                            f"{final_depth[valid].min().item():.6f}, "
                            f"final_depth_mean="
                            f"{final_depth[valid].mean().item():.6f}, "
                            f"final_depth_max="
                            f"{final_depth[valid].max().item():.6f}"
                        )
 
                if self.use_lidar_refine_loss and valid.any():
                    lidar_disp_full = 1.0 / lidar_depth_full.clamp(min=1e-6)
                    final_disp_for_loss = raw_fine_disps[:, :1]
                    lidar_data_loss = (
                        final_disp_for_loss - lidar_disp_full
                    ).abs()[valid].mean()
                    lower_violation = F.relu(
                        disp_min - final_disp_for_loss
                    )
                    upper_violation = F.relu(
                        final_disp_for_loss - disp_max
                    )
                    lidar_bounds_loss = (
                        lower_violation + upper_violation
                    )[valid].mean()
                    lidar_refine_loss = (
                        lidar_data_loss + 1.0 * lidar_bounds_loss
                    )

            # Accumulate full-resolution final-depth metrics over valid LiDAR pixels.
            # These are evaluation diagnostics and do not participate in backpropagation.
            if has_lidar and not self.training:
                with torch.no_grad():
                    lidar_depth_full = rearrange(
                        lidar_depth,
                        "b v c h w -> (v b) c h w",
                    ).to(
                        device=fine_disps.device,
                        dtype=fine_disps.dtype,
                    )
                    lidar_mask_full = rearrange(
                        lidar_mask,
                        "b v c h w -> (v b) c h w",
                    ).to(device=fine_disps.device)

                    final_depth = 1.0 / fine_disps[:, :1].clamp(min=1e-6)
                    valid_depth = (
                        (lidar_mask_full > 0.5)
                        & torch.isfinite(lidar_depth_full)
                        & (lidar_depth_full > 1e-6)
                        & torch.isfinite(final_depth)
                        & (final_depth > 0)
                    )

                    if valid_depth.any():
                        target_depth = lidar_depth_full[valid_depth]
                        predicted_depth = final_depth[valid_depth]
                        abs_error = (predicted_depth - target_depth).abs()

                        self.lidar_diag["final_depth_num_points"] += int(
                            valid_depth.sum().item()
                        )
                        self.lidar_diag["final_depth_abs_error_sum"] += (
                            abs_error.sum().item()
                        )
                        self.lidar_diag["final_depth_abs_rel_sum"] += (
                            (abs_error / target_depth).sum().item()
                        )
                        self.lidar_diag["final_depth_sq_error_sum"] += (
                            (predicted_depth - target_depth).square().sum().item()
                        )

            # ============================================================
            # LiDAR 三阶段误差诊断：
            # 1. 纯视觉 coarse depth
            # 2. 注入 LiDAR bias 后的 coarse depth
            # 3. refinement U-Net 后的 final depth
            # ============================================================
            if (
                need_lidar
                and self.use_lidar_bias
                and lidar_depth is not None
                and lidar_mask is not None
            ):
                with torch.no_grad():
                    # 原始全分辨率 LiDAR：
                    # [B,V,1,H,W] -> [V*B,1,H,W]
                    lidar_depth_full = rearrange(
                        lidar_depth,
                        "b v c h w -> (v b) c h w",
                    ).to(
                        device=fine_disps.device,
                        dtype=fine_disps.dtype,
                    )

                    lidar_mask_full = rearrange(
                        lidar_mask,
                        "b v c h w -> (v b) c h w",
                    ).to(device=fine_disps.device)

                    # 只统计真实有效的 LiDAR 像素
                    valid = (
                        (lidar_mask_full > 0.5)
                        & torch.isfinite(lidar_depth_full)
                        & (lidar_depth_full > 1e-6)
                    )

                    if valid.any():
                        lidar_disp_full = (
                            1.0 / lidar_depth_full.clamp(min=1e-6)
                        )

                        # 纯视觉 coarse disparity 上采样到全分辨率
                        visual_coarse_full = F.interpolate(
                            coarse_disps_vis,
                            size=fine_disps.shape[-2:],
                            mode="bilinear",
                            align_corners=True,
                        )

                        # fullres_disps 就是：
                        # 注入 LiDAR bias 后的 coarse disparity 上采样结果
                        biased_coarse_full = fullres_disps

                        # 当前实验 gaussians_per_pixel=1
                        # 若以后使用多个 surface，这里暂时统计第一个
                        final_disp_for_diag = fine_disps[:, :1]

                        # 三阶段逐像素误差
                        visual_error_map = (
                            visual_coarse_full - lidar_disp_full
                        ).abs()

                        bias_error_map = (
                            biased_coarse_full - lidar_disp_full
                        ).abs()

                        final_error_map = (
                            final_disp_for_diag - lidar_disp_full
                        ).abs()

                        # refinement 实际改动了多少
                        refine_delta_map = (
                            final_disp_for_diag - biased_coarse_full
                        ).abs()

                        visual_error = visual_error_map[valid]
                        bias_error = bias_error_map[valid]
                        final_error = final_error_map[valid]
                        refine_delta = refine_delta_map[valid]

                        num_points = int(valid.sum().item())

                        # 累加整个测试集，而不是只看单个 batch
                        self.lidar_diag["num_points"] += num_points

                        self.lidar_diag["vis_error_sum"] += (
                            visual_error.sum().item()
                        )

                        self.lidar_diag["bias_error_sum"] += (
                            bias_error.sum().item()
                        )

                        self.lidar_diag["refine_delta_sum"] += (
                            refine_delta.sum().item()
                        )

                        # refinement 后比 bias coarse 更接近 LiDAR
                        self.lidar_diag["refine_improve_count"] += int(
                            (final_error < bias_error).sum().item()
                        )

                        # refinement 后反而离 LiDAR 更远
                        self.lidar_diag["refine_worsen_count"] += int(
                            (final_error > bias_error).sum().item()
                        )            
            depths = 1.0 / fine_disps
            depths = repeat(
                depths,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )

        return depths, densities, raw_gaussians, lidar_coarse_loss, lidar_refine_loss
    