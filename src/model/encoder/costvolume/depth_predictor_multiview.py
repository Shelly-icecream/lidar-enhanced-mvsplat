import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel


class LidarGaussianCrossAttention(nn.Module):
    """Local cross-attention from LiDAR queries to surrounding Gaussians."""

    def __init__(
        self,
        visual_dim: int,
        gaussian_dim: int,
        output_dim: int,
        attention_dim: int = 64,
        num_heads: int = 4,
        radius: int = 2,
        query_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError("attention_dim must be divisible by num_heads.")
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.radius = radius
        self.query_chunk_size = query_chunk_size

        # A LiDAR query contains its visual feature, base raw Gaussian,
        # normalized LiDAR disparity and image coordinates. Every surrounding
        # pixel contributes a Gaussian K/V token; its LiDAR mask explicitly
        # distinguishes visual-only tokens from LiDAR-supported tokens.
        self.query_proj = nn.Linear(
            visual_dim + gaussian_dim + 3, attention_dim
        )
        token_dim = visual_dim + gaussian_dim + 4
        self.key_proj = nn.Linear(token_dim, attention_dim)
        self.value_proj = nn.Linear(token_dim, attention_dim)
        self.output_head = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.GELU(),
            nn.Linear(attention_dim, output_dim, bias=False),
        )
        nn.init.zeros_(self.output_head[-1].weight)

    def forward(
        self,
        visual_features: torch.Tensor,
        gaussian_features: torch.Tensor,
        lidar_mask: torch.Tensor,
        lidar_disp: torch.Tensor,
        query_gate: torch.Tensor,
    ) -> torch.Tensor:
        vb, _, height, width = visual_features.shape
        output = visual_features.new_zeros(
            vb, self.output_head[-1].out_features, height, width
        )
        visual_flat = visual_features.flatten(2).transpose(1, 2)
        gaussian_flat = gaussian_features.flatten(2).transpose(1, 2)
        lidar_mask_flat = lidar_mask[:, 0].flatten(1).bool()
        query_mask_flat = query_gate[:, 0].flatten(1) > 0
        lidar_disp_flat = lidar_disp[:, 0].flatten(1)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=visual_features.device),
            torch.arange(width, device=visual_features.device),
            indexing="ij",
        )
        pixel_xy = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
        normalized_xy = pixel_xy.to(visual_features.dtype)
        normalized_xy = normalized_xy / visual_features.new_tensor(
            [max(width - 1, 1), max(height - 1, 1)]
        )
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(
                -self.radius,
                self.radius + 1,
                device=visual_features.device,
            ),
            torch.arange(
                -self.radius,
                self.radius + 1,
                device=visual_features.device,
            ),
            indexing="ij",
        )
        local_offsets = torch.stack((offsets_x, offsets_y), dim=-1).reshape(-1, 2)

        for batch_index in range(vb):
            query_indices = query_mask_flat[batch_index].nonzero(as_tuple=False)[:, 0]
            if query_indices.numel() == 0:
                continue

            all_attributes = torch.cat(
                (
                    visual_flat[batch_index],
                    gaussian_flat[batch_index],
                    lidar_disp_flat[batch_index, :, None],
                    lidar_mask_flat[batch_index, :, None].to(
                        visual_features.dtype
                    ),
                    normalized_xy,
                ),
                dim=-1,
            )
            key_map = self.key_proj(all_attributes).reshape(
                height * width, self.num_heads, self.head_dim
            )
            value_map = self.value_proj(all_attributes).reshape(
                height * width, self.num_heads, self.head_dim
            )

            for start in range(0, query_indices.numel(), self.query_chunk_size):
                chunk_indices = query_indices[start : start + self.query_chunk_size]
                query_input = torch.cat(
                    (
                        visual_flat[batch_index, chunk_indices],
                        gaussian_flat[batch_index, chunk_indices],
                        lidar_disp_flat[batch_index, chunk_indices, None],
                        normalized_xy[chunk_indices],
                    ),
                    dim=-1,
                )
                query = self.query_proj(query_input).reshape(
                    -1, self.num_heads, self.head_dim
                )
                query_xy = pixel_xy[chunk_indices]
                neighbor_xy = query_xy[:, None] + local_offsets[None]
                in_bounds = (
                    (neighbor_xy[..., 0] >= 0)
                    & (neighbor_xy[..., 0] < width)
                    & (neighbor_xy[..., 1] >= 0)
                    & (neighbor_xy[..., 1] < height)
                )
                neighbor_x = neighbor_xy[..., 0].clamp(0, width - 1)
                neighbor_y = neighbor_xy[..., 1].clamp(0, height - 1)
                neighbor_indices = neighbor_y * width + neighbor_x
                local_valid = in_bounds
                local_key = key_map[neighbor_indices]
                local_value = value_map[neighbor_indices]
                scores = torch.einsum("qhd,qkhd->qhk", query, local_key)
                scores = scores / math.sqrt(self.head_dim)
                scores = scores.masked_fill(~local_valid[:, None], -1e4)
                weights = F.softmax(scores, dim=-1)
                weights = weights * local_valid[:, None].to(weights.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                attended = torch.einsum("qhk,qkhd->qhd", weights, local_value)
                attended = attended.reshape(-1, self.attention_dim)
                chunk_output = self.output_head(attended)
                output[batch_index].flatten(1)[:, chunk_indices] = chunk_output.transpose(0, 1)
        return output

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

    def __init__(
        self,
        feature_channels=128,
        upscale_factor=4,
        num_depth_candidates=32,
        costvolume_unet_feat_dim=128,
        costvolume_unet_channel_mult=(1, 1, 1),
        costvolume_unet_attn_res=(),
        gaussian_raw_channels=-1,
        gaussian_channels_per_surface=-1,
        gaussians_per_pixel=1,
        num_views=2,
        depth_unet_feat_dim=64,
        depth_unet_attn_res=(),
        depth_unet_channel_mult=(1, 1, 1),
        wo_depth_refine=False,
        wo_cost_volume=False,
        wo_cost_volume_refine=False,
        
        use_lidar_bias=False,
        use_lidar_gaussian_adapter=False,
        lidar_gaussian_edit_xy=True,
        lidar_gaussian_edit_scale=True,
        lidar_gaussian_edit_rotation=True,
        lidar_gaussian_edit_sh_dc=False,
        lidar_gaussian_edit_sh_rest=False,
        lidar_gaussian_edit_opacity=False,
        lidar_gaussian_opacity_max_delta_logit=1.0,
        lidar_lambda_surface=10.0,
        lidar_lambda_free=2.0,
        lidar_sigma_disp=0.12,
        lidar_free_margin=0.5,
        lidar_temperature=5.0,
        lidar_gaussian_gate_kernel=5,
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
        self.compute_lidar_depth_repair_aux = False
        self.lidar_depth_repair_aux = None
        self.use_lidar_gaussian_adapter = use_lidar_gaussian_adapter
        self.lidar_gaussian_edit_xy = bool(lidar_gaussian_edit_xy)
        self.lidar_gaussian_edit_scale = bool(lidar_gaussian_edit_scale)
        self.lidar_gaussian_edit_rotation = bool(lidar_gaussian_edit_rotation)
        self.lidar_gaussian_edit_sh_dc = bool(lidar_gaussian_edit_sh_dc)
        self.lidar_gaussian_edit_sh_rest = bool(lidar_gaussian_edit_sh_rest)
        self.lidar_gaussian_edit_opacity = bool(lidar_gaussian_edit_opacity)
        self.lidar_gaussian_opacity_max_delta_logit = float(
            lidar_gaussian_opacity_max_delta_logit
        )
        self.lidar_lambda_surface = lidar_lambda_surface
        self.lidar_lambda_free = lidar_lambda_free
        self.lidar_sigma_disp = lidar_sigma_disp
        self.lidar_free_margin = lidar_free_margin
        self.lidar_temperature = lidar_temperature
        self.lidar_gaussian_gate_kernel = int(lidar_gaussian_gate_kernel)
        if self.lidar_gaussian_gate_kernel < 1 or self.lidar_gaussian_gate_kernel % 2 == 0:
            raise ValueError("lidar_gaussian_gate_kernel must be a positive odd integer.")
        if self.lidar_temperature <= 0:
            raise ValueError(
                "lidar_temperature must be positive, "
                f"got {self.lidar_temperature}."
            )
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
        # A zero-initialized residual branch preserves the pretrained Gaussian
        # head exactly at initialization. Only full-resolution LiDAR pixels are
        # queried and edited; each query can attend to nearby LiDAR tokens. The
        # Editable channels are selected explicitly below for controlled
        # geometry/color ablations.
        self.gaussian_raw_channels = gaussian_raw_channels
        self.gaussian_channels_per_surface = gaussian_channels_per_surface
        self.num_gaussian_surfaces = (
            gaussian_raw_channels // gaussian_channels_per_surface
        )
        self.lidar_gaussian_adapter = None
        if self.use_lidar_gaussian_adapter:
            self.lidar_gaussian_adapter = LidarGaussianCrossAttention(
                visual_dim=gau_in,
                gaussian_dim=gaussian_raw_channels,
                output_dim=(
                    gaussian_raw_channels
                    + self.num_gaussian_surfaces
                ),
                attention_dim=64,
                num_heads=4,
                radius=self.lidar_gaussian_gate_kernel // 2,
            )

            editable = torch.zeros(1, gaussian_raw_channels, 1, 1)
            if gaussian_channels_per_surface < 9:
                raise ValueError(
                    "gaussian_channels_per_surface must contain xy, scale and rotation channels."
                )
            sh_channels = gaussian_channels_per_surface - 9
            if sh_channels % 3 != 0:
                raise ValueError(
                    "Gaussian SH channels must be divisible into RGB groups."
                )
            d_sh = sh_channels // 3
            for start in range(0, gaussian_raw_channels, gaussian_channels_per_surface):
                if self.lidar_gaussian_edit_xy:
                    editable[:, start : start + 2] = 1.0
                if self.lidar_gaussian_edit_scale:
                    editable[:, start + 2 : start + 5] = 1.0
                if self.lidar_gaussian_edit_rotation:
                    editable[:, start + 5 : start + 9] = 1.0
                for color_index in range(3):
                    dc_index = start + 9 + color_index * d_sh
                    if self.lidar_gaussian_edit_sh_dc:
                        editable[:, dc_index] = 1.0
                    if self.lidar_gaussian_edit_sh_rest:
                        sh_start = start + 9 + color_index * d_sh
                        editable[:, sh_start + 1 : sh_start + d_sh] = 1.0
            if not editable.bool().any() and not self.lidar_gaussian_edit_opacity:
                raise ValueError(
                    "LiDAR Gaussian adapter is enabled but no editable channels are selected."
                )
            self.register_buffer("lidar_gaussian_editable_channels", editable)

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
        self.lidar_depth_repair_aux = None
        
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
        # Keep the pretrained Sequential parameter names intact.
            corr_feature = raw_correlation_in
            for module in list(self.corr_refine_net.children())[:-1]:
                corr_feature = module(corr_feature)
            raw_correlation = self.corr_refine_net[-1](corr_feature)
            # apply skip connection
            raw_correlation = raw_correlation + self.regressor_residual(
                raw_correlation_in
            )
        depth_logits = self.depth_head_lowres(raw_correlation)

        # The coarse/refinement path is purely visual. Valid full-resolution
        # LiDAR samples are injected only into the final disparity below.
        pdf = F.softmax(depth_logits, dim=1)  # [v*b, D, h, w]
        
        coarse_disps = (disp_candi_curr * pdf).sum(
            dim=1, keepdim=True
        )  # (vb, 1, h, w)
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
        raw_gaussians_base = raw_gaussians.detach()
        self.lidar_gaussian_opacity_logit_residual = None
        if self.use_lidar_gaussian_adapter and lidar_depth is not None and lidar_mask is not None:
            lidar_depth_full = rearrange(
                lidar_depth, "b v c h w -> (v b) c h w"
            ).to(device=raw_gaussians.device, dtype=raw_gaussians.dtype)
            lidar_mask_full = rearrange(
                lidar_mask, "b v c h w -> (v b) c h w"
            ).to(device=raw_gaussians.device, dtype=raw_gaussians.dtype)
            valid_full = (
                (lidar_mask_full > 0.5)
                & torch.isfinite(lidar_depth_full)
                & (lidar_depth_full > 1e-6)
            ).to(raw_gaussians.dtype)
            lidar_disp_full = torch.where(
                valid_full > 0,
                lidar_depth_full.clamp_min(1e-6).reciprocal(),
                torch.zeros_like(lidar_depth_full),
            )
            disp_min_full = 1.0 / rearrange(far, "b v -> (v b) () () ()")
            disp_max_full = 1.0 / rearrange(near, "b v -> (v b) () () ()")
            lidar_disp_normalized = (
                (lidar_disp_full - disp_min_full)
                / (disp_max_full - disp_min_full).clamp_min(1e-6)
            ).clamp(0.0, 1.0) * valid_full

            adapter_output = self.lidar_gaussian_adapter(
                visual_features=raw_gaussians_in,
                gaussian_features=raw_gaussians,
                lidar_mask=valid_full,
                lidar_disp=lidar_disp_normalized,
                query_gate=valid_full,
            )
            residual = adapter_output[:, : self.gaussian_raw_channels]
            residual = residual * self.lidar_gaussian_editable_channels
            # Keep the hard mask at the final write boundary as an explicit
            # guarantee that non-LiDAR pixels remain bit-for-bit unchanged.
            gated_residual = residual * valid_full
            opacity_residual = adapter_output[:, self.gaussian_raw_channels :]
            if self.lidar_gaussian_edit_opacity:
                bounded_opacity_residual = (
                    self.lidar_gaussian_opacity_max_delta_logit
                    * torch.tanh(opacity_residual)
                    * valid_full
                )
                self.lidar_gaussian_opacity_logit_residual = rearrange(
                    bounded_opacity_residual,
                    "(v b) srf h w -> b v (h w) srf",
                    v=v,
                    b=b,
                )
            raw_gaussians = raw_gaussians + gated_residual
        raw_gaussians = rearrange(
            raw_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
        )
        raw_gaussians_base = rearrange(
            raw_gaussians_base, "(v b) c h w -> b v (h w) c", v=v, b=b
        )

        if self.wo_depth_refine:
            densities = repeat(
                pdf_max,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )
            final_disps = fullres_disps
            if self.use_lidar_bias and lidar_depth is not None and lidar_mask is not None:
                lidar_depth_full = rearrange(
                    lidar_depth, "b v c h w -> (v b) c h w"
                ).to(device=final_disps.device, dtype=final_disps.dtype)
                lidar_mask_full = rearrange(
                    lidar_mask, "b v c h w -> (v b) c h w"
                ).to(device=final_disps.device)
                disp_min = 1.0 / rearrange(
                    far, "b v -> (v b) () () ()"
                )
                disp_max = 1.0 / rearrange(
                    near, "b v -> (v b) () () ()"
                )
                lidar_disp_full = lidar_depth_full.clamp_min(1e-6).reciprocal()
                valid = (
                    (lidar_mask_full > 0.5)
                    & torch.isfinite(lidar_depth_full)
                    & (lidar_depth_full > 1e-6)
                    & (lidar_disp_full >= disp_min)
                    & (lidar_disp_full <= disp_max)
                )
                final_disps = torch.where(valid, lidar_disp_full, final_disps)
            depths = 1.0 / final_disps
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

            lidar_depth_full = None
            lidar_mask_full = None
            valid = None
            if lidar_depth is not None and lidar_mask is not None:
                lidar_depth_full = rearrange(
                    lidar_depth,
                    "b v c h w -> (v b) c h w",
                ).to(device=delta_disps.device, dtype=delta_disps.dtype)
                lidar_mask_full = rearrange(
                    lidar_mask,
                    "b v c h w -> (v b) c h w",
                ).to(device=delta_disps.device)
                valid = (
                    (lidar_mask_full > 0.5)
                    & torch.isfinite(lidar_depth_full)
                    & (lidar_depth_full > 1e-6)
                )

            visual_fine_disps = (fullres_disps + delta_disps).clamp(
                min=disp_min,
                max=disp_max,
            )

            # Inject LiDAR only after visual refinement.  A sample outside the
            # camera near/far disparity interval is rejected instead of being
            # clamped to a boundary and turned into a false hard anchor.
            if self.use_lidar_bias and valid is not None:
                lidar_disp_full = lidar_depth_full.clamp_min(1e-6).reciprocal()
                valid = (
                    valid
                    & (lidar_disp_full >= disp_min)
                    & (lidar_disp_full <= disp_max)
                )
                fine_disps = torch.where(
                    valid,
                    lidar_disp_full,
                    visual_fine_disps,
                )
            else:
                fine_disps = visual_fine_disps

            self.lidar_depth_repair_aux = None
            if self.compute_lidar_depth_repair_aux:
                # The visual refinement is now shared exactly.  Only the final
                # disparity at a valid LiDAR pixel differs from this reference.
                self.lidar_depth_repair_aux = {
                    "reference_disparity": visual_fine_disps[:, :1].detach(),
                    "biased_disparity": fine_disps[:, :1].detach(),
                    "shallow_feature": proj_feat_in_fullres.detach(),
                    "refine_feature": refine_out.detach(),
                }

            depths = 1.0 / fine_disps
            depths = repeat(
                depths,
                "(v b) dpt h w -> b v (h w) srf dpt",
                b=b,
                v=v,
                srf=1,
            )

        return (
            depths,
            densities,
            raw_gaussians,
            raw_gaussians_base,
        )
