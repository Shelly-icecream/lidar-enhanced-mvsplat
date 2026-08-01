import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid
from .ldm_unet.unet import UNetModel
class LidarTokenCrossAttention(nn.Module):
    """Condition a dense per-view feature map on sparse LiDAR tokens."""

    def __init__(
        self,
        feature_dim: int,
        attention_dim: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError(
                "LiDAR attention_dim must be divisible by num_heads."
            )

        # Raw normalized xy plus sin/cos at pi and 2*pi for each axis.
        position_dim = 10
        lidar_attribute_dim = 5
        self.query_norm = nn.LayerNorm(feature_dim)
        self.query_proj = nn.Linear(feature_dim, attention_dim)
        self.query_position_proj = nn.Linear(position_dim, attention_dim)
        self.lidar_attribute_encoder = nn.Sequential(
            nn.Linear(lidar_attribute_dim, attention_dim),
            nn.SiLU(),
            nn.Linear(attention_dim, attention_dim),
        )
        self.lidar_position_proj = nn.Linear(position_dim, attention_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.output_proj = nn.Linear(attention_dim, feature_dim)
        # Start close to the pretrained visual path while allowing gradients
        # to reach the token encoder and Q/K/V projections from the first step.
        nn.init.normal_(
            self.output_proj.weight,
            mean=0.0,
            std=1e-3,
        )

        nn.init.zeros_(self.output_proj.bias)

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
        feature: torch.Tensor,
        lidar_disp: torch.Tensor,
        lidar_mask: torch.Tensor,
        visual_disp: torch.Tensor,
        visual_entropy: torch.Tensor,
        lidar_density: torch.Tensor,
        disp_min: torch.Tensor,
        disp_max: torch.Tensor,
    ) -> torch.Tensor:
        vb, channels, height, width = feature.shape
        if lidar_mask.shape[-2:] != (height, width):
            raise ValueError(
                "LiDAR token grid must match the correlation feature grid."
            )

        valid = lidar_mask[:, 0] > 0.5
        has_lidar = valid.flatten(1).any(dim=1)
        if not has_lidar.any():
            return feature

        grid = self._normalized_grid(
            height,
            width,
            feature.device,
            feature.dtype,
        )
        query_position = self._position_encoding(grid)
        dense_tokens = feature.flatten(2).transpose(1, 2)

        active_indices = has_lidar.nonzero(as_tuple=False).flatten()
        active_tokens = dense_tokens[active_indices]
        query = self.query_proj(self.query_norm(active_tokens))
        query = query + self.query_position_proj(query_position)[None]

        disp_scale = (disp_max - disp_min).clamp_min(1e-6)
        lidar_disp_norm = (lidar_disp - disp_min) / disp_scale
        visual_disp_norm = (visual_disp - disp_min) / disp_scale
        residual_norm = (lidar_disp - visual_disp) / disp_scale

        token_sequences = []
        position_sequences = []
        for batch_index in active_indices.tolist():
            valid_flat = valid[batch_index].reshape(-1)
            attributes = torch.stack(
                (
                    lidar_disp_norm[batch_index, 0].reshape(-1)[valid_flat],
                    visual_disp_norm[batch_index, 0].reshape(-1)[valid_flat],
                    residual_norm[batch_index, 0].reshape(-1)[valid_flat],
                    visual_entropy[batch_index, 0].reshape(-1)[valid_flat],
                    lidar_density[batch_index, 0].reshape(-1)[valid_flat],
                ),
                dim=-1,
            )
            token_sequences.append(attributes)
            position_sequences.append(grid[valid_flat])

        max_tokens = max(sequence.shape[0] for sequence in token_sequences)
        token_attributes = feature.new_zeros(
            len(token_sequences),
            max_tokens,
            token_sequences[0].shape[-1],
        )
        token_positions = feature.new_zeros(
            len(position_sequences),
            max_tokens,
            2,
        )
        padding_mask = torch.ones(
            len(token_sequences),
            max_tokens,
            device=feature.device,
            dtype=torch.bool,
        )
        for index, (attributes, positions) in enumerate(
            zip(token_sequences, position_sequences)
        ):
            count = attributes.shape[0]
            token_attributes[index, :count] = attributes
            token_positions[index, :count] = positions
            padding_mask[index, :count] = False

        lidar_tokens = self.lidar_attribute_encoder(token_attributes)
        lidar_tokens = lidar_tokens + self.lidar_position_proj(
            self._position_encoding(token_positions)
        )
        attended, _ = self.cross_attention(
            query=query,
            key=lidar_tokens,
            value=lidar_tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        fused_tokens = active_tokens + self.output_proj(attended)
        output_tokens = dense_tokens.clone()
        output_tokens[active_indices] = fused_tokens
        return output_tokens.transpose(1, 2).reshape(
            vb,
            channels,
            height,
            width,
        )

class AdaptiveLidarFusion(nn.Module):
    """Predict interpretable visual-need and LiDAR-reliability gates."""

    def __init__(self) -> None:
        super().__init__()
        # Visual-only inputs: normalized entropy, inverse top-2 margin, and
        # normalized disparity standard deviation.
        self.visual_need_net = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.SiLU(),
            nn.Conv2d(16, 8, 1),
            nn.SiLU(),
            nn.Conv2d(8, 1, 1),
        )
        # LiDAR/cross-modal inputs: normalized LiDAR disparity, visual-LiDAR
        # residual, local point density, and local disparity disagreement.
        self.lidar_reliability_net = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.SiLU(),
            nn.Conv2d(16, 8, 1),
            nn.SiLU(),
            nn.Conv2d(8, 1, 1),
        )
        # Keep both sigmoid gates in their responsive, non-saturated region.
        # With small final-layer weights, the initial outputs are approximately
        # visual_need_gain=1.5 and lidar_reliability=0.8, so fusion_gain=1.2.
        nn.init.normal_(
            self.visual_need_net[-1].weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.visual_need_net[-1].bias)
        nn.init.normal_(
            self.lidar_reliability_net[-1].weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.constant_(
            self.lidar_reliability_net[-1].bias,
            math.log(4.0),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        lidar_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual_need_gain = 1.0 + torch.sigmoid(
            self.visual_need_net(visual_features)
        )
        lidar_reliability = torch.sigmoid(
            self.lidar_reliability_net(lidar_features)
        )
        fusion_gain = visual_need_gain * lidar_reliability
        return visual_need_gain, lidar_reliability, fusion_gain

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
    depth_parameter_net=None,
    adaptive_fusion_module=None,
    visual_depth_logits=None,
    lambda_surface=10.0,
    lambda_free=2.0,
    lambda_delta_log_max=math.log(4.0),
    sigma_disp=0.32,
    free_margin=0.5,
    lidar_temperature=1.0,
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

    visual_need_gain = torch.ones_like(lidar_disp_low)
    lidar_reliability = torch.ones_like(lidar_disp_low)
    fusion_gain = torch.ones_like(lidar_disp_low)
    normalized_entropy = torch.zeros_like(lidar_disp_low)
    normalized_residual = torch.zeros_like(lidar_disp_low)
    local_density = torch.zeros_like(lidar_disp_low)

    if visual_depth_logits is not None:
        if lidar_temperature <= 0:
            raise ValueError(
                "lidar_temperature must be positive, "
                f"got {lidar_temperature}."
            )
        # Detaching the visual statistics prevents the gate from making the
        # visual predictor artificially uncertain just to increase its gain.
        # Use the same temperature as the visual logits in the LiDAR fusion
        # path so the gate observes the distribution that is actually fused.
        visual_pdf = F.softmax(
            visual_depth_logits.detach() / lidar_temperature,
            dim=1,
        )
        visual_expected_disp = (
            visual_pdf * disp_candi_curr
        ).sum(dim=1, keepdim=True)
        normalized_visual_disp = (
            (visual_expected_disp - disp_min) / disp_range
        ).clamp(0.0, 1.0)

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
        visual_features = torch.cat(
            [
                normalized_entropy,
                inverse_margin,
                normalized_visual_std,
            ],
            dim=1,
        )
        lidar_features = torch.cat(
            [
                normalized_lidar_disp,
                normalized_residual,
                local_density,
                local_disp_disagreement,
            ],
            dim=1,
        )
        if adaptive_fusion_module is not None:
            (
                visual_need_gain,
                lidar_reliability,
                fusion_gain,
            ) = adaptive_fusion_module(
                visual_features,
                lidar_features,
            )

    if adaptive_fusion_module is not None:
        lambda_surface_map = torch.full_like(
            lidar_disp_low, float(lambda_surface)
        )
        lambda_free_map = torch.full_like(
            lidar_disp_low, float(lambda_free)
        )
    elif depth_parameter_net is None:
        lambda_surface_map = torch.full_like(
            lidar_disp_low, float(lambda_surface)
        )
        lambda_free_map = torch.full_like(
            lidar_disp_low, float(lambda_free)
        )
    else:
        delta_log_lambdas = (
            torch.tanh(depth_parameter_net(normalized_lidar_disp))
            * lambda_delta_log_max
        )
        delta_log_surface, delta_log_free = delta_log_lambdas.chunk(2, dim=1)
        lambda_surface_map = float(lambda_surface) * delta_log_surface.exp()
        lambda_free_map = float(lambda_free) * delta_log_free.exp()
        
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
    ) * fusion_gain

    return (
        lidar_bias,
        lidar_mask_low,
        lidar_disp_low,
        lambda_surface_map,
        lambda_free_map,
        visual_need_gain,
        lidar_reliability,
        fusion_gain,
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

    def set_lidar_depth_parameter_net_enabled(self, enabled: bool) -> None:
        """Match the optional LiDAR-parameter network to a checkpoint."""
        self.use_learnable_lidar_bias_params = enabled
        if enabled and self.lidar_depth_parameter_net is None:
            self.lidar_depth_parameter_net = nn.Sequential(
                nn.Conv2d(1, 16, 1),
                nn.SiLU(),
                nn.Conv2d(16, 2, 1),
            )
            nn.init.zeros_(self.lidar_depth_parameter_net[-1].weight)
            nn.init.zeros_(self.lidar_depth_parameter_net[-1].bias)
        elif not enabled:
            self.lidar_depth_parameter_net = None
    def set_adaptive_lidar_fusion_enabled(self, enabled: bool) -> None:
        self.use_adaptive_lidar_fusion = enabled
        if enabled and self.adaptive_lidar_fusion is None:
            self.adaptive_lidar_fusion = AdaptiveLidarFusion()
        elif not enabled:
            self.adaptive_lidar_fusion = None

    def set_lidar_cross_attention_enabled(self, enabled: bool) -> None:
        self.use_lidar_cross_attention = bool(enabled)
        if enabled and self.lidar_cross_attention is None:
            self.lidar_cross_attention = LidarTokenCrossAttention(
                feature_dim=self.regressor_feat_dim,
                attention_dim=self.lidar_cross_attention_dim,
                num_heads=self.lidar_cross_attention_heads,
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
        use_learnable_lidar_bias_params=False,
        use_adaptive_lidar_fusion=False,
        use_lidar_cross_attention=False,
        lidar_cross_attention_dim=128,
        lidar_cross_attention_heads=4,
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
        self.use_learnable_lidar_bias_params = (
            use_learnable_lidar_bias_params
        )
        self.use_adaptive_lidar_fusion = use_adaptive_lidar_fusion
        self.use_lidar_cross_attention = use_lidar_cross_attention
        self.lidar_cross_attention_dim = lidar_cross_attention_dim
        self.lidar_cross_attention_heads = lidar_cross_attention_heads
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
        if (
            self.use_learnable_lidar_bias_params
            and self.use_adaptive_lidar_fusion
        ):
            raise ValueError(
                "Legacy learnable LiDAR parameters and adaptive LiDAR "
                "fusion cannot be enabled at the same time."
            )
        self.lidar_lambda_surface = lidar_lambda_surface
        self.lidar_lambda_free = lidar_lambda_free
        self.lidar_sigma_disp = lidar_sigma_disp
        self.lidar_free_margin = lidar_free_margin
        self.lidar_temperature = lidar_temperature
        self.lidar_lambda_delta_log_max = math.log(4.0)
        self.lidar_depth_parameter_net = None
        if self.use_learnable_lidar_bias_params:
            self.set_lidar_depth_parameter_net_enabled(True)
        self.adaptive_lidar_fusion = None
        if self.use_adaptive_lidar_fusion:
            self.set_adaptive_lidar_fusion_enabled(True)
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
        # + the full-resolution support of the low-resolution LiDAR bias.
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
                visual_need_gain,
                lidar_reliability,
                fusion_gain,
                visual_entropy,
                visual_lidar_residual,
                lidar_local_density,
                lidar_surface_prior,
            ) = build_lidar_visibility_prior(
                lidar_depth=lidar_depth,              # [B,V,1,H,W]
                lidar_mask=lidar_mask,                # [B,V,1,H,W]
                disp_candi_curr=disp_candi_curr,      # [v*b,D,1,1]
                target_hw=depth_logits_vis.shape[-2:],    # (h,w)
                depth_parameter_net=self.lidar_depth_parameter_net,
                adaptive_fusion_module=self.adaptive_lidar_fusion,
                visual_depth_logits=depth_logits_vis,
                lambda_surface=self.lidar_lambda_surface,
                lambda_free=self.lidar_lambda_free,
                lambda_delta_log_max=self.lidar_lambda_delta_log_max,
                sigma_disp=self.lidar_sigma_disp,
                free_margin=self.lidar_free_margin,
                lidar_temperature=self.lidar_temperature,
            )
            if self.use_lidar_cross_attention:
                if corr_feature is None or self.lidar_cross_attention is None:
                    raise RuntimeError(
                        "LiDAR cross-attention requires cost-volume refinement."
                    )
                disp_min_low = disp_candi_curr.amin(
                    dim=1,
                    keepdim=True,
                )
                disp_max_low = disp_candi_curr.amax(
                    dim=1,
                    keepdim=True,
                )
                corr_feature = self.lidar_cross_attention(
                    feature=corr_feature,
                    lidar_disp=lidar_disp_low,
                    lidar_mask=lidar_mask_low,
                    visual_disp=coarse_disps_vis,
                    visual_entropy=visual_entropy,
                    lidar_density=lidar_local_density,
                    disp_min=disp_min_low,
                    disp_max=disp_max_low,
                )
                raw_correlation = self.corr_refine_net[-1](corr_feature)
                raw_correlation = raw_correlation + self.regressor_residual(
                    raw_correlation_in
                )
                depth_logits_vis = self.depth_head_lowres(raw_correlation)
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
                depth_logits_calibrated = (
                    depth_logits_vis * (1.0 - lidar_mask_low) + (depth_logits_vis / self.lidar_temperature) * lidar_mask_low
             )
                depth_logits_lidar = depth_logits_calibrated + lidar_bias
            else:
                depth_logits_lidar = depth_logits_vis
            # loss only uses pure visual prediction before LiDAR bias.
            if self.use_lidar_coarse_loss and mask.any():
                if self.use_lidar_cross_attention:
                    # Reuse the analytic bias's LiDAR surface response as the
                    # target, but supervise logits before any bias is applied.
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
                    lidar_candidate_target = (
                        lidar_surface_prior
                        / lidar_surface_prior.sum(
                            dim=1,
                            keepdim=True,
                        ).clamp_min(1e-8)
                    ).detach()
                    log_pdf_after = F.log_softmax(
                        depth_logits_vis,
                        dim=1,
                    )
                    candidate_kl_map = F.kl_div(
                        log_pdf_after,
                        lidar_candidate_target,
                        reduction="none",
                    ).sum(dim=1, keepdim=True)
                    lidar_disparity_loss = error_after[mask].mean()
                    if candidate_mask.any():
                        lidar_candidate_loss = candidate_kl_map[
                            candidate_mask
                        ].mean()
                    else:
                        lidar_candidate_loss = candidate_kl_map.new_zeros(())
                    lidar_coarse_loss = (
                        lidar_disparity_loss
                        +
                        1.0 * lidar_candidate_loss
                    )
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
                    visual_need_values = visual_need_gain[
                        valid_parameter_cells
                    ]
                    reliability_values = lidar_reliability[
                        valid_parameter_cells
                    ]
                    fusion_gain_values = fusion_gain[
                        valid_parameter_cells
                    ]
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
                        "visual_need_gain_mean": (
                            visual_need_values.mean().detach()
                        ),
                        "visual_need_gain_std": (
                            visual_need_values.std(unbiased=False).detach()
                        ),
                        "visual_need_gain_min": (
                            visual_need_values.min().detach()
                        ),
                        "visual_need_gain_max": (
                            visual_need_values.max().detach()
                        ),
                        "lidar_reliability_mean": (
                            reliability_values.mean().detach()
                        ),
                        "lidar_reliability_std": (
                            reliability_values.std(unbiased=False).detach()
                        ),
                        "lidar_reliability_min": (
                            reliability_values.min().detach()
                        ),
                        "lidar_reliability_max": (
                            reliability_values.max().detach()
                        ),
                        "fusion_gain_mean": (
                            fusion_gain_values.mean().detach()
                        ),
                        "fusion_gain_std": (
                            fusion_gain_values.std(unbiased=False).detach()
                        ),
                        "fusion_gain_min": (
                            fusion_gain_values.min().detach()
                        ),
                        "fusion_gain_max": (
                            fusion_gain_values.max().detach()
                        ),
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
    