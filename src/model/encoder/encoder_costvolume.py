from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import (
    BackboneMultiview,
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


class LidarNeighborDepthMLP(nn.Module):
    """Predict how strongly a neighbor moves toward a LiDAR-biased anchor."""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1, bias=True),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -4.0)

    def forward(self, pair_features: Tensor) -> Tensor:
        return self.network(pair_features).squeeze(-1)


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool
    use_lidar_bias: bool
    use_lidar_gaussian_adapter: bool
    lidar_gaussian_edit_xy: bool
    lidar_gaussian_edit_scale: bool
    lidar_gaussian_edit_rotation: bool
    lidar_gaussian_edit_sh_dc: bool
    lidar_gaussian_edit_sh_rest: bool
    lidar_gaussian_edit_opacity: bool
    lidar_gaussian_opacity_max_delta_logit: float
    use_lidar_gaussian_adapter_loss: bool
    lidar_gaussian_adapter_local_rgb_weight: float
    lidar_gaussian_adapter_improvement_weight: float
    lidar_gaussian_adapter_improvement_margin: float
    lidar_gaussian_adapter_alpha_weight: float
    lidar_gaussian_context_render_weight: float
    use_lidar_gaussian_neighbor_depth_adapter: bool
    use_lidar_gaussian_neighbor_depth_loss: bool
    lidar_neighbor_depth_radius: int
    lidar_neighbor_depth_topk: int
    lidar_neighbor_depth_min_neighbors_before_gap: int
    lidar_neighbor_depth_min_log_score_gap: float
    lidar_neighbor_depth_max_relative_ref_disparity_error: float
    lidar_neighbor_depth_reference_bound_margin: float
    lidar_neighbor_depth_min_shallow_cosine: float
    lidar_neighbor_depth_min_refine_cosine: float
    lidar_neighbor_depth_max_rgb_difference: float
    lidar_neighbor_depth_max_gate: float
    lidar_neighbor_depth_target_disagreement_scale: float
    lidar_neighbor_depth_local_rgb_weight: float
    lidar_neighbor_depth_improvement_weight: float
    lidar_neighbor_depth_coverage_weight: float
    lidar_neighbor_depth_over_alpha_worse_weight: float
    lidar_neighbor_depth_alpha_tolerance: float
    lidar_neighbor_depth_min_attraction: float
    lidar_neighbor_depth_attraction_weight: float
    lidar_neighbor_depth_diagnostic_every_n_steps: int
    frozen_params: list[str]
    lidar_gaussian_gate_kernel: int
    lidar_lambda_surface: float
    lidar_lambda_free: float
    lidar_sigma_disp: float
    lidar_free_margin: float
    lidar_temperature: float


class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)
        # multi-view Transformer backbone
        if cfg.use_epipolar_trans:
            self.epipolar_sampler = EpipolarSampler(
                num_views=get_cfg().dataset.view_sampler.num_context_views,
                num_samples=32,
            )
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        self.backbone = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
            no_cross_attn=cfg.wo_backbone_cross_attn,
            use_epipolar_trans=cfg.use_epipolar_trans,
        )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not cfg.wo_backbone_cross_attn
                self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussian_channels_per_surface=self.gaussian_adapter.d_in + 2,
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            
            use_lidar_bias=cfg.use_lidar_bias,
            use_lidar_gaussian_adapter=cfg.use_lidar_gaussian_adapter,
            lidar_gaussian_edit_xy=cfg.lidar_gaussian_edit_xy,
            lidar_gaussian_edit_scale=cfg.lidar_gaussian_edit_scale,
            lidar_gaussian_edit_rotation=cfg.lidar_gaussian_edit_rotation,
            lidar_gaussian_edit_sh_dc=cfg.lidar_gaussian_edit_sh_dc,
            lidar_gaussian_edit_sh_rest=cfg.lidar_gaussian_edit_sh_rest,
            lidar_gaussian_edit_opacity=cfg.lidar_gaussian_edit_opacity,
            lidar_gaussian_opacity_max_delta_logit=(
                cfg.lidar_gaussian_opacity_max_delta_logit
            ),
            lidar_lambda_surface=cfg.lidar_lambda_surface,
            lidar_lambda_free=cfg.lidar_lambda_free,
            lidar_sigma_disp=cfg.lidar_sigma_disp,
            lidar_free_margin=cfg.lidar_free_margin,
            lidar_temperature=cfg.lidar_temperature,
            lidar_gaussian_gate_kernel=cfg.lidar_gaussian_gate_kernel,
        )
        self.lidar_neighbor_depth_mlp = None
        self.lidar_neighbor_depth_repair_enabled = bool(
            cfg.use_lidar_gaussian_neighbor_depth_adapter
        )
        self.depth_predictor.compute_lidar_depth_repair_aux = (
            self.lidar_neighbor_depth_repair_enabled
        )
        if cfg.use_lidar_gaussian_neighbor_depth_adapter:
            if cfg.num_surfaces != 1 or cfg.gaussians_per_pixel != 1:
                raise ValueError(
                    "LiDAR neighbor repair currently requires one surface and "
                    "one Gaussian per context pixel."
                )
            if cfg.wo_depth_refine:
                raise ValueError("LiDAR neighbor depth repair requires depth refinement.")
            if cfg.lidar_neighbor_depth_radius < 1:
                raise ValueError("LiDAR neighbor depth radius must be positive.")
            max_neighbors = (2 * cfg.lidar_neighbor_depth_radius + 1) ** 2 - 1
            if not 1 <= cfg.lidar_neighbor_depth_topk <= max_neighbors:
                raise ValueError(
                    "LiDAR neighbor repair top-k must be between 1 and "
                    f"{max_neighbors}."
                )
            if not (
                1
                <= cfg.lidar_neighbor_depth_min_neighbors_before_gap
                <= cfg.lidar_neighbor_depth_topk
            ):
                raise ValueError(
                    "LiDAR neighbor minimum neighbors before score-gap "
                    "truncation must be between 1 and top-k."
                )
            if cfg.lidar_neighbor_depth_min_log_score_gap < 0:
                raise ValueError(
                    "LiDAR neighbor minimum log score gap must be non-negative."
                )
            if cfg.lidar_neighbor_depth_max_relative_ref_disparity_error <= 0:
                raise ValueError(
                    "LiDAR neighbor repair relative depth threshold must be positive."
                )
            if not 0 <= cfg.lidar_neighbor_depth_reference_bound_margin < 0.5:
                raise ValueError(
                    "LiDAR neighbor reference bound margin must be in [0, 0.5)."
                )
            for name, value in (
                ("shallow", cfg.lidar_neighbor_depth_min_shallow_cosine),
                ("refine", cfg.lidar_neighbor_depth_min_refine_cosine),
            ):
                if not -1.0 <= value <= 1.0:
                    raise ValueError(
                        f"LiDAR neighbor {name} cosine must be in [-1, 1]."
                    )
            if cfg.lidar_neighbor_depth_max_rgb_difference <= 0:
                raise ValueError("LiDAR neighbor RGB threshold must be positive.")
            if not 0 < cfg.lidar_neighbor_depth_max_gate <= 1:
                raise ValueError(
                    "LiDAR neighbor max gate must be in (0, 1]."
                )
            if cfg.lidar_neighbor_depth_target_disagreement_scale <= 0:
                raise ValueError(
                    "LiDAR neighbor target disagreement scale must be positive."
                )
            if not 0 <= cfg.lidar_neighbor_depth_min_attraction <= 1:
                raise ValueError(
                    "LiDAR neighbor minimum attraction must be in [0, 1]."
                )
            if cfg.lidar_neighbor_depth_attraction_weight < 0:
                raise ValueError(
                    "LiDAR neighbor attraction weight must be non-negative."
                )
            if cfg.lidar_neighbor_depth_diagnostic_every_n_steps < 0:
                raise ValueError(
                    "LiDAR neighbor diagnostic interval must be non-negative."
                )
            self.lidar_neighbor_depth_mlp = LidarNeighborDepthMLP()
            self._last_lidar_neighbor_depth_diagnostic_step = None
        for frozen_prefix in cfg.frozen_params:
            matched_parameters = []
            for name, param in self.named_parameters():
                if name == frozen_prefix or name.startswith(f"{frozen_prefix}."):
                    param.requires_grad_(False)
                    matched_parameters.append(name)
            if not matched_parameters:
                raise ValueError(
                    "No encoder parameters matched frozen prefix "
                    f"{frozen_prefix!r}."
                )
            if get_cfg().mode == "train":
                print(
                    "==> Freeze encoder parameters: "
                    f"{frozen_prefix} ({len(matched_parameters)} tensors)"
                )


    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _predict_lidar_neighbor_depth_residual(
        self,
        context: dict,
        depth_aux: dict[str, Tensor],
        height: int,
        width: int,
        global_step: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Select same-surface neighbors and propagate a bounded disparity shift."""
        if self.lidar_neighbor_depth_mlp is None:
            raise RuntimeError("LiDAR neighbor depth MLP is not initialized.")
        lidar_depth = context.get("lidar_depth")
        lidar_mask = context.get("lidar_mask")
        if lidar_depth is None or lidar_mask is None:
            reference = depth_aux["biased_disparity"]
            zero_link = next(self.lidar_neighbor_depth_mlp.parameters()).sum() * 0.0
            zeros = reference.new_zeros(reference.shape) + zero_link
            return zeros, zeros, zeros, zeros

        b, v = lidar_mask.shape[:2]
        n = b * v
        num_pixels = height * width
        radius = int(self.cfg.lidar_neighbor_depth_radius)
        kernel_size = 2 * radius + 1
        topk = int(self.cfg.lidar_neighbor_depth_topk)

        reference_disp = rearrange(
            depth_aux["reference_disparity"].detach(),
            "(v b) c h w -> (b v) c h w",
            b=b,
            v=v,
        )
        biased_disp = rearrange(
            depth_aux["biased_disparity"].detach(),
            "(v b) c h w -> (b v) c h w",
            b=b,
            v=v,
        )
        shallow_feature = F.normalize(
            rearrange(
                depth_aux["shallow_feature"].detach(),
                "(v b) c h w -> (b v) c h w",
                b=b,
                v=v,
            ),
            dim=1,
        )
        refine_feature = F.normalize(
            rearrange(
                depth_aux["refine_feature"].detach(),
                "(v b) c h w -> (b v) c h w",
                b=b,
                v=v,
            ),
            dim=1,
        )
        rgb_feature = rearrange(
            context["image"].detach(),
            "b v c h w -> (b v) c h w",
        ).to(device=biased_disp.device, dtype=biased_disp.dtype)
        disp_lower = rearrange(
            context["far"].reciprocal(),
            "b v -> (b v) 1 1 1",
        )
        disp_upper = rearrange(
            context["near"].reciprocal(),
            "b v -> (b v) 1 1 1",
        )
        normalized_reference = (
            (reference_disp - disp_lower)
            / (disp_upper - disp_lower).clamp_min(1e-6)
        )
        bound_margin = float(
            self.cfg.lidar_neighbor_depth_reference_bound_margin
        )
        reference_interior = (
            (normalized_reference > bound_margin)
            & (normalized_reference < 1.0 - bound_margin)
        )
        lidar_depth_map = rearrange(
            lidar_depth,
            "b v 1 h w -> (b v) 1 h w",
        ).to(device=biased_disp.device, dtype=biased_disp.dtype)
        lidar_disp_map = lidar_depth_map.clamp_min(1e-6).reciprocal()
        lidar_mask_map = rearrange(
            lidar_mask > 0.5,
            "b v 1 h w -> (b v) 1 h w",
        ).to(device=biased_disp.device)
        valid_anchor = (
            lidar_mask_map
            & torch.isfinite(lidar_depth_map)
            & (lidar_depth_map > 1e-6)
            & (lidar_disp_map >= disp_lower)
            & (lidar_disp_map <= disp_upper)
        )
        static_map = torch.ones_like(valid_anchor)
        dynamic_mask = context.get("dynamic_mask")
        if dynamic_mask is not None:
            static_map = rearrange(
                dynamic_mask < 0.5,
                "b v 1 h w -> (b v) 1 h w",
            ).to(device=biased_disp.device)
            valid_anchor = valid_anchor & static_map

        neighbor_valid = (
            static_map
            & ~lidar_mask_map
            & reference_interior
            & torch.isfinite(reference_disp)
            & torch.isfinite(biased_disp)
            & (reference_disp > 1e-6)
            & (biased_disp > 1e-6)
        )
        neighbor_ref_patches = F.unfold(
            reference_disp,
            kernel_size=kernel_size,
            padding=radius,
        ).reshape(n, kernel_size**2, num_pixels)
        neighbor_valid_patches = F.unfold(
            neighbor_valid.to(biased_disp.dtype),
            kernel_size=kernel_size,
            padding=radius,
        ).reshape(n, kernel_size**2, num_pixels) > 0.5
        reference_interior_patches = F.unfold(
            reference_interior.to(biased_disp.dtype),
            kernel_size=kernel_size,
            padding=radius,
        ).reshape(n, kernel_size**2, num_pixels) > 0.5
        anchor_ref = reference_disp.flatten(2)
        relative_ref_error = (
            (neighbor_ref_patches - anchor_ref).abs()
            / anchor_ref.clamp_min(1e-6)
        )
        shallow_patches = F.unfold(
            shallow_feature,
            kernel_size=kernel_size,
            padding=radius,
        ).reshape(n, shallow_feature.shape[1], kernel_size**2, num_pixels)
        shallow_cosine = (
            shallow_patches * shallow_feature.flatten(2)[:, :, None, :]
        ).sum(dim=1).clamp(-1.0, 1.0)
        refine_patches = F.unfold(
            refine_feature,
            kernel_size=kernel_size,
            padding=radius,
        ).reshape(n, refine_feature.shape[1], kernel_size**2, num_pixels)
        refine_cosine = (
            refine_patches * refine_feature.flatten(2)[:, :, None, :]
        ).sum(dim=1).clamp(-1.0, 1.0)
        rgb_patches = F.unfold(
            rgb_feature,
            kernel_size=kernel_size,
            padding=radius,
        ).reshape(n, rgb_feature.shape[1], kernel_size**2, num_pixels)
        rgb_difference = (
            rgb_patches - rgb_feature.flatten(2)[:, :, None, :]
        ).abs().mean(dim=1)
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=biased_disp.device),
            torch.arange(-radius, radius + 1, device=biased_disp.device),
            indexing="ij",
        )
        offsets_x = offsets_x.reshape(-1)
        offsets_y = offsets_y.reshape(-1)
        spatial_distance_sq = (
            offsets_x.square() + offsets_y.square()
        ).to(biased_disp.dtype)
        center_index = kernel_size**2 // 2
        geometric_candidate = (
            valid_anchor.flatten(2)
            & reference_interior.flatten(2)
            & neighbor_valid_patches
            & reference_interior_patches
            & (
                relative_ref_error
                < float(
                    self.cfg.lidar_neighbor_depth_max_relative_ref_disparity_error
                )
            )
        )
        shallow_candidate = geometric_candidate & (
            shallow_cosine
            >= float(self.cfg.lidar_neighbor_depth_min_shallow_cosine)
        )
        refine_candidate = shallow_candidate & (
            refine_cosine
            >= float(self.cfg.lidar_neighbor_depth_min_refine_cosine)
        )
        candidate_valid = refine_candidate & (
            rgb_difference
            <= float(self.cfg.lidar_neighbor_depth_max_rgb_difference)
        )
        candidate_valid[:, center_index] = False
        spatial_score = torch.exp(
            -spatial_distance_sq
            / max(float(radius * radius), 1.0)
        )[None, :, None]
        disparity_scale = max(
            float(
                self.cfg.lidar_neighbor_depth_max_relative_ref_disparity_error
            ) / 2.0,
            1e-6,
        )
        # Select candidates using only appearance and geometry. Pixel distance
        # must not decide which neighbors enter top-k.
        semantic_score = torch.exp(-relative_ref_error / disparity_scale)
        semantic_score = semantic_score * ((shallow_cosine + 1.0) * 0.5)
        semantic_score = semantic_score * ((refine_cosine + 1.0) * 0.5)
        rgb_scale = max(
            float(self.cfg.lidar_neighbor_depth_max_rgb_difference) / 2.0,
            1e-6,
        )
        semantic_score = semantic_score * torch.exp(-rgb_difference / rgb_scale)
        semantic_score = semantic_score * candidate_valid.to(semantic_score.dtype)
        top_semantic, top_patch_index = semantic_score.topk(topk, dim=1)
        valid_top = top_semantic > 0
        valid_count = valid_top.sum(dim=1, keepdim=True)
        min_neighbors = int(
            self.cfg.lidar_neighbor_depth_min_neighbors_before_gap
        )
        min_log_gap = float(
            self.cfg.lidar_neighbor_depth_min_log_score_gap
        )
        if topk > 1:
            log_score = torch.log(top_semantic.clamp_min(1e-12))
            log_gap = log_score[:, :-1] - log_score[:, 1:]
            gap_slot = torch.arange(
                topk - 1,
                device=top_semantic.device,
            )[None, :, None]
            eligible_gap = (
                (gap_slot >= min_neighbors - 1)
                & valid_top[:, :-1]
                & valid_top[:, 1:]
            )
            masked_gap = log_gap.masked_fill(~eligible_gap, float("-inf"))
            largest_gap, largest_gap_slot = masked_gap.max(dim=1, keepdim=True)
            gap_keep_count = largest_gap_slot + 1
            use_gap = largest_gap >= min_log_gap
            keep_count = torch.where(use_gap, gap_keep_count, valid_count)
        else:
            use_gap = torch.zeros_like(valid_count, dtype=torch.bool)
            keep_count = valid_count
        top_slot_grid = torch.arange(
            topk,
            device=top_semantic.device,
        )[None, :, None]
        selected_mask = valid_top & (top_slot_grid < keep_count)
        selected = selected_mask.nonzero(as_tuple=False)

        if selected.numel() == 0:
            diagnostic_interval = int(
                self.cfg.lidar_neighbor_depth_diagnostic_every_n_steps
            )
            distributed_rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            if (
                diagnostic_interval > 0
                and global_step % diagnostic_interval == 0
                and self._last_lidar_neighbor_depth_diagnostic_step != global_step
                and distributed_rank == 0
            ):
                self._last_lidar_neighbor_depth_diagnostic_step = global_step
                print(
                    "\n[NeighborDepth Global] "
                    f"step={global_step} no_selected_neighbors "
                    f"valid_anchors={int(valid_anchor.sum().item())} "
                    f"interior_anchors="
                    f"{int((valid_anchor & reference_interior).sum().item())} "
                    f"geometric_pairs={int(geometric_candidate.sum().item())} "
                    f"after_shallow={int(shallow_candidate.sum().item())} "
                    f"after_refine={int(refine_candidate.sum().item())} "
                    f"after_rgb={int(candidate_valid.sum().item())}"
                )
            zero_link = next(self.lidar_neighbor_depth_mlp.parameters()).sum() * 0.0
            zeros = biased_disp.new_zeros(biased_disp.shape) + zero_link
            return zeros, zeros, zeros, zeros

        sample_index = selected[:, 0]
        top_slot = selected[:, 1]
        anchor_index = selected[:, 2]
        patch_index = top_patch_index[sample_index, top_slot, anchor_index]
        pair_semantic = top_semantic[sample_index, top_slot, anchor_index]
        anchor_x = anchor_index % width
        anchor_y = anchor_index // width
        offset_x = offsets_x[patch_index]
        offset_y = offsets_y[patch_index]
        pair_spatial = spatial_score[0, patch_index, 0]
        # Distance is used only when several anchors are aggregated for the
        # same neighbor. It does not affect top-k or the per-pair MLP gate.
        support = pair_semantic * pair_spatial
        neighbor_x = anchor_x + offset_x
        neighbor_y = anchor_y + offset_y
        neighbor_index = neighbor_y * width + neighbor_x
        global_anchor_index = sample_index * num_pixels + anchor_index
        global_neighbor_index = sample_index * num_pixels + neighbor_index

        ref_flat = reference_disp.flatten()
        biased_flat = biased_disp.flatten()
        lidar_flat = lidar_disp_map.flatten()
        anchor_ref_value = ref_flat[global_anchor_index]
        anchor_biased_value = biased_flat[global_anchor_index]
        anchor_lidar_value = lidar_flat[global_anchor_index]
        neighbor_ref_value = ref_flat[global_neighbor_index]
        neighbor_biased_value = biased_flat[global_neighbor_index]
        biased_gap = anchor_biased_value - neighbor_biased_value
        disp_scale = torch.maximum(
            anchor_ref_value.abs(),
            neighbor_ref_value.abs(),
        ).clamp_min(1e-6)
        pair_shallow_cosine = shallow_cosine[
            sample_index, patch_index, anchor_index
        ]
        pair_refine_cosine = refine_cosine[
            sample_index, patch_index, anchor_index
        ]
        pair_rgb_difference = rgb_difference[
            sample_index, patch_index, anchor_index
        ]
        pair_features = torch.cat(
            (
                (anchor_ref_value / disp_scale)[:, None],
                (anchor_biased_value / disp_scale)[:, None],
                (anchor_lidar_value / disp_scale)[:, None],
                (neighbor_ref_value / disp_scale)[:, None],
                (neighbor_biased_value / disp_scale)[:, None],
                (biased_gap / disp_scale)[:, None],
                pair_shallow_cosine[:, None],
                pair_refine_cosine[:, None],
                pair_rgb_difference[:, None],
                pair_semantic[:, None],
            ),
            dim=-1,
        ).detach()
        # Each pair independently decides how far to move the neighbor toward
        # the LiDAR-biased anchor. Direction comes from the disparity gap, so a
        # selected neighbor cannot be pushed away from that anchor.
        raw_proposal = self.lidar_neighbor_depth_mlp(pair_features)
        pair_gate = (
            float(self.cfg.lidar_neighbor_depth_max_gate)
            * torch.sigmoid(raw_proposal)
        )
        pair_residual = pair_gate * biased_gap
        global_size = n * num_pixels
        best_support = support.new_zeros(global_size)
        best_support.scatter_reduce_(
            0,
            global_neighbor_index,
            support,
            reduce="amax",
            include_self=True,
        )
        weighted_gate_sum = pair_gate.new_zeros(global_size)
        weighted_target_sum = pair_gate.new_zeros(global_size)
        weighted_target_square_sum = pair_gate.new_zeros(global_size)
        support_sum = pair_gate.new_zeros(global_size)
        weighted_gate_sum.scatter_add_(
            0,
            global_neighbor_index,
            pair_gate * support,
        )
        weighted_target_sum.scatter_add_(
            0,
            global_neighbor_index,
            anchor_biased_value * support,
        )
        weighted_target_square_sum.scatter_add_(
            0,
            global_neighbor_index,
            anchor_biased_value.square() * support,
        )
        support_sum.scatter_add_(
            0,
            global_neighbor_index,
            support,
        )
        aggregate_gate = weighted_gate_sum / support_sum.clamp_min(1e-6)
        target_global = weighted_target_sum / support_sum.clamp_min(1e-6)
        target_variance = (
            weighted_target_square_sum / support_sum.clamp_min(1e-6)
            - target_global.square()
        ).clamp_min(0.0)
        relative_target_std = (
            target_variance.sqrt() / target_global.abs().clamp_min(1e-6)
        )
        target_agreement = torch.exp(
            -relative_target_std
            / float(self.cfg.lidar_neighbor_depth_target_disagreement_scale)
        )
        biased_global = biased_disp.flatten()
        residual_global = (
            aggregate_gate * target_agreement * (target_global - biased_global)
        )
        residual_map = residual_global
        selected_support = best_support
        residual_map = residual_map.reshape(n, 1, height, width)
        selected_support = selected_support.reshape(n, 1, height, width)
        target_map = target_global.reshape(n, 1, height, width)
        attraction_confidence = (
            best_support * target_agreement
        ).reshape(n, 1, height, width)

        diagnostic_interval = int(
            self.cfg.lidar_neighbor_depth_diagnostic_every_n_steps
        )
        distributed_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available()
            and torch.distributed.is_initialized()
            else 0
        )
        should_diagnose = (
            diagnostic_interval > 0
            and global_step % diagnostic_interval == 0
            and self._last_lidar_neighbor_depth_diagnostic_step != global_step
            and distributed_rank == 0
        )
        if should_diagnose:
            self._last_lidar_neighbor_depth_diagnostic_step = global_step
            with torch.no_grad():
                selected_neighbor = support_sum > 0
                pair_count = support.new_zeros(global_size)
                pair_count.scatter_add_(
                    0,
                    global_neighbor_index,
                    torch.ones_like(support),
                )
                selected_relative = (
                    residual_global[selected_neighbor].abs()
                    / biased_global[selected_neighbor].clamp_min(1e-6)
                )
                before_disp = biased_global
                proposed_disp = before_disp + residual_map.flatten()
                near_disp = rearrange(
                    context["near"].reciprocal(),
                    "b v -> (b v) 1",
                ).expand(n, num_pixels).reshape(-1)
                far_disp = rearrange(
                    context["far"].reciprocal(),
                    "b v -> (b v) 1",
                ).expand(n, num_pixels).reshape(-1)
                after_disp = proposed_disp.clamp(min=far_disp, max=near_disp)
                distance_before = (target_global - before_disp).abs()
                distance_after = (target_global - after_disp).abs()
                attraction_ratio = 1.0 - (
                    distance_after / distance_before.clamp_min(1e-6)
                )
                before_depth = before_disp.clamp_min(1e-6).reciprocal()
                after_depth = after_disp.clamp_min(1e-6).reciprocal()
                depth_delta = after_depth - before_depth
                clamp_mask = selected_neighbor & (
                    (proposed_disp < far_disp) | (proposed_disp > near_disp)
                )

                anchor_support_sum = top_semantic.sum(dim=1)
                representative_flat = anchor_support_sum.reshape(-1).argmax()
                representative_sample = representative_flat // num_pixels
                representative_anchor = representative_flat % num_pixels
                representative_y = representative_anchor // width
                representative_x = representative_anchor % width
                representative_batch = representative_sample // v
                representative_view = representative_sample % v
                representative_global = (
                    representative_sample * num_pixels + representative_anchor
                )
                representative_pairs = (
                    (sample_index == representative_sample)
                    & (anchor_index == representative_anchor)
                ).nonzero(as_tuple=False).flatten()

                quantiles = torch.quantile(
                    selected_relative,
                    selected_relative.new_tensor([0.5, 0.9, 0.99]),
                )
                gate_quantiles = torch.quantile(
                    pair_gate.detach(),
                    pair_gate.new_tensor([0.5, 0.9, 0.99]),
                )
                selected_before_depth = before_depth[selected_neighbor]
                selected_after_depth = after_depth[selected_neighbor]
                anchors_with_neighbors = (semantic_score.sum(dim=1) > 0).sum()
                unique_neighbors = selected_neighbor.sum()
                print(
                    "\n[NeighborDepth Global] "
                    f"step={global_step} "
                    f"valid_anchors={int(valid_anchor.sum().item())} "
                    f"interior_anchors="
                    f"{int((valid_anchor & reference_interior).sum().item())} "
                    f"geometric_pairs={int(geometric_candidate.sum().item())} "
                    f"after_shallow={int(shallow_candidate.sum().item())} "
                    f"after_refine={int(refine_candidate.sum().item())} "
                    f"after_rgb={int(candidate_valid.sum().item())} "
                    f"anchors_with_neighbors={int(anchors_with_neighbors.item())} "
                    f"gap_truncated_anchors={int(use_gap.sum().item())} "
                    f"selected_pairs={selected.shape[0]} "
                    f"unique_neighbors={int(unique_neighbors.item())} "
                    f"mean_anchor_supports_per_neighbor="
                    f"{pair_count[selected_neighbor].mean().item():.3f} "
                    f"depth_before_mean/median="
                    f"{selected_before_depth.mean().item():.3f}/"
                    f"{selected_before_depth.median().item():.3f}m "
                    f"depth_after_mean/median="
                    f"{selected_after_depth.mean().item():.3f}/"
                    f"{selected_after_depth.median().item():.3f}m "
                    f"abs_rel_delta_p50/p90/p99="
                    f"{quantiles[0].item():.4f}/"
                    f"{quantiles[1].item():.4f}/"
                    f"{quantiles[2].item():.4f} "
                    f"gate_p50/p90/p99="
                    f"{gate_quantiles[0].item():.4f}/"
                    f"{gate_quantiles[1].item():.4f}/"
                    f"{gate_quantiles[2].item():.4f} "
                    f"attraction_ratio_mean="
                    f"{attraction_ratio[selected_neighbor].mean().item():.4f} "
                    f"near_far_clamp_ratio="
                    f"{clamp_mask.sum().float().div(unique_neighbors).item():.4f}"
                )
                print(
                    "[NeighborDepth Representative] "
                    f"step={global_step} "
                    f"batch={int(representative_batch.item())} "
                    f"view={int(representative_view.item())} "
                    f"anchor=({int(representative_x.item())},"
                    f"{int(representative_y.item())}) "
                    f"static={bool(static_map.flatten()[representative_global].item())} "
                    f"lidar_depth={lidar_depth_map.flatten()[representative_global].item():.3f}m "
                    f"lidar_disp={lidar_flat[representative_global].item():.6f} "
                    f"reference_disp={ref_flat[representative_global].item():.6f} "
                    f"biased_disp={biased_flat[representative_global].item():.6f}"
                )
                print(
                    "neighbor offset shallow_cos refine_cos rgb_diff semantic "
                    "aggregate_weight "
                    "raw_mlp gate anchor_target "
                    "pair_delta final_delta disp_before disp_after distance_before "
                    "distance_after attraction depth_before depth_after depth_delta anchors"
                )
                for pair in representative_pairs.tolist():
                    neighbor_global = int(global_neighbor_index[pair].item())
                    print(
                        f"({int(neighbor_x[pair].item())},"
                        f"{int(neighbor_y[pair].item())}) "
                        f"({int(offset_x[pair].item()):+d},"
                        f"{int(offset_y[pair].item()):+d}) "
                        f"{pair_shallow_cosine[pair].item():.6f} "
                        f"{pair_refine_cosine[pair].item():.6f} "
                        f"{pair_rgb_difference[pair].item():.4f} "
                        f"{pair_semantic[pair].item():.3f} "
                        f"{support[pair].item():.3f} "
                        f"{raw_proposal[pair].item():+.4f} "
                        f"{pair_gate[pair].item():.4f} "
                        f"{anchor_biased_value[pair].item():.6f} "
                        f"{pair_residual[pair].item():+.6f} "
                        f"{residual_global[neighbor_global].item():+.6f} "
                        f"{before_disp[neighbor_global].item():.6f} "
                        f"{after_disp[neighbor_global].item():.6f} "
                        f"{distance_before[neighbor_global].item():.6f} "
                        f"{distance_after[neighbor_global].item():.6f} "
                        f"{attraction_ratio[neighbor_global].item():+.3f} "
                        f"{before_depth[neighbor_global].item():.3f} "
                        f"{after_depth[neighbor_global].item():.3f} "
                        f"{depth_delta[neighbor_global].item():+.3f} "
                        f"{int(pair_count[neighbor_global].item())}"
                    )

                y0 = max(int(representative_y.item()) - radius, 0)
                y1 = min(int(representative_y.item()) + radius + 1, height)
                x0 = max(int(representative_x.item()) - radius, 0)
                x1 = min(int(representative_x.item()) + radius + 1, width)
                sample_offset = int(representative_sample.item()) * num_pixels

                def format_grid(values: Tensor, fmt: str) -> str:
                    grid = values[sample_offset : sample_offset + num_pixels].reshape(
                        height, width
                    )[y0:y1, x0:x1]
                    return "\n".join(
                        " ".join(format(value.item(), fmt) for value in row)
                        for row in grid
                    )

                print("before depth (m):\n" + format_grid(before_depth, ".2f"))
                print("depth delta (m):\n" + format_grid(depth_delta, "+.2f"))
                print(
                    "selected mask:\n"
                    + format_grid(selected_neighbor.to(torch.int32), "d")
                )
        return (
            residual_map,
            selected_support.detach(),
            target_map.detach(),
            attraction_confidence.detach(),
        )

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
        
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        if self.cfg.use_epipolar_trans:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = None
        trans_features, cnn_features = self.backbone(
            context["image"],
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )

        # Sample depths from the resulting features.
        in_feats = trans_features
        extra_info = {}
        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel
        (
            depths,
            densities,
            raw_gaussians,
            raw_gaussians_base,
        ) = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
            lidar_depth=context.get("lidar_depth", None),
            lidar_mask=context.get("lidar_mask", None),
        )

        neighbor_depth_base_depths = None
        neighbor_depth_residual = None
        neighbor_depth_support = None
        neighbor_depth_target = None
        neighbor_depth_attraction_confidence = None
        if self.lidar_neighbor_depth_repair_enabled:
            depth_aux = self.depth_predictor.lidar_depth_repair_aux
            if depth_aux is None:
                raise RuntimeError(
                    "LiDAR neighbor depth repair requires frozen visual/base "
                    "disparity references from the depth predictor."
                )
            neighbor_depth_base_depths = depths.detach()
            (
                neighbor_depth_residual,
                neighbor_depth_support,
                neighbor_depth_target,
                neighbor_depth_attraction_confidence,
            ) = self._predict_lidar_neighbor_depth_residual(
                context,
                depth_aux,
                h,
                w,
                global_step,
            )
            base_disparity = rearrange(
                depths[..., 0, 0].reciprocal(),
                "b v (h w) -> (b v) 1 h w",
                h=h,
                w=w,
            ).detach()
            disp_min = rearrange(
                context["far"].reciprocal(),
                "b v -> (b v) 1 1 1",
            )
            disp_max = rearrange(
                context["near"].reciprocal(),
                "b v -> (b v) 1 1 1",
            )
            repaired_disparity = (
                base_disparity + neighbor_depth_residual
            ).clamp(min=disp_min, max=disp_max)
            repaired_depth = repaired_disparity.reciprocal()
            depths = rearrange(
                repaired_depth,
                "(b v) 1 h w -> b v (h w) 1 1",
                b=b,
                v=v,
            )

        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        # The depth predictor and LiDAR branches use camera z-depth, while the
        # shared GaussianAdapter multiplies depth by unit-length world rays.
        # Convert every predicted z-depth to a ray distance using the same
        # offset pixel coordinates that the adapter uses to construct rays.
        homogeneous_xy = torch.cat(
            (xy_ray, torch.ones_like(xy_ray[..., :1])),
            dim=-1,
        )
        camera_rays = torch.linalg.solve(
            rearrange(
                context["intrinsics"],
                "b v i j -> b v () () i j",
            ),
            homogeneous_xy.unsqueeze(-1),
        ).squeeze(-1)
        ray_norm = camera_rays.norm(dim=-1, keepdim=True)
        gaussian_depths = depths * ray_norm

        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            gaussian_depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )
        opacity_logit_residual = getattr(
            self.depth_predictor,
            "lidar_gaussian_opacity_logit_residual",
            None,
        )
        if opacity_logit_residual is not None:
            base_opacity = gaussians.opacities.clamp(1e-6, 1.0 - 1e-6)
            corrected_opacity = torch.sigmoid(
                torch.logit(base_opacity)
                + opacity_logit_residual.unsqueeze(-1)
            )
            gaussians.opacities = corrected_opacity

        neighbor_depth_base_gaussians = None
        if neighbor_depth_base_depths is not None:
            with torch.no_grad():
                base_gaussian_depths = neighbor_depth_base_depths * ray_norm
                neighbor_depth_base_gaussians = self.gaussian_adapter.forward(
                    rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                    rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    base_gaussian_depths,
                    self.map_pdf_to_opacity(densities.detach(), global_step) / gpp,
                    rearrange(
                        rearrange(
                            raw_gaussians,
                            "b v r (srf c) -> b v r srf c",
                            srf=self.cfg.num_surfaces,
                        )[..., 2:].detach(),
                        "b v r srf c -> b v r srf () c",
                    ),
                    (h, w),
                )
                if opacity_logit_residual is not None:
                    base_opacity = neighbor_depth_base_gaussians.opacities.clamp(
                        1e-6, 1.0 - 1e-6
                    )
                    neighbor_depth_base_gaussians.opacities = torch.sigmoid(
                        torch.logit(base_opacity)
                        + opacity_logit_residual.detach().unsqueeze(-1)
                    )

        # Adapter-loss baseline: keep the corrected depths and densities, but
        # remove the Gaussian adapter residual itself.
        base_gaussians = None
        if self.cfg.use_lidar_gaussian_adapter_loss:
            with torch.no_grad():
                base_raw = rearrange(
                    raw_gaussians_base,
                    "... (srf c) -> ... srf c",
                    srf=self.cfg.num_surfaces,
                )
                base_xy_ray, _ = sample_image_grid((h, w), device)
                base_xy_ray = rearrange(base_xy_ray, "h w xy -> (h w) () xy")
                base_offset_xy = base_raw[..., :2].sigmoid()
                base_xy_ray = (
                    base_xy_ray
                    + (base_offset_xy - 0.5) * pixel_size
                )
                base_homogeneous_xy = torch.cat(
                    (
                        base_xy_ray,
                        torch.ones_like(base_xy_ray[..., :1]),
                    ),
                    dim=-1,
                )
                base_camera_rays = torch.linalg.solve(
                    rearrange(
                        context["intrinsics"],
                        "b v i j -> b v () () i j",
                    ),
                    base_homogeneous_xy.unsqueeze(-1),
                ).squeeze(-1)
                base_gaussian_depths = (
                    depths.detach()
                    * base_camera_rays.norm(dim=-1, keepdim=True)
                )
                base_gaussians = self.gaussian_adapter.forward(
                    rearrange(
                        context["extrinsics"],
                        "b v i j -> b v () () () i j",
                    ),
                    rearrange(
                        context["intrinsics"],
                        "b v i j -> b v () () () i j",
                    ),
                    rearrange(
                        base_xy_ray,
                        "b v r srf xy -> b v r srf () xy",
                    ),
                    base_gaussian_depths,
                    self.map_pdf_to_opacity(densities.detach(), global_step) / gpp,
                    rearrange(
                        base_raw[..., 2:],
                        "b v r srf c -> b v r srf () c",
                    ),
                    (h, w),
                )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        output_gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )
        if neighbor_depth_base_gaussians is not None:
            output_gaussians.lidar_neighbor_depth_base_gaussians = Gaussians(
                rearrange(neighbor_depth_base_gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"),
                rearrange(neighbor_depth_base_gaussians.covariances, "b v r srf spp i j -> b (v r srf spp) i j"),
                rearrange(neighbor_depth_base_gaussians.harmonics, "b v r srf spp c d -> b (v r srf spp) c d"),
                rearrange(neighbor_depth_base_gaussians.opacities, "b v r srf spp -> b (v r srf spp)"),
            )
            output_gaussians.lidar_neighbor_depth_mask = rearrange(
                neighbor_depth_support > 0,
                "(b v) 1 h w -> b (v h w)",
                b=b,
                v=v,
            ).detach()
            output_gaussians.lidar_neighbor_depth_residual = (
                neighbor_depth_residual
            )
            output_gaussians.lidar_neighbor_depth_target_disparity = rearrange(
                neighbor_depth_target,
                "(b v) 1 h w -> b (v h w)",
                b=b,
                v=v,
            )
            output_gaussians.lidar_neighbor_depth_attraction_confidence = rearrange(
                neighbor_depth_attraction_confidence,
                "(b v) 1 h w -> b (v h w)",
                b=b,
                v=v,
            )
            output_gaussians.lidar_neighbor_depth_base_disparity = rearrange(
                neighbor_depth_base_depths[..., 0, 0].reciprocal(),
                "b v r -> b (v r)",
            )
        if base_gaussians is not None:
            output_gaussians.lidar_base_gaussians = Gaussians(
                rearrange(
                    base_gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    base_gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    base_gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    opacity_multiplier * base_gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
            )
        return output_gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
