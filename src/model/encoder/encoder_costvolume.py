from dataclasses import dataclass
from typing import Literal, Optional, List
import math

import torch
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
from .common.gaussians import build_covariance
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
    use_lidar_refine_loss: bool
    use_lidar_world_mean_anchor: bool
    lidar_world_anchor_rescale_covariance: bool
    lidar_world_anchor_scale_ratio_min: float
    lidar_world_anchor_scale_ratio_max: float
    use_lidar_post_gaussian_adapter: bool
    lidar_post_gaussian_adapter_hidden_dim: int
    lidar_post_gaussian_attention_heads: int
    lidar_post_gaussian_window_size: int
    lidar_post_gaussian_max_scale_factor: float
    lidar_post_gaussian_max_rotation_degrees: float
    lidar_post_gaussian_max_sh_residual: float
    use_lidar_cross_attention: bool
    lidar_cross_attention_dim: int
    lidar_cross_attention_heads: int
    lidar_cross_attention_radius: int
    lidar_cross_attention_max_delta_logit: float
    lidar_cross_attention_inference_mode: str
    frozen_params: list[str]
    lidar_final_loss_weight: float


class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)
        if cfg.use_lidar_world_mean_anchor:
            if cfg.num_surfaces != 1 or cfg.gaussians_per_pixel != 1:
                raise ValueError(
                    "LiDAR world mean anchoring currently requires "
                    "num_surfaces=1 and gaussians_per_pixel=1."
                )
            if get_cfg().dataset.augment:
                raise ValueError(
                    "LiDAR world mean anchoring requires dataset.augment=false "
                    "until LiDAR maps participate in the augmentation shim."
                )
        if cfg.use_lidar_post_gaussian_adapter:
            if not cfg.use_lidar_world_mean_anchor:
                raise ValueError(
                    "The post-Gaussian LiDAR adapter requires "
                    "use_lidar_world_mean_anchor=true."
                )
            if not cfg.lidar_world_anchor_rescale_covariance:
                raise ValueError(
                    "The post-Gaussian LiDAR adapter requires deterministic "
                    "anchor scale/covariance rescaling."
                )
            if cfg.lidar_post_gaussian_max_scale_factor < 1.0:
                raise ValueError(
                    "lidar_post_gaussian_max_scale_factor must be >= 1."
                )
            if cfg.lidar_post_gaussian_window_size % 2 != 1:
                raise ValueError("lidar_post_gaussian_window_size must be odd.")
            if (
                cfg.lidar_post_gaussian_adapter_hidden_dim
                % cfg.lidar_post_gaussian_attention_heads
                != 0
            ):
                raise ValueError(
                    "Post-Gaussian adapter hidden dim must be divisible by "
                    "its attention head count."
                )
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
        if cfg.use_lidar_post_gaussian_adapter:
            # Query: visual feature/state plus the center LiDAR discrepancy.
            query_dim = cfg.d_feature + 3 + 4 + 1 + 3 + 3 + 3 + 3 + 1 + 2 + 1
            # Neighbor: visual feature/state, relative geometry, and optional
            # LiDAR discrepancy (zero-filled and disambiguated by lidar_valid).
            neighbor_dim = (
                cfg.d_feature + 3 + 4 + 1 + 3 + 2 + 3 + 1 + 1 + 1 + 3 + 1
            )
            hidden_dim = cfg.lidar_post_gaussian_adapter_hidden_dim
            self.lidar_post_query_projection = nn.Linear(query_dim, hidden_dim)
            self.lidar_post_neighbor_projection = nn.Linear(
                neighbor_dim, hidden_dim
            )
            self.lidar_post_attention = nn.MultiheadAttention(
                hidden_dim,
                cfg.lidar_post_gaussian_attention_heads,
                batch_first=True,
            )
            self.lidar_post_attention_norm = nn.LayerNorm(hidden_dim)
            self.lidar_post_ffn = nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.GELU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.lidar_post_ffn_norm = nn.LayerNorm(hidden_dim)
            # Only DC color is corrected in the conservative first version.
            self.lidar_post_output = nn.Linear(hidden_dim, 3 + 3 + 3)
            # Start as the exact deterministic anchor. Learning only adds bounded
            # residuals, so loading/training begins without changing its geometry.
            nn.init.zeros_(self.lidar_post_output.weight)
            nn.init.zeros_(self.lidar_post_output.bias)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            
            use_lidar_refine_loss=cfg.use_lidar_refine_loss,
            use_lidar_cross_attention=cfg.use_lidar_cross_attention,
            lidar_cross_attention_dim=cfg.lidar_cross_attention_dim,
            lidar_cross_attention_heads=cfg.lidar_cross_attention_heads,
            lidar_cross_attention_radius=cfg.lidar_cross_attention_radius,
            lidar_cross_attention_max_delta_logit=(
                cfg.lidar_cross_attention_max_delta_logit
            ),
            lidar_cross_attention_inference_mode=(
                cfg.lidar_cross_attention_inference_mode
            ),
        )

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
        alpha_diagnostic_gaussians = None

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
            lidar_refine_loss,
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
        gpp = self.cfg.gaussians_per_pixel
        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

        if self.cfg.use_lidar_world_mean_anchor:
            if "lidar_world" not in context or "lidar_mask" not in context:
                raise RuntimeError(
                    "LiDAR world mean anchoring requires context lidar_world "
                    "and lidar_mask tensors."
                )
            lidar_world = rearrange(
                context["lidar_world"],
                "b v xyz h w -> b v (h w) () () xyz",
            ).to(device=gaussians.means.device, dtype=gaussians.means.dtype)
            lidar_valid = rearrange(
                context["lidar_mask"] > 0.5,
                "b v c h w -> b v (h w) () () c",
            ).to(device=gaussians.means.device)
            lidar_valid = lidar_valid & torch.isfinite(lidar_world).all(
                dim=-1, keepdim=True
            )

            visual_means = gaussians.means
            camera_origins = rearrange(
                context["extrinsics"][..., :3, 3],
                "b v xyz -> b v () () () xyz",
            ).to(device=visual_means.device, dtype=visual_means.dtype)
            visual_distance = (visual_means - camera_origins).norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            lidar_distance = (lidar_world - camera_origins).norm(
                dim=-1, keepdim=True
            )
            raw_scale_ratio = lidar_distance / visual_distance
            scale_ratio = raw_scale_ratio.clamp(
                min=self.cfg.lidar_world_anchor_scale_ratio_min,
                max=self.cfg.lidar_world_anchor_scale_ratio_max,
            )

            anchored_means = torch.where(
                lidar_valid,
                lidar_world,
                visual_means,
            )
            anchored_covariances = gaussians.covariances
            anchored_scales = gaussians.scales
            rescaled_covariances = gaussians.covariances
            if (
                self.cfg.lidar_world_anchor_rescale_covariance
                or not self.training
            ):
                rescaled_covariances = torch.where(
                    lidar_valid[..., None],
                    gaussians.covariances * scale_ratio.square()[..., None],
                    gaussians.covariances,
                )
            if self.cfg.lidar_world_anchor_rescale_covariance:
                anchored_covariances = rescaled_covariances
                anchored_scales = torch.where(
                    lidar_valid,
                    gaussians.scales * scale_ratio,
                    gaussians.scales,
                )

            anchored_rotations = gaussians.rotations
            anchored_harmonics = gaussians.harmonics
            if self.cfg.use_lidar_post_gaussian_adapter:
                visual_vector = visual_means - camera_origins
                lidar_vector = lidar_world - camera_origins
                visual_direction = visual_vector / visual_distance
                lidar_direction = lidar_vector / lidar_distance.clamp_min(1e-6)
                c2w_rotation = rearrange(
                    context["extrinsics"][..., :3, :3],
                    "b v i j -> b v () () () i j",
                ).to(device=visual_means.device, dtype=visual_means.dtype)
                w2c_rotation = c2w_rotation.transpose(-1, -2)

                def world_to_camera(vector: Tensor) -> Tensor:
                    return (w2c_rotation @ vector[..., None]).squeeze(-1)

                normalized_shift_camera = world_to_camera(
                    (lidar_world - visual_means) / visual_distance
                )
                visual_direction_camera = world_to_camera(visual_direction)
                lidar_direction_camera = world_to_camera(lidar_direction)
                log_distance_ratio = raw_scale_ratio.clamp_min(1e-6).log()

                post_features = torch.nn.functional.interpolate(
                    rearrange(trans_features, "b v c fh fw -> (b v) c fh fw"),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                post_features = rearrange(
                    post_features, "(b v) c h w -> b v (h w) () () c", b=b, v=v
                )
                normalized_xy = rearrange(
                    xy_ray, "b v r srf xy -> b v r srf () xy"
                )
                center_query_map = torch.cat(
                    (
                        post_features,
                        gaussians.scales,
                        gaussians.rotations,
                        gaussians.opacities[..., None],
                        gaussians.harmonics[..., 0],
                        visual_direction_camera,
                        lidar_direction_camera,
                        normalized_shift_camera,
                        log_distance_ratio,
                        normalized_xy,
                        lidar_valid.to(visual_means.dtype),
                    ),
                    dim=-1,
                )
                valid_tokens = lidar_valid.squeeze(-1)
                residual = center_query_map.new_zeros(
                    (*center_query_map.shape[:-1], 9)
                )

                # Only LiDAR Gaussians become queries. Each query attends to all
                # visual Gaussians in its local image-plane window; neighbors are
                # context only and are never modified here.
                if valid_tokens.any():
                    valid_positions = valid_tokens.nonzero(as_tuple=False)
                    batch_view = valid_positions[:, 0] * v + valid_positions[:, 1]
                    center_linear = valid_positions[:, 2]
                    center_y = torch.div(center_linear, w, rounding_mode="floor")
                    center_x = center_linear.remainder(w)

                    window_size = self.cfg.lidar_post_gaussian_window_size
                    radius = window_size // 2
                    offsets_y, offsets_x = torch.meshgrid(
                        torch.arange(
                            -radius, radius + 1, device=visual_means.device
                        ),
                        torch.arange(
                            -radius, radius + 1, device=visual_means.device
                        ),
                        indexing="ij",
                    )
                    offsets_y = offsets_y.flatten()
                    offsets_x = offsets_x.flatten()
                    neighbor_y_unclamped = center_y[:, None] + offsets_y[None]
                    neighbor_x_unclamped = center_x[:, None] + offsets_x[None]
                    neighbor_padding = (
                        (neighbor_y_unclamped < 0)
                        | (neighbor_y_unclamped >= h)
                        | (neighbor_x_unclamped < 0)
                        | (neighbor_x_unclamped >= w)
                    )
                    neighbor_y = neighbor_y_unclamped.clamp(0, h - 1)
                    neighbor_x = neighbor_x_unclamped.clamp(0, w - 1)

                    def as_dense_map(tensor: Tensor) -> Tensor:
                        return rearrange(
                            tensor,
                            "b v (h w) () () c -> (b v) h w c",
                            h=h,
                            w=w,
                        )

                    def gather_neighbors(tensor: Tensor) -> Tensor:
                        dense = as_dense_map(tensor)
                        return dense[
                            batch_view[:, None], neighbor_y, neighbor_x
                        ]

                    center_query = center_query_map[valid_tokens]
                    center_mean = visual_means[valid_tokens]
                    center_distance = visual_distance[valid_tokens]
                    center_direction = visual_direction[valid_tokens]
                    center_w2c = w2c_rotation[
                        valid_positions[:, 0], valid_positions[:, 1], 0, 0, 0
                    ]

                    neighbor_mean = gather_neighbors(visual_means)
                    relative_mean_world = neighbor_mean - center_mean[:, None]
                    relative_mean_camera = (
                        center_w2c[:, None]
                        @ relative_mean_world[..., None]
                    ).squeeze(-1) / center_distance[:, None]
                    neighbor_distance = gather_neighbors(visual_distance)
                    relative_ray_distance = torch.log(
                        neighbor_distance / center_distance[:, None]
                    )
                    neighbor_direction = gather_neighbors(visual_direction)
                    direction_cosine = (
                        neighbor_direction * center_direction[:, None]
                    ).sum(dim=-1, keepdim=True)
                    relative_xy = torch.stack(
                        (offsets_x, offsets_y), dim=-1
                    ).to(center_query.dtype)
                    relative_xy = relative_xy / max(radius, 1)
                    relative_xy = relative_xy[None].expand(
                        center_query.shape[0], -1, -1
                    )

                    neighbor_valid = gather_neighbors(
                        lidar_valid.to(visual_means.dtype)
                    )
                    neighbor_shift = gather_neighbors(
                        normalized_shift_camera
                        * lidar_valid.to(visual_means.dtype)
                    )
                    neighbor_log_ratio = gather_neighbors(
                        log_distance_ratio
                        * lidar_valid.to(visual_means.dtype)
                    )
                    neighbor_tokens = torch.cat(
                        (
                            gather_neighbors(post_features),
                            gather_neighbors(gaussians.scales),
                            gather_neighbors(gaussians.rotations),
                            gather_neighbors(gaussians.opacities[..., None]),
                            gather_neighbors(gaussians.harmonics[..., 0]),
                            relative_xy,
                            relative_mean_camera,
                            relative_ray_distance,
                            direction_cosine,
                            neighbor_valid,
                            neighbor_shift,
                            neighbor_log_ratio,
                        ),
                        dim=-1,
                    )

                    query_embedding = self.lidar_post_query_projection(
                        center_query
                    )[:, None]
                    neighbor_embedding = self.lidar_post_neighbor_projection(
                        neighbor_tokens
                    )
                    attended, _ = self.lidar_post_attention(
                        query_embedding,
                        neighbor_embedding,
                        neighbor_embedding,
                        key_padding_mask=neighbor_padding,
                        need_weights=False,
                    )
                    adapted = self.lidar_post_attention_norm(
                        query_embedding + attended
                    )
                    adapted = self.lidar_post_ffn_norm(
                        adapted + self.lidar_post_ffn(adapted)
                    )
                    residual[valid_tokens] = self.lidar_post_output(
                        adapted.squeeze(1)
                    )

                raw_log_scale, raw_axis_angle, raw_dc = residual.split(
                    (3, 3, 3), dim=-1
                )
                max_log_scale = math.log(
                    self.cfg.lidar_post_gaussian_max_scale_factor
                )
                delta_log_scale = max_log_scale * raw_log_scale.tanh()
                anchored_scales = anchored_scales * delta_log_scale.exp()

                raw_angle = raw_axis_angle.norm(dim=-1, keepdim=True)
                angle = math.radians(
                    self.cfg.lidar_post_gaussian_max_rotation_degrees
                ) * raw_angle.tanh()
                axis_angle = raw_axis_angle / raw_angle.clamp_min(1e-8) * angle
                half_angle = 0.5 * angle
                xyz = axis_angle * (0.5 * torch.sinc(half_angle / math.pi))
                delta_rotation = torch.cat((xyz, half_angle.cos()), dim=-1)
                x1, y1, z1, w1 = delta_rotation.unbind(dim=-1)
                x2, y2, z2, w2 = anchored_rotations.unbind(dim=-1)
                anchored_rotations = torch.stack(
                    (
                        w1*x2 + x1*w2 + y1*z2 - z1*y2,
                        w1*y2 - x1*z2 + y1*w2 + z1*x2,
                        w1*z2 + x1*y2 - y1*x2 + z1*w2,
                        w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    ),
                    dim=-1,
                )
                anchored_rotations = anchored_rotations / anchored_rotations.norm(
                    dim=-1, keepdim=True
                ).clamp_min(1e-8)
                delta_dc = self.cfg.lidar_post_gaussian_max_sh_residual * (
                    raw_dc.tanh()
                )
                anchored_harmonics = anchored_harmonics.clone()
                anchored_harmonics[..., 0] = (
                    anchored_harmonics[..., 0] + delta_dc
                )

                camera_covariances = build_covariance(
                    anchored_scales, anchored_rotations
                )
                rebuilt_covariances = (
                    c2w_rotation
                    @ camera_covariances
                    @ c2w_rotation.transpose(-1, -2)
                )
                anchored_covariances = torch.where(
                    lidar_valid[..., None],
                    rebuilt_covariances,
                    gaussians.covariances,
                )

            with torch.no_grad():
                mean_shift = (lidar_world - visual_means).norm(
                    dim=-1, keepdim=True
                )
                valid_ratio = raw_scale_ratio[lidar_valid]
                clamped = (
                    (valid_ratio < self.cfg.lidar_world_anchor_scale_ratio_min)
                    | (valid_ratio > self.cfg.lidar_world_anchor_scale_ratio_max)
                )
                if lidar_valid.any():
                    print(
                        "[LiDAR World Geometry] "
                        f"points={int(lidar_valid.sum().item())}, "
                        f"mean_shift={mean_shift[lidar_valid].mean().item():.6f}, "
                        f"max_shift={mean_shift[lidar_valid].max().item():.6f}, "
                        f"scale_ratio_mean={valid_ratio.mean().item():.6f}, "
                        f"scale_ratio_min={valid_ratio.min().item():.6f}, "
                        f"scale_ratio_max={valid_ratio.max().item():.6f}, "
                        f"scale_clamp_ratio={clamped.float().mean().item():.6f}"
                    )

            # Keep the three geometry variants needed for coverage diagnostics.
            # They are only retained during evaluation and are rendered as white
            # Gaussians over black in ModelWrapper.test_step, which yields the
            # accumulated alpha independently of the predicted SH colors.
            if not self.training:
                alpha_diagnostic_gaussians = {
                    "alpha_before_anchor": (
                        visual_means,
                        gaussians.covariances,
                    ),
                    "alpha_after_anchor_no_rescale": (
                        anchored_means,
                        gaussians.covariances,
                    ),
                    "alpha_after_anchor_rescale": (
                        anchored_means,
                        rescaled_covariances,
                    ),
                }

            gaussians = type(gaussians)(
                means=anchored_means,
                covariances=anchored_covariances,
                harmonics=anchored_harmonics,
                opacities=gaussians.opacities,
                scales=anchored_scales,
                rotations=anchored_rotations,
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
        output_gaussians.lidar_refine_loss = lidar_refine_loss
        if alpha_diagnostic_gaussians is not None:
            output_gaussians.alpha_diagnostic_gaussians = {
                name: Gaussians(
                    means=rearrange(
                        means,
                        "b v r srf spp xyz -> b (v r srf spp) xyz",
                    ),
                    covariances=rearrange(
                        covariances,
                        "b v r srf spp i j -> b (v r srf spp) i j",
                    ),
                    harmonics=output_gaussians.harmonics,
                    opacities=output_gaussians.opacities,
                )
                for name, (means, covariances) in alpha_diagnostic_gaussians.items()
            }

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
