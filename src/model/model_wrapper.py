from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .types import Gaussians
from .encoder.visualization.encoder_visualizer import EncoderVisualizer


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool



@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_lidar_alpha_diagnostics: bool
    eval_time_skip_steps: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"==> Trainable params: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0
        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}
            
    def _lidar_gaussian_adapter_losses(
        self,
        edited_output,
        edited_gaussians: Gaussians,
        batch: BatchedExample,
        image_shape: tuple[int, int],
    ) -> dict[str, Tensor]:
        """Compute losses only where selected base LiDAR Gaussians contribute."""
        base_gaussians = getattr(edited_gaussians, "lidar_base_gaussians", None)
        context_mask = batch["context"].get("lidar_mask")
        if base_gaussians is None or context_mask is None:
            raise RuntimeError(
                "LiDAR Gaussian adapter loss requires base Gaussians and a context LiDAR mask."
            )

        b, context_views, _, height, width = context_mask.shape
        num_context_pixels = context_views * height * width
        num_gaussians = base_gaussians.means.shape[1]
        if num_context_pixels == 0 or num_gaussians % num_context_pixels != 0:
            raise ValueError(
                "Cannot map LiDAR context pixels to flattened Gaussians: "
                f"num_gaussians={num_gaussians}, "
                f"context_shape={(context_views, height, width)}."
            )
        gaussians_per_pixel = num_gaussians // num_context_pixels
        selected = rearrange(
            context_mask > 0.5,
            "b v 1 h w -> b (v h w)",
        ).repeat_interleave(gaussians_per_pixel, dim=1)
        selected_base_gaussians = Gaussians(
            means=base_gaussians.means,
            covariances=base_gaussians.covariances,
            harmonics=base_gaussians.harmonics,
            opacities=base_gaussians.opacities * selected.to(
                base_gaussians.opacities.dtype
            ),
        )

        render_args = (
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            image_shape,
        )
        with torch.no_grad():
            base_output = self.decoder.forward(
                base_gaussians,
                *render_args,
                depth_mode=self.train_cfg.depth_mode,
            )
            influence = self.decoder.render_alpha(
                selected_base_gaussians,
                *render_args,
            ).unsqueeze(2)
            base_alpha = self.decoder.render_alpha(
                base_gaussians,
                *render_args,
            ).unsqueeze(2)
        edited_alpha = self.decoder.render_alpha(
            edited_gaussians,
            *render_args,
        ).unsqueeze(2)

        target = batch["target"]["image"]
        valid = torch.ones_like(target[:, :, :1])
        dynamic_mask = batch["target"].get("dynamic_mask")
        if dynamic_mask is not None:
            valid = valid * (dynamic_mask < 0.5).to(
                device=valid.device,
                dtype=valid.dtype,
            )
        weight = influence.detach().clamp(0.0, 1.0) * valid
        pixel_denominator = weight.sum().clamp_min(1.0)
        rgb_denominator = (pixel_denominator * target.shape[2]).clamp_min(1.0)

        edited_abs_error = (edited_output.color - target).abs()
        base_abs_error = (base_output.color - target).abs().detach()
        local_rgb = (edited_abs_error * weight).sum() / rgb_denominator
        improvement_margin = float(
            getattr(
                self.encoder.cfg,
                "lidar_gaussian_adapter_improvement_margin",
                0.0,
            )
        )
        improvement = (
            torch.relu(
                edited_abs_error.mean(dim=2, keepdim=True)
                - base_abs_error.mean(dim=2, keepdim=True)
                + improvement_margin
            )
            * weight
        ).sum() / pixel_denominator
        alpha_worse = (
            torch.relu(
                (1.0 - edited_alpha).abs()
                - (1.0 - base_alpha).abs().detach()
            )
            * weight
        ).sum() / pixel_denominator
        return {
            "lidar_gaussian_local_rgb": local_rgb,
            "lidar_gaussian_improvement": improvement,
            "lidar_gaussian_alpha_worse": alpha_worse,
        }

    def _render_lidar_neighbor_depth_reference_gaussians(
        self,
        batch: BatchedExample,
    ) -> Gaussians:
        """Render-time RE10K reference with every LiDAR intervention disabled."""
        depth_predictor = getattr(self.encoder, "depth_predictor", None)
        if depth_predictor is None:
            raise RuntimeError("LiDAR neighbor repair requires a depth predictor.")
        original_use_lidar_bias = depth_predictor.use_lidar_bias
        original_use_anchor_adapter = depth_predictor.use_lidar_gaussian_adapter
        original_neighbor_enabled = (
            self.encoder.lidar_neighbor_depth_repair_enabled
        )
        original_compute_aux = depth_predictor.compute_lidar_depth_repair_aux
        try:
            depth_predictor.use_lidar_bias = False
            depth_predictor.use_lidar_gaussian_adapter = False
            self.encoder.lidar_neighbor_depth_repair_enabled = False
            depth_predictor.compute_lidar_depth_repair_aux = False
            with torch.no_grad():
                return self.encoder(
                    batch["context"],
                    self.global_step,
                    False,
                    scene_names=batch["scene"],
                )
        finally:
            depth_predictor.use_lidar_bias = original_use_lidar_bias
            depth_predictor.use_lidar_gaussian_adapter = (
                original_use_anchor_adapter
            )
            self.encoder.lidar_neighbor_depth_repair_enabled = (
                original_neighbor_enabled
            )
            depth_predictor.compute_lidar_depth_repair_aux = original_compute_aux

    def _lidar_neighbor_depth_losses(
        self,
        edited_output,
        edited_gaussians: Gaussians,
        batch: BatchedExample,
        image_shape: tuple[int, int],
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Train local neighbor depth repair against frozen anchor/RE10K branches."""
        base_gaussians = getattr(
            edited_gaussians,
            "lidar_neighbor_depth_base_gaussians",
            None,
        )
        repair_mask = getattr(
            edited_gaussians,
            "lidar_neighbor_depth_mask",
            None,
        )
        if base_gaussians is None or repair_mask is None:
            raise RuntimeError(
                "LiDAR neighbor depth loss requires its frozen base branch, "
                "and selection mask."
            )

        selected_base_gaussians = Gaussians(
            base_gaussians.means,
            base_gaussians.covariances,
            base_gaussians.harmonics,
            base_gaussians.opacities * repair_mask.to(
                base_gaussians.opacities.dtype
            ),
        )
        selected_edit_gaussians = Gaussians(
            edited_gaussians.means,
            edited_gaussians.covariances,
            edited_gaussians.harmonics,
            edited_gaussians.opacities * repair_mask.to(
                edited_gaussians.opacities.dtype
            ),
        )
        reference_gaussians = self._render_lidar_neighbor_depth_reference_gaussians(
            batch
        )
        render_args = (
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            image_shape,
        )
        with torch.no_grad():
            base_output = self.decoder.forward(
                base_gaussians,
                *render_args,
                depth_mode=self.train_cfg.depth_mode,
            )
            base_alpha = self.decoder.render_alpha(
                base_gaussians,
                *render_args,
            ).unsqueeze(2)
            reference_alpha = self.decoder.render_alpha(
                reference_gaussians,
                *render_args,
            ).unsqueeze(2)
            influence = self.decoder.render_alpha(
                selected_base_gaussians,
                *render_args,
            ).unsqueeze(2)
            edited_influence = self.decoder.render_alpha(
                selected_edit_gaussians,
                *render_args,
            ).unsqueeze(2)
            influence = torch.maximum(influence, edited_influence)
        edited_alpha = self.decoder.render_alpha(
            edited_gaussians,
            *render_args,
        ).unsqueeze(2)

        target = batch["target"]["image"]
        valid = torch.ones_like(target[:, :, :1])
        dynamic_mask = batch["target"].get("dynamic_mask")
        if dynamic_mask is not None:
            valid = valid * (dynamic_mask < 0.5).to(
                device=valid.device,
                dtype=valid.dtype,
            )
        weight = influence.detach().clamp(0.0, 1.0) * valid
        pixel_denominator = weight.sum().clamp_min(1.0)
        rgb_denominator = (pixel_denominator * target.shape[2]).clamp_min(1.0)
        edited_abs_error = (edited_output.color - target).abs()
        base_abs_error = (base_output.color - target).abs().detach()
        local_rgb = (edited_abs_error * weight).sum() / rgb_denominator
        improvement = (
            torch.relu(
                edited_abs_error.mean(dim=2, keepdim=True)
                - base_abs_error.mean(dim=2, keepdim=True)
            )
            * weight
        ).sum() / pixel_denominator

        coverage_deficit = (
            weight
            * torch.relu(reference_alpha.detach() - base_alpha.detach())
        ).detach()
        deficit_denominator = coverage_deficit.sum().clamp_min(1e-6)
        alpha_under = (
            coverage_deficit
            * torch.relu(reference_alpha.detach() - edited_alpha)
        ).sum() / deficit_denominator
        alpha_tolerance = float(
            self.encoder.cfg.lidar_neighbor_depth_alpha_tolerance
        )
        base_alpha_over = torch.relu(
            base_alpha.detach() - reference_alpha.detach() - alpha_tolerance
        )
        edited_alpha_over = torch.relu(
            edited_alpha - reference_alpha.detach() - alpha_tolerance
        )
        alpha_over_worse = (
            weight
            * torch.relu(
                edited_alpha_over - base_alpha_over
            )
        ).sum() / pixel_denominator
        initial_missing = torch.relu(
            reference_alpha.detach() - base_alpha.detach()
        )
        remaining_missing = torch.relu(
            reference_alpha.detach() - edited_alpha.detach()
        )
        alpha_deficit_before = (
            weight * initial_missing
        ).sum() / pixel_denominator
        alpha_deficit_after = (
            weight * remaining_missing
        ).sum() / pixel_denominator
        alpha_deficit_improvement = (
            alpha_deficit_before - alpha_deficit_after
        )
        alpha_deficit_relative_improvement = (
            alpha_deficit_improvement
            / alpha_deficit_before.clamp_min(1e-6)
        )
        deficit_pixel_weight = (
            weight * (initial_missing > 1e-6).to(weight.dtype)
        )
        deficit_pixel_denominator = deficit_pixel_weight.sum().clamp_min(1.0)
        alpha_improved_pixel_ratio = (
            deficit_pixel_weight
            * (remaining_missing < initial_missing - 1e-6).to(weight.dtype)
        ).sum() / deficit_pixel_denominator
        alpha_worsened_pixel_ratio = (
            deficit_pixel_weight
            * (remaining_missing > initial_missing + 1e-6).to(weight.dtype)
        ).sum() / deficit_pixel_denominator
        alpha_mean_change = (
            weight * (edited_alpha.detach() - base_alpha.detach())
        ).sum() / pixel_denominator
        recovered_ratio = (
            coverage_deficit
            * (initial_missing - remaining_missing)
        ).sum() / (
            coverage_deficit * initial_missing
        ).sum().clamp_min(1e-6)
        residual = getattr(
            edited_gaussians,
            "lidar_neighbor_depth_residual",
            None,
        )
        base_disparity = getattr(
            edited_gaussians,
            "lidar_neighbor_depth_base_disparity",
            None,
        )
        if residual is None or base_disparity is None:
            raise RuntimeError(
                "LiDAR neighbor depth residual or base disparity is missing."
            )
        residual_flat = residual.reshape_as(repair_mask)
        relative_residual = residual_flat / base_disparity.clamp_min(1e-6)
        target_disparity = getattr(
            edited_gaussians,
            "lidar_neighbor_depth_target_disparity",
            None,
        )
        attraction_confidence = getattr(
            edited_gaussians,
            "lidar_neighbor_depth_attraction_confidence",
            None,
        )
        if target_disparity is None or attraction_confidence is None:
            raise RuntimeError(
                "LiDAR neighbor attraction target or confidence is missing."
            )
        initial_target_distance = (target_disparity - base_disparity).abs()
        edited_target_distance = (
            target_disparity - (base_disparity + residual_flat)
        ).abs()
        attraction_ratio = 1.0 - (
            edited_target_distance
            / initial_target_distance.clamp_min(1e-6)
        )
        attraction_weight = (
            attraction_confidence
            * repair_mask.to(attraction_confidence.dtype)
            * (initial_target_distance > 1e-6).to(attraction_confidence.dtype)
        ).detach()
        attraction = (
            attraction_weight
            * torch.relu(
                float(self.encoder.cfg.lidar_neighbor_depth_min_attraction)
                - attraction_ratio
            )
        ).sum() / attraction_weight.sum().clamp_min(1.0)
        losses = {
            "lidar_neighbor_depth_local_rgb": local_rgb,
            "lidar_neighbor_depth_improvement": improvement,
            "lidar_neighbor_depth_alpha_under": alpha_under,
            "lidar_neighbor_depth_alpha_over_worse": alpha_over_worse,
            "lidar_neighbor_depth_attraction": attraction,
        }
        metrics = {
            "lidar_neighbor_depth_selected_count": repair_mask.sum().detach(),
            "lidar_neighbor_depth_alpha_deficit_recovered": recovered_ratio.detach(),
            "lidar_neighbor_depth_alpha_deficit_before": (
                alpha_deficit_before.detach()
            ),
            "lidar_neighbor_depth_alpha_deficit_after": (
                alpha_deficit_after.detach()
            ),
            "lidar_neighbor_depth_alpha_deficit_improvement": (
                alpha_deficit_improvement.detach()
            ),
            "lidar_neighbor_depth_alpha_deficit_relative_improvement": (
                alpha_deficit_relative_improvement.detach()
            ),
            "lidar_neighbor_depth_alpha_improved_pixel_ratio": (
                alpha_improved_pixel_ratio.detach()
            ),
            "lidar_neighbor_depth_alpha_worsened_pixel_ratio": (
                alpha_worsened_pixel_ratio.detach()
            ),
            "lidar_neighbor_depth_alpha_mean_change": alpha_mean_change.detach(),
            "lidar_neighbor_depth_attraction_ratio": (
                attraction_weight * attraction_ratio
            ).sum().div(attraction_weight.sum().clamp_min(1.0)).detach(),
            "lidar_neighbor_depth_attraction_confidence": (
                attraction_weight.sum()
                / repair_mask.sum().clamp_min(1.0)
            ).detach(),
            "lidar_neighbor_depth_mean_abs_relative_delta": (
                relative_residual.abs().sum()
                / repair_mask.sum().clamp_min(1.0)
            ).detach(),
            "lidar_neighbor_depth_positive_delta_ratio": (
                ((relative_residual > 0) & repair_mask.bool()).sum()
                / repair_mask.sum().clamp_min(1.0)
            ).detach(),
            "lidar_neighbor_depth_negative_delta_ratio": (
                ((relative_residual < 0) & repair_mask.bool()).sum()
                / repair_mask.sum().clamp_min(1.0)
            ).detach(),
        }
        return losses, metrics

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        

        # Run the model.
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            False,
            scene_names=batch["scene"],
        )

        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,
        )
        target_gt = batch["target"]["image"]

        # Compute metrics.
        use_adapter_loss = bool(
            getattr(
                self.encoder.cfg,
                "use_lidar_gaussian_adapter_loss",
                False,
            )
        )
        use_neighbor_depth_loss = bool(
            getattr(
                self.encoder.cfg,
                "use_lidar_gaussian_neighbor_depth_loss",
                False,
            )
        )
        if not use_adapter_loss and not use_neighbor_depth_loss:
            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(output.color, "b v c h w -> (b v) c h w"),
            )
            self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0
        if not use_adapter_loss and not use_neighbor_depth_loss:
            for loss_fn in self.losses:
                loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                self.log(f"loss/{loss_fn.name}", loss)
                for name, value in getattr(loss_fn, "diagnostics", {}).items():
                    self.log(f"loss/{name}", value)
                total_loss = total_loss + loss
        if use_adapter_loss:
            adapter_losses = self._lidar_gaussian_adapter_losses(
                output,
                gaussians,
                batch,
                (h, w),
            )
            adapter_loss_weights = {
                "lidar_gaussian_local_rgb": float(
                    self.encoder.cfg.lidar_gaussian_adapter_local_rgb_weight
                ),
                "lidar_gaussian_improvement": float(
                    self.encoder.cfg.lidar_gaussian_adapter_improvement_weight
                ),
                "lidar_gaussian_alpha_worse": float(
                    self.encoder.cfg.lidar_gaussian_adapter_alpha_weight
                ),
            }
            for name, loss in adapter_losses.items():
                total_loss = total_loss + adapter_loss_weights[name] * loss
                self.log(f"loss/{name}", loss)
        if use_neighbor_depth_loss:
            neighbor_losses, neighbor_metrics = self._lidar_neighbor_depth_losses(
                output,
                gaussians,
                batch,
                (h, w),
            )
            coverage_loss = (
                neighbor_losses["lidar_neighbor_depth_alpha_under"]
                + float(
                    self.encoder.cfg.lidar_neighbor_depth_over_alpha_worse_weight
                )
                * neighbor_losses["lidar_neighbor_depth_alpha_over_worse"]
            )
            neighbor_total = (
                float(self.encoder.cfg.lidar_neighbor_depth_local_rgb_weight)
                * neighbor_losses["lidar_neighbor_depth_local_rgb"]
                + float(self.encoder.cfg.lidar_neighbor_depth_improvement_weight)
                * neighbor_losses["lidar_neighbor_depth_improvement"]
                + float(self.encoder.cfg.lidar_neighbor_depth_coverage_weight)
                * coverage_loss
                + float(self.encoder.cfg.lidar_neighbor_depth_attraction_weight)
                * neighbor_losses["lidar_neighbor_depth_attraction"]
            )
            total_loss = total_loss + neighbor_total
            for name, loss in neighbor_losses.items():
                self.log(f"loss/{name}", loss)
            self.log("loss/lidar_neighbor_depth_total", neighbor_total)
            for name, value in neighbor_metrics.items():
                self.log(f"metric/{name}", value)
        context_render_weight = float(
            getattr(
                self.encoder.cfg,
                "lidar_gaussian_context_render_weight",
                0.0,
            )
        )
        context_lidar_mask = batch["context"].get("lidar_mask")
        if context_render_weight > 0.0 and context_lidar_mask is not None:
            context_height, context_width = batch["context"]["image"].shape[-2:]
            context_output = self.decoder.forward(
                gaussians,
                batch["context"]["extrinsics"],
                batch["context"]["intrinsics"],
                batch["context"]["near"],
                batch["context"]["far"],
                (context_height, context_width),
                depth_mode=None,
            )
            static_lidar_mask = (context_lidar_mask > 0.5).to(
                device=context_output.color.device,
                dtype=context_output.color.dtype,
            )
            context_dynamic_mask = batch["context"].get("dynamic_mask")
            if context_dynamic_mask is not None:
                static_lidar_mask = static_lidar_mask * (
                    context_dynamic_mask < 0.5
                ).to(
                    device=context_output.color.device,
                    dtype=context_output.color.dtype,
                )
            context_render_loss = (
                (context_output.color - batch["context"]["image"])
                .abs()
                .mul(static_lidar_mask)
                .sum()
                / (
                    static_lidar_mask.sum()
                    * context_output.color.shape[2]
                ).clamp_min(1.0)
            )
            weighted_context_render_loss = (
                context_render_weight * context_render_loss
            )
            total_loss = total_loss + weighted_context_render_loss
            self.log(
                "loss/lidar_gaussian_context_render",
                context_render_loss,
            )
        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    def on_after_backward(self) -> None:
        """Report whether optional LiDAR branches receive gradients."""
        if self.global_rank != 0:
            return
        if (
            self.global_step
            % self.train_cfg.print_log_every_n_steps
            != 0
        ):
            return

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
            )
        with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            )

        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]

        lidar_alpha_diagnostics = None
        if self.test_cfg.save_lidar_alpha_diagnostics:
            depth_predictor = getattr(self.encoder, "depth_predictor", None)
            if depth_predictor is None:
                raise RuntimeError("LiDAR alpha diagnostics require a depth predictor.")
            original_use_lidar_bias = depth_predictor.use_lidar_bias
            original_use_gaussian_adapter = (
                depth_predictor.use_lidar_gaussian_adapter
            )
            try:
                # Isolate the analytic depth bias: disable Gaussian parameter
                # edits in both branches and vary only use_lidar_bias.
                depth_predictor.use_lidar_gaussian_adapter = False
                depth_predictor.use_lidar_bias = False
                visual_gaussians = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=True,
                )
                depth_predictor.use_lidar_bias = True
                biased_gaussians = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=True,
                )
            finally:
                depth_predictor.use_lidar_bias = original_use_lidar_bias
                depth_predictor.use_lidar_gaussian_adapter = (
                    original_use_gaussian_adapter
                )

            visual_rgb = self.decoder.forward(
                visual_gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            ).color
            biased_rgb = self.decoder.forward(
                biased_gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            ).color
            visual_alpha = self.decoder.render_alpha(
                visual_gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
            )
            biased_alpha = self.decoder.render_alpha(
                biased_gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
            )
            visual_depth = self.decoder.render_depth(
                visual_gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                mode="relative_disparity",
            )
            biased_depth = self.decoder.render_depth(
                biased_gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                mode="relative_disparity",
            )
            lidar_alpha_diagnostics = {
                "visual_rgb": visual_rgb[0],
                "biased_rgb": biased_rgb[0],
                "visual_alpha": visual_alpha[0],
                "biased_alpha": biased_alpha[0],
                "visual_depth": visual_depth[0],
                "biased_depth": biased_depth[0],
            }

        # Save images.
        if self.test_cfg.save_image:
            expected_image_shape = (176, 320)
            assert rgb_gt.shape[-2:] == expected_image_shape, (
                "Processed target images must be 320x176 (width x height), "
                f"but got {tuple(rgb_gt.shape[-2:][::-1])}."
            )
            assert images_prob.shape[-2:] == expected_image_shape, (
                "Predictions must be 320x176 (width x height), "
                f"but got {tuple(images_prob.shape[-2:][::-1])}."
            )

            for index, target, prediction in zip(
                batch["target"]["index"][0], rgb_gt, images_prob
            ):
                filename = f"{index.item():0>6}.png"
                save_image(target, path / scene / "target_processed" / filename)
                save_image(prediction, path / scene / "prediction" / filename)
                if lidar_alpha_diagnostics is not None:
                    target_position = (
                        batch["target"]["index"][0] == index
                    ).nonzero(as_tuple=False)[0, 0]
                    diag_dir = path / scene / "lidar_alpha_diagnostics"
                    visual_rgb = lidar_alpha_diagnostics["visual_rgb"][target_position]
                    biased_rgb = lidar_alpha_diagnostics["biased_rgb"][target_position]
                    visual_alpha = lidar_alpha_diagnostics["visual_alpha"][target_position]
                    biased_alpha = lidar_alpha_diagnostics["biased_alpha"][target_position]
                    visual_depth = lidar_alpha_diagnostics["visual_depth"][target_position]
                    biased_depth = lidar_alpha_diagnostics["biased_depth"][target_position]
                    save_image(visual_rgb, diag_dir / f"{index.item():0>6}_rgb_visual.png")
                    save_image(biased_rgb, diag_dir / f"{index.item():0>6}_rgb_bias.png")
                    save_image(visual_alpha, diag_dir / f"{index.item():0>6}_alpha_visual.png")
                    save_image(biased_alpha, diag_dir / f"{index.item():0>6}_alpha_bias.png")
                    save_image(visual_depth, diag_dir / f"{index.item():0>6}_depth_visual.png")
                    save_image(biased_depth, diag_dir / f"{index.item():0>6}_depth_bias.png")
                    save_image(
                        (biased_alpha - visual_alpha).abs(),
                        diag_dir / f"{index.item():0>6}_alpha_abs_diff.png",
                    )
                    save_image(
                        (biased_depth - visual_depth).abs(),
                        diag_dir / f"{index.item():0>6}_depth_abs_diff.png",
                    )

                    darkened = (
                        biased_rgb.mean(dim=0)
                        < visual_rgb.mean(dim=0) - 0.05
                    )
                    if darkened.any():
                        print(
                            "[LiDAR Alpha Diagnostics] "
                            f"scene={scene}, target={index.item()}, "
                            f"darkened_pixels={int(darkened.sum().item())}, "
                            f"alpha_visual={visual_alpha[darkened].mean().item():.4f}, "
                            f"alpha_bias={biased_alpha[darkened].mean().item():.4f}, "
                            f"alpha_bias_gt_0.9="
                            f"{(biased_alpha[darkened] > 0.9).float().mean().item():.4f}, "
                            f"alpha_bias_lt_0.1="
                            f"{(biased_alpha[darkened] < 0.1).float().mean().item():.4f}, "
                            f"depth_abs_diff="
                            f"{(biased_depth[darkened] - visual_depth[darkened]).abs().mean().item():.4f}"
                        )

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = images_prob

            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()
    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        return
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )
        output_softmax = self.decoder.forward(
            gaussians_softmax,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_softmax = output_softmax.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("val",), (rgb_softmax,)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        projections = hcat(*render_projections(
                                gaussians_softmax,
                                256,
                                extra_label="(Softmax)",
                            )[0])
        self.logger.log_image(
            "projection",
            [prep_image(add_border(projections))],
            step=self.global_step,
        )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=self.global_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], self.global_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
