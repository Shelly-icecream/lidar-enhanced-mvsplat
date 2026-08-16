"""Overfit the LiDAR SH-DC adapter on one fixed test batch.

This is a diagnostic, not a training entry point. It answers whether the current
Gaussian parameterization can remove a color cast when cross-batch conflicts are
eliminated.
"""

import csv
from pathlib import Path

import hydra
import torch
from einops import rearrange
from omegaconf import DictConfig
from torchvision.utils import save_image

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule, get_data_shim
from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.global_cfg import set_cfg
from src.misc.step_tracker import StepTracker
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def region_channel_metrics(prediction, target, mask):
    mask = mask.expand_as(prediction)
    count = mask[:, :, :1].sum().clamp_min(1.0)
    delta = prediction - target
    mae = (delta.abs() * mask).sum(dim=(0, 1, 3, 4)) / count
    bias = (delta * mask).sum(dim=(0, 1, 3, 4)) / count
    return {
        **{f"mae_{name}": value.item() for name, value in zip("rgb", mae)},
        **{f"bias_{name}": value.item() for name, value in zip("rgb", bias)},
    }


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    device = torch.device("cuda")
    output_dir = Path(cfg_dict.diagnostic.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    checkpoint = torch.load(
        cfg_dict.diagnostic.checkpoint,
        map_location="cpu",
        weights_only=False,
    )["state_dict"]
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in checkpoint.items()
        if key.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.to(device).eval()
    decoder.to(device)

    predictor = encoder.depth_predictor
    predictor.use_lidar_bias = True
    predictor.lidar_gaussian_edit_opacity = False
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in predictor.lidar_gaussian_adapter.parameters():
        parameter.requires_grad_(True)
    predictor.lidar_gaussian_adapter.train()

    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker())
    batch = next(iter(data_module.test_dataloader()))
    batch = get_data_shim(encoder)(to_device(batch, device))
    target = batch["target"]["image"]
    _, _, _, height, width = target.shape

    with torch.no_grad():
        predictor.use_lidar_gaussian_adapter = False
        base_gaussians = encoder(batch["context"], 0, deterministic=True)
        base_color = decoder.forward(
            base_gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (height, width),
            depth_mode=None,
        ).color
        predictor.use_lidar_gaussian_adapter = True
        initial_gaussians = encoder(batch["context"], 0, deterministic=True)
        initial_color = decoder.forward(
            initial_gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (height, width),
            depth_mode=None,
        ).color

    static = torch.ones_like(target[:, :, :1])
    if "dynamic_mask" in batch["target"]:
        static = static * (batch["target"]["dynamic_mask"] < 0.5)
    affected = (
        (initial_color - base_color).abs().mean(dim=2, keepdim=True) > 0.01
    ) * static
    yellow = (
        (
            0.5 * (initial_color[:, :, 0:1] + initial_color[:, :, 1:2])
            - initial_color[:, :, 2:3]
        )
        > 0.05
    ) * affected
    if not yellow.any():
        yellow = affected

    weight = static.float()

    optimizer = torch.optim.Adam(
        predictor.lidar_gaussian_adapter.parameters(),
        lr=float(cfg_dict.diagnostic.lr),
    )
    rows = []
    steps = int(cfg_dict.diagnostic.steps)
    log_every = int(cfg_dict.diagnostic.log_every)
    for step in range(steps + 1):
        gaussians = encoder(batch["context"], step, deterministic=True)
        color = decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (height, width),
            depth_mode=None,
        ).color
        loss = ((color - target).square() * weight).sum() / (
            weight.sum() * 3
        ).clamp_min(1.0)

        if step % log_every == 0 or step == steps:
            flat_target = rearrange(target, "b v c h w -> (b v) c h w")
            flat_color = rearrange(color, "b v c h w -> (b v) c h w")
            row = {
                "step": step,
                "loss": loss.item(),
                "psnr": compute_psnr(flat_target, flat_color).mean().item(),
                "ssim": compute_ssim(flat_target, flat_color).mean().item(),
                "lpips": compute_lpips(flat_target, flat_color).mean().item(),
                "affected_fraction": affected.float().mean().item(),
                "yellow_fraction": yellow.float().mean().item(),
            }
            for prefix, mask in (("affected", affected), ("yellow", yellow)):
                row.update(
                    {
                        f"{prefix}_{key}": value
                        for key, value in region_channel_metrics(
                            color, target, mask.float()
                        ).items()
                    }
                )
            rows.append(row)
            print(row)
            save_image(color[0, 0].detach(), output_dir / f"step_{step:04d}.png")

        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with (output_dir / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save_image(target[0, 0], output_dir / "target.png")
    save_image(base_color[0, 0], output_dir / "base_no_dc_edit.png")


if __name__ == "__main__":
    main()
