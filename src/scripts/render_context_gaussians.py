"""Render each context's Gaussian subset and all Gaussians at target cameras.

Run from the repository root:
    python -m src.scripts.render_context_gaussians render.max_batches=1
"""

from dataclasses import fields
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.global_cfg import set_cfg
from src.misc.image_io import save_image
from src.misc.wandb_tools import update_checkpoint_path
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper
from src.model.types import Gaussians


def select_gaussians(gaussians: Gaussians, start: int, stop: int) -> Gaussians:
    """The costvolume encoder flattens (view, pixel, surface, sample), in order."""
    return Gaussians(**{
        field.name: getattr(gaussians, field.name)[:, start:stop]
        for field in fields(Gaussians)
    })


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


@hydra.main(version_base=None, config_path="../../config", config_name="render_context_gaussians")
def main(cfg_dict: DictConfig) -> None:
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    if cfg.mode != "test":
        raise ValueError("This rendering script requires mode=test.")
    if cfg.model.encoder.name != "costvolume":
        raise ValueError("Gaussian view splitting requires the costvolume encoder.")
    if not torch.cuda.is_available():
        raise RuntimeError("Rendering requires CUDA and the CUDA Gaussian rasterizer.")
    max_batches = cfg_dict.render.max_batches
    if max_batches is not None and max_batches <= 0:
        raise ValueError("render.max_batches must be positive or null (all batches).")
    checkpoint = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    if checkpoint is None:
        raise ValueError("Set checkpointing.load to trained model weights.")

    seed_everything(cfg.seed, workers=True)
    device = torch.device("cuda")
    encoder, visualizer = get_encoder(cfg.model.encoder)
    model = ModelWrapper.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
        strict=True,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=visualizer,
        decoder=get_decoder(cfg.model.decoder, cfg.dataset),
        losses=[],
        step_tracker=None,
    ).to(device).eval()
    output_dir = Path(cfg_dict.render.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg_dict, output_dir / "config.yaml")
    loader = DataModule(cfg.dataset, cfg.data_loader).test_dataloader()

    # Match test_step's opacity schedule (global_step=0) and inference settings.
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = model.data_shim(to_device(batch, device))
            context = batch["context"]
            target = batch["target"]
            num_views = context["image"].shape[1]
            h, w = target["image"].shape[-2:]
            gaussians = model.encoder(context, 0, deterministic=False)
            per_view = (
                context["image"].shape[-2] * context["image"].shape[-1]
                * cfg.model.encoder.num_surfaces * cfg.model.encoder.gaussians_per_pixel
            )
            if gaussians.means.shape[1] != num_views * per_view:
                raise RuntimeError("Unexpected Gaussian layout; cannot split by context view.")

            def render_and_save(selected: Gaussians, filename: str) -> None:
                colors = model.decoder(
                    selected, target["extrinsics"], target["intrinsics"],
                    target["near"], target["far"], (h, w), depth_mode=None,
                ).color
                for sample, scene in enumerate(batch["scene"]):
                    for view, index in enumerate(target["index"][sample]):
                        path = output_dir / scene / f"batch_{batch_index:06d}" / f"target_{int(index):06d}"
                        save_image(colors[sample, view], path / filename)
                        if filename == "full.png":
                            save_image(target["image"][sample, view], path / "ground_truth.png")

            render_and_save(gaussians, "full.png")
            for view in range(num_views):
                render_and_save(
                    select_gaussians(gaussians, view * per_view, (view + 1) * per_view),
                    f"context_{view}.png",
                )
            print(f"Rendered batch {batch_index} to {output_dir}")
            if max_batches is not None and batch_index + 1 >= max_batches:
                break


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
