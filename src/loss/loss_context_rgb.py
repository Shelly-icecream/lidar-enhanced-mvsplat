from dataclasses import dataclass

from .loss_mse import LossMse


@dataclass
class LossContextRgbCfg:
    weight: float = 1.0


@dataclass
class LossContextRgbCfgWrapper:
    context_rgb: LossContextRgbCfg


class LossContextRgb(LossMse):
    """RGB reconstruction of contexts using their own dynamic masks."""

    def forward(self, prediction, batch, gaussians, global_step):
        context = batch["context"]
        delta = prediction.color - context["image"]
        dynamic_mask = context.get("dynamic_mask")
        if dynamic_mask is None:
            return self.cfg.weight * delta.square().mean()

        static_mask = (dynamic_mask < 0.5).to(device=delta.device)
        squared_error = delta.square().masked_fill(~static_mask, 0)
        valid_elements = static_mask.sum() * delta.shape[2]
        return self.cfg.weight * squared_error.sum() / valid_elements.clamp_min(1)
