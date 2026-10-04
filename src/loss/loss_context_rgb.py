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
        # Reuse masked RGB MSE with context ground truth and context-time masks.
        context_batch = {**batch, "target": batch["context"]}
        return super().forward(prediction, context_batch, gaussians, global_step)
