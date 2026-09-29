from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossGradientCfg:
    weight: float


@dataclass
class LossGradientCfgWrapper:
    gradient: LossGradientCfg


class LossGradient(Loss[LossGradientCfg, LossGradientCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        # Float32 differences preserve the gradient back to the prediction.
        pred = prediction.color.float()
        target = batch["target"]["image"].float()
        valid = torch.ones_like(target[:, :, :1], dtype=torch.bool)
        dynamic_mask = batch["target"].get("dynamic_mask")
        if dynamic_mask is not None:
            valid = dynamic_mask.to(device=pred.device) < 0.5

        # Both endpoints must be valid. Masking images before differencing
        # would introduce artificial edges at mask borders.
        error_x = (
            (pred[..., 1:] - pred[..., :-1])
            - (target[..., 1:] - target[..., :-1])
        ).abs()
        error_y = (
            (pred[..., 1:, :] - pred[..., :-1, :])
            - (target[..., 1:, :] - target[..., :-1, :])
        ).abs()
        valid_x = valid[..., 1:] & valid[..., :-1]
        valid_y = valid[..., 1:, :] & valid[..., :-1, :]

        def masked_mean(error: Tensor, mask: Tensor) -> Tensor:
            denominator = (mask.sum() * error.shape[2]).clamp_min(1)
            return torch.where(mask, error, 0.0).sum() / denominator

        return self.cfg.weight * (
            masked_mean(error_x, valid_x) + masked_mean(error_y, valid_y)
        )
