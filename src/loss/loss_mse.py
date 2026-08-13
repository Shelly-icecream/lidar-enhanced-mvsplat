from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        delta = prediction.color - batch["target"]["image"]
        dynamic_mask = batch["target"].get("dynamic_mask")
        if dynamic_mask is None:
            return self.cfg.weight * (delta**2).mean()

        valid_mask = (1.0 - dynamic_mask).to(
            device=delta.device,
            dtype=delta.dtype,
        )
        squared_error = delta.square() * valid_mask
        num_channels = delta.shape[2]
        denominator = (valid_mask.sum() * num_channels).clamp_min(1.0)
        return self.cfg.weight * squared_error.sum() / denominator
