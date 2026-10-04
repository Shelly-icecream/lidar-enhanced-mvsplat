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

        # The dataset supplies the union of target-time and projected
        # context-time dynamic regions, with one mask channel per view.
        static_mask = (dynamic_mask < 0.5).to(device=delta.device)
        squared_error = delta.square().masked_fill(~static_mask, 0)
        valid_elements = static_mask.sum() * delta.shape[2]
        return self.cfg.weight * squared_error.sum() / valid_elements.clamp_min(1)
