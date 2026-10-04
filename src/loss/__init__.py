from .loss import Loss
from .loss_context_rgb import LossContextRgb, LossContextRgbCfgWrapper
from .loss_cross_visual_depth import LossCrossVisualDepth, LossCrossVisualDepthCfgWrapper
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossContextRgbCfgWrapper: LossContextRgb,
    LossCrossVisualDepthCfgWrapper: LossCrossVisualDepth,
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossCrossVisualDepthCfgWrapper | LossContextRgbCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
