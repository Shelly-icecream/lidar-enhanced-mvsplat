from .loss import Loss
from .loss_context_rgb import LossContextRgb, LossContextRgbCfgWrapper
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_lidar_inverse_depth import LossLidarInverseDepth, LossLidarInverseDepthCfgWrapper

LOSSES = {
    LossLidarInverseDepthCfgWrapper: LossLidarInverseDepth,
    LossContextRgbCfgWrapper: LossContextRgb,
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossContextRgbCfgWrapper | LossLidarInverseDepthCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
