from .loss import Loss
from .loss_gradient import LossGradient, LossGradientCfgWrapper
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_lidar_depth import LossLidarDepth, LossLidarDepthCfgWrapper

LOSSES = {
    LossGradientCfgWrapper: LossGradient,
    LossLidarDepthCfgWrapper: LossLidarDepth,
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossLidarDepthCfgWrapper | LossGradientCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
