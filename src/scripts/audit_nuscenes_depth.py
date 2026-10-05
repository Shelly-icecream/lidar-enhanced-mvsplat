"""Compare predicted camera Z-depth with sparse LiDAR (CPU, no rendering).

python -m src.scripts.audit_nuscenes_depth --checkpoint checkpoints/stage1.ckpt
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from omegaconf import OmegaConf
from dacite import from_dict
from src.model.encoder import get_encoder, EncoderCfg
from src.global_cfg import set_cfg
from src.dataset.dataset_nuscenes import DatasetNuScenes
from src.dataset.shims.crop_shim import rescale_and_crop
from nuscenes.nuscenes import NuScenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/stage1.ckpt")
    parser.add_argument("--item-index", type=int, default=0)
    parser.add_argument("--lidar-bias", action="store_true", help="Apply hard LiDAR replacement; default audits raw network depth.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[2] / "config"),
        version_base=None,
    ):
        cfg = compose(config_name="render_context_gaussians")
    cfg.checkpointing.load = args.checkpoint
    cfg.model.encoder.use_lidar_bias_train = args.lidar_bias
    cfg.model.encoder.use_lidar_bias_test = args.lidar_bias
    set_cfg(cfg)
    enc, _ = get_encoder(
        from_dict(EncoderCfg, OmegaConf.to_container(cfg.model.encoder))
    )
    weights = torch.load(
        cfg.checkpointing.load, map_location="cpu", weights_only=False
    )["state_dict"]
    enc.load_state_dict(
        {
            k[len("encoder.") :]: v
            for k, v in weights.items()
            if k.startswith("encoder.")
        },
        strict=True,
    )
    enc.eval()
    head_stats = {}
    captured = {}

    def save_stats(name):
        def hook(module, inputs, output):
            disparity = output[:, : cfg.model.encoder.gaussians_per_pixel]
            captured["residual"] = disparity.detach()
            head_stats[name] = {
                "min": float(disparity.min()),
                "median": float(disparity.median()),
                "max": float(disparity.max()),
            }

        return hook

    enc.depth_predictor.to_disparity.register_forward_hook(
        save_stats("disparity_residual")
    )

    def save_coarse(module, inputs):
        disparity = inputs[0][:, -2:-1]
        captured["coarse"] = disparity.detach()
        head_stats["coarse_disparity"] = {
            "min": float(disparity.min()),
            "median": float(disparity.median()),
            "max": float(disparity.max()),
        }

    enc.depth_predictor.refine_unet.register_forward_pre_hook(save_coarse)
    ds = DatasetNuScenes.__new__(DatasetNuScenes)
    ds.cfg = SimpleNamespace(
        root=Path("datasets/nuscenes"),
        version="v1.0-mini",
        camera_name="CAM_FRONT",
        num_context_views=2,
        num_target_views=1,
        frame_stride=1,
        use_keyframe_only=True,
    )
    ds.stage = "test"
    ds.nusc = NuScenes(version="v1.0-mini", dataroot="datasets/nuscenes", verbose=False)
    import torchvision.transforms as tf

    ds.to_tensor = tf.ToTensor()
    item = ds._build_index()[args.item_index]
    images = []
    poses = []
    ks = []
    camera_tokens = []
    for tok in item["context_sample_tokens"]:
        cam = ds.nusc.get("sample", tok)["data"]["CAM_FRONT"]
        camera_tokens.append(cam)
        im, pose, k = ds._load_camera(cam)
        im, k = rescale_and_crop(im, k, tuple(cfg.dataset.image_shape))
        images.append(im)
        poses.append(pose)
        ks.append(k)
    context = {
        "image": torch.stack(images)[None],
        "extrinsics": torch.stack(poses)[None],
        "intrinsics": torch.stack(ks)[None],
        "near": torch.full((1, 2), float(cfg.dataset.near)),
        "far": torch.full((1, 2), float(cfg.dataset.far)),
    }
    if args.lidar_bias:
        context["lidar_depth"] = torch.stack([
            ds._load_lidar_depth(tok, cam, k, tuple(cfg.dataset.image_shape))
            for tok, cam, k in zip(item["context_sample_tokens"], camera_tokens, ks)
        ])[None]
    torch.manual_seed(cfg.seed)
    with torch.inference_mode():
        gaussians = enc(context, 0, deterministic=False)
    depth = gaussians.context_depth[0].numpy()
    raw_disparity = (captured["coarse"] + captured["residual"]).numpy()
    report = {
        "checkpoint": str(cfg.checkpointing.load),
        "lidar_bias": args.lidar_bias,
        "frame": item,
        "head_stats": head_stats,
        "views": [],
    }
    h, w = depth.shape[-2:]
    for i, (tok, cam) in enumerate(zip(item["context_sample_tokens"], camera_tokens)):
        lidar = ds.nusc.get("sample", tok)["data"]["LIDAR_TOP"]
        points, z, _ = ds.nusc.explorer.map_pointcloud_to_image(lidar, cam)
        # Use the exact crop transform used by the dataset.
        cs = ds.nusc.get(
            "calibrated_sensor",
            ds.nusc.get("sample_data", cam)["calibrated_sensor_token"],
        )
        k0 = np.array(cs["camera_intrinsic"])
        rays = np.linalg.inv(k0) @ np.vstack((points[:2], np.ones(points.shape[1])))
        xy = ks[i].numpy() @ rays
        x = (xy[0] * w).astype(int)
        y = (xy[1] * h).astype(int)
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h) & (z > 3)
        pred = depth[i, y[valid], x[valid]]
        true = z[valid]
        # Ground depth and rendered Gaussian camera Z must agree.
        means = gaussians.means[0, i * h * w : (i + 1) * h * w]
        homogeneous = torch.cat((means, torch.ones_like(means[:, :1])), dim=-1)
        camera_points = (torch.linalg.inv(poses[i]) @ homogeneous.T).T
        report["views"].append(
            {
                "context": i,
                "lidar_points": int(valid.sum()),
                "depth_median_m": float(np.median(depth[i])),
                "depth_min_m": float(depth[i].min()),
                "depth_max_m": float(depth[i].max()),
                "raw_inverse_depth_min": float(raw_disparity[i].min()),
                "raw_inverse_depth_max": float(raw_disparity[i].max()),
                "fraction_clamped_to_far": float(
                    np.mean(raw_disparity[i] < 1.0 / float(cfg.dataset.far))
                ),
                "fraction_clamped_to_near": float(
                    np.mean(raw_disparity[i] > 1.0 / float(cfg.dataset.near))
                ),
                "lidar_depth_median_m": float(np.median(true)),
                "predicted_to_lidar_depth_median_ratio": float(np.median(pred / true)),
                "fraction_at_far_bound": float(
                    np.mean(depth[i] >= float(cfg.dataset.far) - 0.01)
                ),
                "gaussian_z_vs_depth_max_error_m": float(
                    (camera_points[:, 2] - torch.from_numpy(depth[i]).flatten())
                    .abs()
                    .max()
                ),
            }
        )
    out = Path("outputs/geometry_audit")
    out.mkdir(exist_ok=True)
    (
        out / f"{Path(args.checkpoint).stem}_item{args.item_index}{'_bias' if args.lidar_bias else ''}_depth_lidar.json"
    ).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
