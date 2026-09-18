"""Read-only nuScenes geometry / pretrained raw cost-volume diagnostics.

Run: MPLCONFIGDIR=/tmp/mvsplat-mpl python -m src.scripts.diagnose_foe
No LiDAR, training, or experiment configuration is used.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pyquaternion import Quaternion

from src.dataset.shims.crop_shim import rescale_and_crop
from src.model.encoder.backbone.backbone_multiview import BackboneMultiview
from src.model.encoder.costvolume.depth_predictor_multiview import warp_with_pose_depth_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/nuscenes")
    parser.add_argument("--checkpoint", default="checkpoints/re10k.ckpt")
    parser.add_argument("--output", default="outputs/foe_diagnostics")
    parser.add_argument("--scenes", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(0)
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name in ["scene", "sample", "sample_data", "calibrated_sensor", "ego_pose", "sensor"]:
        rows = json.loads((root / "v1.0-mini" / f"{name}.json").read_text())
        tables[name] = {r["token"]: r for r in rows}
    cam_data = {}
    for sd in tables["sample_data"].values():
        cs = tables["calibrated_sensor"][sd["calibrated_sensor_token"]]
        if sd["is_key_frame"] and tables["sensor"][cs["sensor_token"]]["channel"] == "CAM_FRONT":
            cam_data[sd["sample_token"]] = sd

    def transform(record):
        mat = torch.eye(4, dtype=torch.float64)
        mat[:3, :3] = torch.tensor(Quaternion(record["rotation"]).rotation_matrix)
        mat[:3, 3] = torch.tensor(record["translation"], dtype=torch.float64)
        return mat

    def camera(sample):
        sd = cam_data[sample["token"]]
        cs = tables["calibrated_sensor"][sd["calibrated_sensor_token"]]
        im = torch.tensor(np.array(Image.open(root / sd["filename"]).convert("RGB")), dtype=torch.float32).permute(2, 0, 1) / 255
        k = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
        k[0] /= im.shape[2]
        k[1] /= im.shape[1]
        im, k = rescale_and_crop(im, k, (256, 256))
        pose = transform(tables["ego_pose"][sd["ego_pose_token"]]) @ transform(cs)
        return im, k, pose, sd

    backbone = BackboneMultiview(feature_channels=128, downscale_factor=4).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    prefix = "encoder.backbone."
    weights = {k[len(prefix):]: v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)}
    backbone.load_state_dict(weights, strict=True)
    del checkpoint, weights
    summaries = []
    scenes = sorted(tables["scene"].values(), key=lambda x: x["name"])[:args.scenes]
    for scene in scenes:
        chain, token = [], scene["first_sample_token"]
        while token:
            sample = tables["sample"][token]
            chain.append(sample)
            token = sample["next"]
        center = len(chain) // 2
        cams = [camera(chain[center - 1]), camera(chain[center + 1])]
        images = torch.stack([c[0] for c in cams])[None]
        with torch.inference_mode():
            features = backbone(images, attn_splits=2)
        if isinstance(features, (tuple, list)):
            features = features[0]
        _, _, channels, h, w = features.shape
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        pixels = torch.stack([xx, yy, torch.ones_like(xx)]).float().reshape(3, -1)
        for ref in range(2):
            src = 1 - ref
            k = cams[ref][1].clone()
            k[0] *= w
            k[1] *= h
            ks = cams[src][1].clone()
            ks[0] *= w
            ks[1] *= h
            pose = (torch.linalg.inv(cams[src][2]) @ cams[ref][2]).float()
            rays = torch.linalg.solve(k, pixels)

            def project(depths):
                points = (pose[:3, :3] @ rays)[None] * depths[:, None, None] + pose[:3, 3][None, :, None]
                proj = torch.einsum("ij,djn->din", ks, points)
                xy = proj[:, :2] / proj[:, 2:3].clamp_min(1e-3)
                valid = (points[:, 2] > 1e-3) & (xy[:, 0] >= 0) & (xy[:, 0] <= w-1) & (xy[:, 1] >= 0) & (xy[:, 1] <= h-1)
                return xy.permute(0, 2, 1), valid

            xy, valid = project(torch.tensor([10., 30., 50.]))
            maps = []
            for a, b in [(0, 2), (1, 2)]:
                dist = (xy[a] - xy[b]).norm(dim=-1)
                maps.append(torch.where(valid[a] & valid[b], dist, torch.nan).reshape(h, w).numpy())
            # Original MVSplat RE10K candidate range; inverse-depth uniform.
            depths = torch.linspace(1/100, 1, 128).reciprocal()
            coords, all_valid = project(depths)
            grid = coords.clone()
            grid[..., 0] = 2 * grid[..., 0] / (w-1) - 1
            grid[..., 1] = 2 * grid[..., 1] / (h-1) - 1
            correlations = []
            with torch.inference_mode():
                for chunk in grid.split(8):
                    warped = F.grid_sample(features[0, src:src+1], chunk.reshape(1, -1, w, 2), align_corners=True, mode="bilinear", padding_mode="zeros")
                    warped = warped.reshape(channels, len(chunk), h, w)
                    correlations.append((features[0, ref, :, None] * warped).sum(0) / channels**0.5)
            corr = torch.cat(correlations).numpy()
            # Verify the diagnostic reproduces the repository's actual warp/correlation.
            assert torch.allclose(k, ks), "Repository warp assumes shared intrinsics."
            with torch.inference_mode():
                indices = torch.tensor([0, 32, 64, 127])
                actual = warp_with_pose_depth_candidates(
                    features[0, src:src+1], k[None], pose[None],
                    depths[indices][None, :, None, None].expand(1, 4, h, w),
                )
                actual_corr = (features[0, ref:ref+1, :, None] * actual).sum(1) / channels**0.5
                np.testing.assert_allclose(corr[indices], actual_corr[0].numpy(), atol=2e-4, rtol=2e-4)
            valid_np = all_valid.reshape(128, h, w).numpy()
            corr_valid = np.where(valid_np, corr, np.nan)
            far = (depths.numpy() >= 20) & (depths.numpy() <= 80)
            with np.errstate(invalid="ignore"):
                far_std = np.std(corr[far], axis=0)
            far_std[~valid_np[far].all(0)] = np.nan
            tag = f"{scene['name']}_ref{ref}"
            np.savez_compressed(out / f"{tag}.npz", depths=depths.numpy(), raw_correlation=corr, valid=valid_np, sensitivity_10_50=maps[0], sensitivity_30_50=maps[1], far_score_std=far_std)
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            axes[0].imshow(cams[ref][0].permute(1, 2, 0)); axes[0].set_title("Reference RGB")
            for ax, data, title in zip(axes[1:], [*maps, far_std], ["10 vs 50 m (feature px)", "30 vs 50 m (feature px)", "Raw score std: 20-80 m"]):
                artist = ax.imshow(data, cmap="magma", vmin=0)
                fig.colorbar(artist, ax=ax, shrink=.7); ax.set_title(title)
            fig.suptitle(tag + " | gray/blank = invalid projection")
            fig.tight_layout(); fig.savefig(out / f"{tag}_maps.png", dpi=140); plt.close(fig)
            # Fixed spatial probes, not cherry-picked by score. Semantic/static labels require visual review.
            probes = [(w//2,h//2), (w//2,h//3), (w//4,h//2), (3*w//4,h//2), (w//2,3*h//4)]
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].imshow(cams[ref][0].permute(1,2,0))
            order = np.argsort(depths.numpy())
            for i, (x,y) in enumerate(probes):
                color = f"C{i}"
                axes[0].plot(x*4,y*4,"o",color=color); axes[0].text(x*4+3,y*4,str(i),color=color)
                axes[1].plot(depths.numpy()[order],corr_valid[order,y,x],label=f"P{i} ({x},{y})",color=color,marker=".")
            axes[1].set(xlabel="Hypothesized depth (m)", ylabel="Raw dot-product correlation", xlim=(1,100))
            axes[1].legend(); axes[1].grid(alpha=.3)
            fig.suptitle(tag + " | original 128 inverse-depth candidates; invalid values omitted")
            fig.tight_layout(); fig.savefig(out / f"{tag}_curves.png",dpi=140); plt.close(fig)
            central = (xx.numpy() >= w*.375)&(xx.numpy()<w*.625)&(yy.numpy()>=h*.375)&(yy.numpy()<h*.625)
            row = {"scene":scene["name"],"reference":ref,"sample_tokens":[c[3]["sample_token"] for c in cams],"time_gap_s":abs(cams[1][3]["timestamp"]-cams[0][3]["timestamp"])/1e6,"baseline_m":float(pose[:3,3].norm()),"translation_ref_in_source":pose[:3,3].tolist()}
            for region, mask in [("center",central),("outer",~central)]:
                values = maps[1][mask]; values=values[np.isfinite(values)]
                row[region] = {"valid_pixels":len(values),"median_30_50_feature_px":float(np.median(values)) if len(values) else None,"fraction_below_025px":float(np.mean(values<.25)) if len(values) else None,"median_far_score_std":float(np.nanmedian(far_std[mask]))}
            summaries.append(row)
            print(json.dumps(row),flush=True)
        (out / "summary.json").write_text(json.dumps(summaries,indent=2))
    (out / "settings.json").write_text(json.dumps(vars(args)|{"input_shape":[256,256],"feature_scale":4,"near":1,"far":100,"depth_candidates":128,"checkpoint_loading":"strict backbone only","device":"cpu","selection":"first N scenes sorted by name, middle triplet, both context directions","limitations":["No semantic or occlusion mask: in-frame projection does not guarantee visibility", "Fixed probes are not verified static/distant objects", "Raw correlation is not a calibrated depth probability", "No training or reconstruction causal test"]},indent=2))


if __name__ == "__main__":
    main()
