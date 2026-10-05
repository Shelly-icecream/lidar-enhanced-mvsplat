"""CPU audit of actual temporal windows, camera poses and projection conventions.

Run: python -m src.scripts.audit_nuscenes_geometry
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

from src.dataset.dataset_nuscenes import DatasetNuScenes
from src.geometry.projection import project, unproject, homogenize_points


def main():
    dataset = DatasetNuScenes.__new__(DatasetNuScenes)
    dataset.cfg = SimpleNamespace(
        version="v1.0-mini",
        camera_name="CAM_FRONT",
        num_context_views=2,
        num_target_views=1,
        frame_stride=1,
        use_keyframe_only=True,
    )
    dataset.nusc = NuScenes(
        version="v1.0-mini", dataroot="datasets/nuscenes", verbose=False
    )
    result = {"splits": {}, "max_pose_error": 0.0, "max_projection_error_pixels": 0.0}
    for stage in ("train", "test"):
        dataset.stage = stage
        items = dataset._build_index()
        deltas, translations = [], []
        for item in items:
            left, right = item["context_sample_tokens"]
            target = item["target_sample_tokens"][0]
            records = [
                dataset.nusc.get("sample", token) for token in (left, target, right)
            ]
            assert records[1]["prev"] == left and records[1]["next"] == right
            cameras = [
                dataset.nusc.get("sample_data", s["data"]["CAM_FRONT"]) for s in records
            ]
            ts = [c["timestamp"] for c in cameras]
            assert ts[0] < ts[1] < ts[2]
            deltas.append([(ts[0] - ts[1]) / 1e6, (ts[2] - ts[1]) / 1e6])
            poses, intrinsics = [], []
            for camera in cameras:
                cs = dataset.nusc.get(
                    "calibrated_sensor", camera["calibrated_sensor_token"]
                )
                ep = dataset.nusc.get("ego_pose", camera["ego_pose_token"])
                pose = dataset._make_transform(
                    ep["translation"], ep["rotation"]
                ) @ dataset._make_transform(cs["translation"], cs["rotation"])
                # Independently compare to SDK's float64 rigid transforms.
                sdk_pose = transform_matrix(
                    ep["translation"], Quaternion(ep["rotation"])
                ) @ transform_matrix(cs["translation"], Quaternion(cs["rotation"]))
                result["max_pose_error"] = max(
                    result["max_pose_error"],
                    float(np.abs(pose.numpy() - sdk_pose).max()),
                )
                poses.append(pose)
                k = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
                k[0] /= camera["width"]
                k[1] /= camera["height"]
                intrinsics.append(k)
            translations.append(
                [
                    (torch.linalg.inv(poses[1]) @ poses[i])[:3, 3].tolist()
                    for i in (0, 2)
                ]
            )
            xy = torch.tensor([[0.25, 0.3], [0.5, 0.5], [0.75, 0.7]])
            points = unproject(xy, torch.tensor([10.0, 20.0, 40.0]), intrinsics[1])
            world = (poses[1] @ homogenize_points(points).T).T[:, :3]
            for i in (0, 2):
                projected, _ = project(world, poses[i], intrinsics[i])
                relative = torch.linalg.inv(poses[i]) @ poses[1]
                p = (relative @ homogenize_points(points).T).T[:, :3]
                expected = (intrinsics[i] @ p.T).T
                expected = expected[:, :2] / expected[:, 2:]
                error = (
                    ((projected - expected).abs() * torch.tensor([1600.0, 900.0]))
                    .max()
                    .item()
                )
                result["max_projection_error_pixels"] = max(
                    result["max_projection_error_pixels"], error
                )
        result["splits"][stage] = {
            "windows": len(items),
            "all_immediate_neighbors": True,
            "context_time_offsets_seconds_min": np.min(deltas, axis=0).tolist(),
            "context_time_offsets_seconds_max": np.max(deltas, axis=0).tolist(),
            "first_window": items[0],
            "first_context_positions_in_target_camera_meters": translations[0],
        }
    output = Path("outputs/geometry_audit")
    output.mkdir(parents=True, exist_ok=True)
    (output / "temporal_projection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
