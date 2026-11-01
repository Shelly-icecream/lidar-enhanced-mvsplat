from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
import numpy as np


DYNAMIC_ATTRIBUTES = {
    "vehicle.moving",
    "pedestrian.moving",
    "cycle.with_rider",
}


@dataclass
class DatasetNuScenesCfg(DatasetCfgCommon):
    name: Literal["nuscenes"]
    root: Path
    version: str = "v1.0-mini"
    augment: bool = False
    near: float = 1.0
    far: float = 80.0
    skip_bad_shape: bool = True
    max_fov: float = 120.0
    use_keyframe_only: bool = True
    load_lidar_depth: bool = False

    camera_name: str = "CAM_FRONT"
    # Each selected camera contributes independent two-context temporal windows.
    camera_names: list[str] | None = None
    num_context_views: int = 2
    num_target_views: int = 1
    frame_stride: int = 1
    dynamic_mask_expansion_ratio: float = 0.15
    dynamic_mask_min_padding_px: int = 8
    dynamic_mask_min_depth: float = 0.01


class DatasetNuScenes(IterableDataset):
    cfg: DatasetNuScenesCfg
    stage: Stage
    view_sampler: ViewSampler

    def __init__(
        self,
        cfg: DatasetNuScenesCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.cfg.root = Path(self.cfg.root)
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self.nusc = NuScenes(
            version=cfg.version,
            dataroot=str(cfg.root),
            verbose=False,
        )
        self.dynamic_attribute_tokens = {
            attribute["token"]
            for attribute in self.nusc.attribute
            if attribute["name"] in DYNAMIC_ATTRIBUTES
        }

        self.items = self._build_index()

    @staticmethod
    def _invert_transform(T: Tensor) -> Tensor:
        return torch.linalg.inv(T)

    def _project_dynamic_mask_to_camera(
        self,
        sample_token: str,
        camera_sd_token: str,
        K_norm: Tensor,
        image_shape: tuple[int, int],
    ) -> Tensor:
        """Project dynamic nuScenes annotations to an image-space box mask."""
        height, width = image_shape
        dynamic_mask = torch.zeros((1, height, width), dtype=torch.float32)

        sample = self.nusc.get("sample", sample_token)
        camera_sd = self.nusc.get("sample_data", camera_sd_token)
        camera_cs = self.nusc.get(
            "calibrated_sensor", camera_sd["calibrated_sensor_token"]
        )
        camera_pose = self.nusc.get("ego_pose", camera_sd["ego_pose_token"])

        camera_to_ego = self._make_transform(
            camera_cs["translation"], camera_cs["rotation"]
        )
        ego_to_world = self._make_transform(
            camera_pose["translation"], camera_pose["rotation"]
        )
        world_to_camera = self._invert_transform(ego_to_world @ camera_to_ego)

        fx = K_norm[0, 0] * width
        fy = K_norm[1, 1] * height
        cx = K_norm[0, 2] * width
        cy = K_norm[1, 2] * height

        for annotation_token in sample["anns"]:
            annotation = self.nusc.get("sample_annotation", annotation_token)
            if not (
                set(annotation["attribute_tokens"])
                & self.dynamic_attribute_tokens
            ):
                continue

            # get_box returns the annotation box in the global coordinate frame.
            corners_world = torch.from_numpy(
                self.nusc.get_box(annotation_token).corners().T
            ).to(dtype=torch.float32)
            corners_h = torch.cat(
                [corners_world, torch.ones((corners_world.shape[0], 1))], dim=1
            )
            corners_camera = (world_to_camera @ corners_h.T).T[:, :3]
            visible = corners_camera[:, 2] > self.cfg.dynamic_mask_min_depth
            if not visible.any():
                continue

            corners_camera = corners_camera[visible]
            projected_x = fx * (
                corners_camera[:, 0] / corners_camera[:, 2]
            ) + cx
            projected_y = fy * (
                corners_camera[:, 1] / corners_camera[:, 2]
            ) + cy

            xmin = projected_x.min().item()
            xmax = projected_x.max().item()
            ymin = projected_y.min().item()
            ymax = projected_y.max().item()

            # Test intersection before clipping so partially visible boxes survive.
            if xmax < 0 or xmin >= width or ymax < 0 or ymin >= height:
                continue

            pad_x = max(
                self.cfg.dynamic_mask_min_padding_px,
                round((xmax - xmin) * self.cfg.dynamic_mask_expansion_ratio),
            )
            pad_y = max(
                self.cfg.dynamic_mask_min_padding_px,
                round((ymax - ymin) * self.cfg.dynamic_mask_expansion_ratio),
            )
            x0 = max(0, int(np.floor(xmin - pad_x)))
            y0 = max(0, int(np.floor(ymin - pad_y)))
            x1 = min(width, int(np.ceil(xmax + pad_x)) + 1)
            y1 = min(height, int(np.ceil(ymax + pad_y)) + 1)
            if x0 < x1 and y0 < y1:
                dynamic_mask[:, y0:y1, x0:x1] = 1.0

        return dynamic_mask

    def _build_index(self) -> list[dict]:
        """Build temporal windows per camera, keeping calibration streams separate."""
        split_scenes = create_splits_scenes()

        if self.cfg.version == "v1.0-mini":
            if self.stage == "train":
                allowed_scene_names = set(split_scenes["mini_train"])
            elif self.stage in ("val", "test"):
                allowed_scene_names = set(split_scenes["mini_val"])
            else:
                raise ValueError(f"Unknown stage: {self.stage}")
        else:
            if self.stage == "train":
                allowed_scene_names = set(split_scenes["train"])
            elif self.stage == "val":
                allowed_scene_names = set(split_scenes["val"])
            elif self.stage == "test":
                allowed_scene_names = set(split_scenes["test"])
            else:
                raise ValueError(f"Unknown stage: {self.stage}")

        items = []
        camera_names = getattr(self.cfg, "camera_names", None)
        if camera_names is None:
            camera_names = [self.cfg.camera_name]
        supported = {
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
        }
        if not camera_names or len(set(camera_names)) != len(camera_names):
            raise ValueError("camera_names must be nonempty and contain no duplicates.")
        if set(camera_names) - supported:
            raise ValueError(f"Unsupported nuScenes cameras: {set(camera_names) - supported}")

        num_context = self.cfg.num_context_views
        num_target = self.cfg.num_target_views
        stride = self.cfg.frame_stride
        
        if num_context != 2:
            raise ValueError(
                "Current temporal NuScenes sampler expects num_context_views=2. "
                f"Got num_context_views={num_context}."
            )
        if stride < 1:
            raise ValueError(f"frame_stride must be positive, got {stride}.")

        if num_target != 1:
            raise ValueError(
                "Current temporal NuScenes sampler expects num_target_views=1. "
                f"Got num_target_views={num_target}."
            )

        for scene in self.nusc.scene:
            scene_name = scene["name"]
            if scene_name not in allowed_scene_names:
                continue

            for camera_name in camera_names:
                # Traverse sample chain of this scene.
                sample_tokens = []
                token = scene["first_sample_token"]
                while token != "":
                    sample = self.nusc.get("sample", token)

                    if camera_name not in sample["data"]:
                        token = sample["next"]
                        continue

                    sd_token = sample["data"][camera_name]
                    sd = self.nusc.get("sample_data", sd_token)

                    # optional: keyframe only
                    if self.cfg.use_keyframe_only and not sd["is_key_frame"]:
                        token = sample["next"]
                        continue

                    sample_tokens.append(token)
                    token = sample["next"]

                # num_context=2, stride=1: context=[t-1, t+1], target=[t].
                # num_context=2, stride=2: context=[t-2, t+2], target=[t].
                # Target-centered interpolation: [t-stride, t+stride] -> t.
                for center in range(stride, len(sample_tokens) - stride):
                    context_sample_tokens = [
                        sample_tokens[center - stride],
                        sample_tokens[center + stride],
                    ]

                    target_sample_tokens = [sample_tokens[center]]

                    items.append(
                        {
                            "scene_name": scene_name,
                            "camera_name": camera_name,
                            "context_sample_tokens": context_sample_tokens,
                            "target_sample_tokens": target_sample_tokens,
                            "target_time_index": center,
                        }
                    )

        return items

    def _load_lidar_depth(self, sample_token, camera_token, intrinsics, shape, dynamic_mask):
        """Project static single-sweep LiDAR; missing/dynamic pixels remain zero."""
        sample = self.nusc.get("sample", sample_token)
        points, z, _ = self.nusc.explorer.map_pointcloud_to_image(
            sample["data"]["LIDAR_TOP"], camera_token
        )
        camera = self.nusc.get("sample_data", camera_token)
        calibration = self.nusc.get("calibrated_sensor", camera["calibrated_sensor_token"])
        original_k = np.asarray(calibration["camera_intrinsic"])
        rays = np.linalg.solve(original_k, np.vstack((points[:2], np.ones(len(z)))))
        xy = intrinsics.numpy() @ rays
        h, w = shape
        x = np.floor(xy[0] * w).astype(np.int64)
        y = np.floor(xy[1] * h).astype(np.int64)
        valid = np.isfinite(z) & (z > 0) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
        # Discard dynamic-box points before rasterizing the context depth map.
        point_indices = np.flatnonzero(valid)
        static = dynamic_mask.squeeze(0).cpu().numpy() < 0.5
        valid[point_indices] &= static[y[point_indices], x[point_indices]]
        depth = np.full(h * w, np.inf, dtype=np.float32)
        # Retain the nearest surface when several points land in one pixel.
        np.minimum.at(depth, y[valid] * w + x[valid], z[valid])
        depth[~np.isfinite(depth)] = 0
        return torch.from_numpy(depth.reshape(h, w))

    def __iter__(self):
        indices = list(range(len(self.items)))

        if self.stage == "train":
            perm = torch.randperm(len(indices)).tolist()
            indices = [indices[i] for i in perm]

        for idx in indices:
            item = self.items[idx]
            scene_name = item["scene_name"]
            camera_name = item["camera_name"]

            # 【修改】现在 _build_index() 已经显式区分 context / target
            context_sample_tokens = item["context_sample_tokens"]
            target_sample_tokens = item["target_sample_tokens"]

            # 【新增】后续统一加载这些 token。
            # 顺序固定为：context 在前，target 在后。
            # 因此后面的 context_indices / target_indices 仍然可以保持原写法。
            sample_tokens = context_sample_tokens + target_sample_tokens

            image_tensors = []
            intrinsics_list = []
            extrinsics_list = []

            camera_sd_tokens = []

            for sample_token in sample_tokens:
                sample = self.nusc.get("sample", sample_token)

                cam_sd_token = sample["data"][camera_name]

                img_tensor, c2w, K = self._load_camera(cam_sd_token)

                image_tensors.append(img_tensor)
                intrinsics_list.append(K)
                extrinsics_list.append(c2w)

                camera_sd_tokens.append(cam_sd_token)

            images = torch.stack(image_tensors, dim=0)         # [v, 3, h, w]
            intrinsics = torch.stack(intrinsics_list, dim=0)   # [v, 3, 3]
            extrinsics = torch.stack(extrinsics_list, dim=0)   # [v, 4, 4]

            # 暂时手动固定，绕过 view_sampler
            num_context = self.cfg.num_context_views
            num_target = self.cfg.num_target_views

            context_indices = torch.arange(0, num_context, dtype=torch.long)
            target_indices = torch.arange(
                num_context, num_context + num_target, dtype=torch.long
            )

            context_images = images[context_indices]
            target_images = images[target_indices]

            example = {
                "context": {
                    "extrinsics": extrinsics[context_indices],
                    "intrinsics": intrinsics[context_indices],
                    "image": context_images,
                    "near": self.get_bound("near", len(context_indices)),
                    "far": self.get_bound("far", len(context_indices)),
                    # Loading slots [0, 1] are not scene time indices.
                    "index": torch.tensor([
                        item["target_time_index"] - self.cfg.frame_stride,
                        item["target_time_index"] + self.cfg.frame_stride,
                    ], dtype=torch.long),
                },
                "target": {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)),
                    "far": self.get_bound("far", len(target_indices)),
                    "index": torch.tensor([item["target_time_index"]], dtype=torch.long),
                },
                "scene": (
                    f"{scene_name}_{camera_name}_{sample_tokens[0]}"
                    if self.cfg.camera_names is not None
                    else f"{scene_name}_{sample_tokens[0]}"
                ),
            }

            example = apply_crop_shim(example, tuple(self.cfg.image_shape))
            # Only context annotations are used for dynamic masks. RGB remains
            # untouched for global feature extraction and cost-volume construction.
            example["context"]["dynamic_mask"] = torch.stack([
                self._project_dynamic_mask_to_camera(
                    sample_token=sample_token,
                    camera_sd_token=camera_sd_tokens[i],
                    K_norm=example["context"]["intrinsics"][i],
                    image_shape=tuple(example["context"]["image"].shape[-2:]),
                )
                for i, sample_token in enumerate(context_sample_tokens)
            ], dim=0)
            if self.cfg.load_lidar_depth:
                example["context"]["lidar_depth"] = torch.stack([
                    self._load_lidar_depth(
                        token, camera_sd_tokens[i],
                        example["context"]["intrinsics"][i],
                        tuple(example["context"]["image"].shape[-2:]),
                        example["context"]["dynamic_mask"][i],
                    ) for i, token in enumerate(context_sample_tokens)
                ])

            # Keep projected dynamic masks aligned with augmented RGB images.
            if self.stage == "train" and self.cfg.augment:
                example = apply_augmentation_shim(example)

            yield example

    def _load_camera(
        self,
        sample_data_token: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Returns:
            image:      [3, H, W]
            extrinsics: [4, 4]  camera-to-world
            intrinsics: [3, 3]  normalized intrinsics
        """
        sd = self.nusc.get("sample_data", sample_data_token)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ep = self.nusc.get("ego_pose", sd["ego_pose_token"])

        image_path = self.cfg.root / sd["filename"]
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        image = self.to_tensor(image)

        K = torch.eye(3, dtype=torch.float32)
        cam_K = torch.tensor(cs["camera_intrinsic"], dtype=torch.float32)
        K[:3, :3] = cam_K

        # normalize intrinsics to align with re10k convention
        K[0, 0] /= w
        K[1, 1] /= h
        K[0, 2] /= w
        K[1, 2] /= h

        cam_to_ego = self._make_transform(
            cs["translation"],
            cs["rotation"],
        )
        ego_to_world = self._make_transform(
            ep["translation"],
            ep["rotation"],
        )

        c2w = ego_to_world @ cam_to_ego

        # nuScenes camera coords -> common NeRF/3DGS coords
        #flip = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))
        #c2w = c2w @ flip

        return image, c2w, K

    @staticmethod
    def _make_transform(
        translation,
        rotation,
    ) -> Tensor:
        T = torch.eye(4, dtype=torch.float32)
        q = Quaternion(rotation)
        R = torch.tensor(q.rotation_matrix, dtype=torch.float32)
        t = torch.tensor(translation, dtype=torch.float32)

        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Tensor:
        value = torch.tensor(getattr(self.cfg, bound), dtype=torch.float32)
        return value.repeat(num_views)

    def __len__(self) -> int:
        return len(self.items)
