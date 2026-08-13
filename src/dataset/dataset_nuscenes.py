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
from nuscenes.utils.data_classes import LidarPointCloud


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

    camera_name: str = "CAM_FRONT"
    num_context_views: int = 2
    num_target_views: int = 1
    frame_stride: int = 1


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

        self.items = self._build_index()

    def _moving_vehicle_mask(
        self,
        sample_token: str,
        camera_to_world: Tensor,
        K_norm: Tensor,
        image_shape: tuple[int, int],
    ) -> Tensor:
        """Return an expanded image mask for explicit vehicle.moving boxes."""
        height, width = image_shape
        mask = torch.zeros((1, height, width), dtype=torch.float32)
        world_to_camera = self._invert_transform(camera_to_world)
        sample = self.nusc.get("sample", sample_token)

        for annotation_token in sample["anns"]:
            annotation = self.nusc.get("sample_annotation", annotation_token)
            attribute_names = {
                self.nusc.get("attribute", token)["name"]
                for token in annotation["attribute_tokens"]
            }
            if "vehicle.moving" not in attribute_names:
                continue

            corners_world = torch.from_numpy(
                self.nusc.get_box(annotation_token).corners().T
            ).to(dtype=torch.float32)
            corners_h = torch.cat(
                (corners_world, torch.ones((corners_world.shape[0], 1))),
                dim=-1,
            )
            corners_camera = (world_to_camera @ corners_h.T).T[:, :3]
            # Very small positive depths make perspective projection explode and
            # can turn one partially visible box into an almost full-image mask.
            visible = corners_camera[:, 2] > 0.1
            if not visible.any():
                continue

            corners_camera = corners_camera[visible]
            x, y, z = corners_camera.unbind(dim=-1)
            u = K_norm[0, 0] * width * x / z + K_norm[0, 2] * width
            v = K_norm[1, 1] * height * y / z + K_norm[1, 2] * height

            x_min, x_max = u.min().item(), u.max().item()
            y_min, y_max = v.min().item(), v.max().item()
            if x_max < 0 or x_min >= width or y_max < 0 or y_min >= height:
                continue

            # Expand the part visible in the image. Using the unbounded projected
            # extent would let a mostly off-screen box create a huge padding.
            x_min_visible = max(0.0, x_min)
            x_max_visible = min(float(width - 1), x_max)
            y_min_visible = max(0.0, y_min)
            y_max_visible = min(float(height - 1), y_max)
            padding_x = max(8.0, 0.2 * (x_max_visible - x_min_visible))
            padding_y = max(8.0, 0.2 * (y_max_visible - y_min_visible))
            x0 = max(0, int(np.floor(x_min_visible - padding_x)))
            x1 = min(width, int(np.ceil(x_max_visible + padding_x)) + 1)
            y0 = max(0, int(np.floor(y_min_visible - padding_y)))
            y1 = min(height, int(np.ceil(y_max_visible + padding_y)) + 1)
            if x0 < x1 and y0 < y1:
                mask[:, y0:y1, x0:x1] = 1.0

        return mask

    @staticmethod
    def _invert_transform(T: Tensor) -> Tensor:
        return torch.linalg.inv(T)

    def _project_lidar_to_camera(
        self,
        lidar_sd_token: str,
        camera_sd_token: str,
        K_norm: Tensor,
        image_shape: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
    Project LIDAR_TOP points to the given camera image plane.

    Args:
        lidar_sd_token:  sample_data token for LIDAR_TOP
        camera_sd_token: sample_data token for camera
        K_norm:          normalized intrinsics [3,3]
        image_shape:     (H, W) after crop

    Returns:
        lidar_depth: [1, H, W]
        lidar_mask:  [1, H, W]
        lidar_world: [3, H, W], nearest projected point in world coordinates
        """
        H, W = image_shape

        lidar_sd = self.nusc.get("sample_data", lidar_sd_token)
        cam_sd = self.nusc.get("sample_data", camera_sd_token)

        lidar_cs = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_ep = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        cam_cs = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_ep = self.nusc.get("ego_pose", cam_sd["ego_pose_token"])

        # 读 LiDAR 点云
        lidar_path = self.cfg.root / lidar_sd["filename"]
        pc = LidarPointCloud.from_file(str(lidar_path))

        # points: [4, N] or [5, N], we only need xyz
        pts = torch.from_numpy(pc.points[:3].T).float()  # [N, 3]
        if pts.numel() == 0:
            depth = torch.zeros((1, H, W), dtype=torch.float32)
            mask = torch.zeros((1, H, W), dtype=torch.float32)
            world = torch.zeros((3, H, W), dtype=torch.float32)
            return depth, mask, world

        # lidar -> ego(lidar time)
        T_lidar_to_ego = self._make_transform(
            lidar_cs["translation"],
            lidar_cs["rotation"],
        )
        # ego(lidar time) -> world
        T_ego_to_world_lidar = self._make_transform(
            lidar_ep["translation"],
            lidar_ep["rotation"],
        )

        # camera -> ego(camera time)
        T_cam_to_ego = self._make_transform(
            cam_cs["translation"],
            cam_cs["rotation"],
        )
        # ego(camera time) -> world
        T_ego_to_world_cam = self._make_transform(
            cam_ep["translation"],
            cam_ep["rotation"],
        )

        # lidar -> world
        T_lidar_to_world = T_ego_to_world_lidar @ T_lidar_to_ego
        # world -> camera
        T_world_to_cam = self._invert_transform(T_ego_to_world_cam @ T_cam_to_ego)

        # transform points: lidar -> world -> camera
        pts_h = torch.cat([pts, torch.ones((pts.shape[0], 1))], dim=1)   # [N,4]
        pts_world = (T_lidar_to_world @ pts_h.T).T[:, :3]
        pts_cam = (T_world_to_cam @ T_lidar_to_world @ pts_h.T).T[:, :3] # [N,3]

        x = pts_cam[:, 0]
        y = pts_cam[:, 1]
        z = pts_cam[:, 2]

        # 只保留在相机前方的点
        valid = (z > self.cfg.near) & (z < self.cfg.far)
        x = x[valid]
        y = y[valid]
        z = z[valid]
        pts_world = pts_world[valid]

        if z.numel() == 0:
            depth = torch.zeros((1, H, W), dtype=torch.float32)
            mask = torch.zeros((1, H, W), dtype=torch.float32)
            world = torch.zeros((3, H, W), dtype=torch.float32)
            return depth, mask, world

        # normalized K -> pixel K
        fx = K_norm[0, 0] * W
        fy = K_norm[1, 1] * H
        cx = K_norm[0, 2] * W
        cy = K_norm[1, 2] * H

        u = fx * (x / z) + cx
        v = fy * (y / z) + cy

        in_image = (
            (u >= 0)
            & (u < W)
            & (v >= 0)
            & (v < H)
        )
        u = u[in_image]
        v = v[in_image]
        z = z[in_image]
        pts_world = pts_world[in_image]

        u = u.long()
        v = v.long()

        if z.numel() == 0:
            depth = torch.zeros((1, H, W), dtype=torch.float32)
            mask = torch.zeros((1, H, W), dtype=torch.float32)
            world = torch.zeros((3, H, W), dtype=torch.float32)
            return depth, mask, world

        # 同一像素可能投到多个点，保留最近深度
        linear_idx = v * W + u
        depth_flat = torch.full((H * W,), float("inf"), dtype=torch.float32)

        # 需要 PyTorch 支持 scatter_reduce_
        depth_flat.scatter_reduce_(0, linear_idx, z, reduce="amin", include_self=True)

        # Use camera Z only for visibility. Store the complete XYZ tuple of
        # the selected point; reducing its components independently is invalid.
        nearest = z <= depth_flat[linear_idx] + 1e-6
        world_flat = torch.zeros((H * W, 3), dtype=torch.float32)
        world_flat[linear_idx[nearest]] = pts_world[nearest]

        mask_flat = torch.isfinite(depth_flat)
        depth_flat[~mask_flat] = 0.0

        depth = depth_flat.view(1, H, W)
        mask = mask_flat.float().view(1, H, W)
        world = world_flat.view(H, W, 3).permute(2, 0, 1).contiguous()
        return depth, mask, world
    def _build_index(self) -> list[dict]:
        """
        Build temporal windows on a single camera stream, e.g. CAM_FRONT.

        【修改后】
        Each item explicitly stores:
            context_sample_tokens = [t-s, t+s]
            target_sample_tokens  = [t]

        For example:
            num_context_views = 2
            num_target_views  = 1
            frame_stride      = 1

        Then:
            context = [t-1, t+1]
            target  = [t]
        """
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
        camera_name = self.cfg.camera_name

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
                        "context_sample_tokens": context_sample_tokens,
                        "target_sample_tokens": target_sample_tokens,
                        "target_time_index": center,
                    }
                )

        return items

    def __iter__(self):
        indices = list(range(len(self.items)))

        if self.stage == "train":
            perm = torch.randperm(len(indices)).tolist()
            indices = [indices[i] for i in perm]

        for idx in indices:
            item = self.items[idx]
            scene_name = item["scene_name"]

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
            lidar_sd_tokens = []

            for sample_token in sample_tokens:
                sample = self.nusc.get("sample", sample_token)

                cam_sd_token = sample["data"][self.cfg.camera_name]
                lidar_sd_token = sample["data"]["LIDAR_TOP"]

                img_tensor, c2w, K = self._load_camera(cam_sd_token)

                image_tensors.append(img_tensor)
                intrinsics_list.append(K)
                extrinsics_list.append(c2w)

                camera_sd_tokens.append(cam_sd_token)
                lidar_sd_tokens.append(lidar_sd_token)

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

            if self.cfg.skip_bad_shape:
                context_image_invalid = context_images.ndim != 4
                target_image_invalid = target_images.ndim != 4
                if context_image_invalid or target_image_invalid:
                    print(
                        f"Skipped bad example {scene_name}. "
                        f"Context shape={context_images.shape}, "
                        f"Target shape={target_images.shape}"
                    )
                    continue

            example = {
                "context": {
                    "extrinsics": extrinsics[context_indices],
                    "intrinsics": intrinsics[context_indices],
                    "image": context_images,
                    "near": self.get_bound("near", len(context_indices)),
                    "far": self.get_bound("far", len(context_indices)),
                    "index": context_indices,
                },
                "target": {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)),
                    "far": self.get_bound("far", len(target_indices)),
                    "index": target_indices,
                },
                "scene": f"{scene_name}_{sample_tokens[0]}",
            }

            if self.stage == "train" and self.cfg.augment:
                example = apply_augmentation_shim(example)

            example = apply_crop_shim(example, tuple(self.cfg.image_shape))

            target_dynamic_masks = []
            for i, target_idx in enumerate(target_indices.tolist()):
                target_image = example["target"]["image"][i]
                target_dynamic_masks.append(
                    self._moving_vehicle_mask(
                        sample_token=sample_tokens[target_idx],
                        camera_to_world=example["target"]["extrinsics"][i],
                        K_norm=example["target"]["intrinsics"][i],
                        image_shape=tuple(target_image.shape[-2:]),
                    )
                )
            example["target"]["dynamic_mask"] = torch.stack(
                target_dynamic_masks, dim=0
            )

            context_dynamic_masks = []
            for i, context_idx in enumerate(context_indices.tolist()):
                context_image = example["context"]["image"][i]
                context_dynamic_masks.append(
                    self._moving_vehicle_mask(
                        sample_token=sample_tokens[context_idx],
                        camera_to_world=example["context"]["extrinsics"][i],
                        K_norm=example["context"]["intrinsics"][i],
                        image_shape=tuple(context_image.shape[-2:]),
                    )
                )
            example["context"]["dynamic_mask"] = torch.stack(
                context_dynamic_masks, dim=0
            )

            # ===== 在 crop 之后生成 LiDAR depth / mask =====
            context_lidar_depths = []
            context_lidar_masks = []
            context_lidar_worlds = []

            for i, ctx_idx in enumerate(context_indices.tolist()):
                H, W = example["context"]["image"][i].shape[-2:]
                K_norm = example["context"]["intrinsics"][i]

                lidar_depth, lidar_mask, lidar_world = self._project_lidar_to_camera(
                lidar_sd_token=lidar_sd_tokens[ctx_idx],
                camera_sd_token=camera_sd_tokens[ctx_idx],
                K_norm=K_norm,
                image_shape=(H, W),)

                # Remove moving-vehicle returns before any LiDAR tensor reaches
                # the encoder. RGB context features and cost-volume construction
                # remain untouched.
                static_lidar_mask = lidar_mask * (
                    1.0 - example["context"]["dynamic_mask"][i]
                )
                lidar_depth = lidar_depth * static_lidar_mask
                lidar_world = lidar_world * static_lidar_mask
                lidar_mask = static_lidar_mask

                context_lidar_depths.append(lidar_depth)
                context_lidar_masks.append(lidar_mask)
                context_lidar_worlds.append(lidar_world)

            example["context"]["lidar_depth"] = torch.stack(context_lidar_depths, dim=0)  # [v,1,H,W]
            example["context"]["lidar_mask"] = torch.stack(context_lidar_masks, dim=0)    # [v,1,H,W]
            example["context"]["lidar_world"] = torch.stack(context_lidar_worlds, dim=0)  # [v,3,H,W]

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
