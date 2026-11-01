"""Exercise dataset LiDAR filtering without nuScenes files or CUDA."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

path = Path(__file__).resolve().parents[1] / 'src/dataset/dataset_nuscenes.py'
tree = ast.parse(path.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DatasetNuScenes')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_load_lidar_depth')
namespace = {'np': np, 'torch': torch}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)


class LidarDynamicFilterTests(unittest.TestCase):
    def test_dynamic_points_never_enter_depth_map(self):
        records = {
            'sample': {'data': {'LIDAR_TOP': 'lidar'}},
            'sample_data': {'calibrated_sensor_token': 'calib'},
            'calibrated_sensor': {'camera_intrinsic': np.eye(3)},
        }
        # Two points collide at static pixel 0; dynamic pixel 1 is discarded.
        points = np.array([[0., 0., 1., 2.], [0., 0., 0., 0.], [1., 1., 1., 1.]])
        nusc = SimpleNamespace(
            get=lambda table, token: records[table],
            explorer=SimpleNamespace(map_pointcloud_to_image=lambda *args: (points, np.array([8., 5., 3., 7.]), None)),
        )
        load = namespace['_load_lidar_depth']
        mask = torch.tensor([[[0., 1., 0.]]])
        depth = load(SimpleNamespace(nusc=nusc), 'sample', 'camera', torch.diag(torch.tensor([1/3, 1., 1.])), (1, 3), mask)
        torch.testing.assert_close(depth, torch.tensor([[5., 0., 7.]]))
        empty = load(SimpleNamespace(nusc=nusc), 'sample', 'camera', torch.diag(torch.tensor([1/3, 1., 1.])), (1, 3), torch.ones_like(mask))
        torch.testing.assert_close(empty, torch.zeros_like(empty))


if __name__ == '__main__':
    unittest.main()
