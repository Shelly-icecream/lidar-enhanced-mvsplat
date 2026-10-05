"""Sparse depth replacement and gradients, without the optional CUDA stack."""

import importlib.util
from pathlib import Path
import unittest
import torch

path = (
    Path(__file__).resolve().parents[1] / "src/model/encoder/costvolume/lidar_bias.py"
)
spec = importlib.util.spec_from_file_location("lidar_bias_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LidarBiasTests(unittest.TestCase):
    def test_valid_points_replace_all_samples_and_stop_network_gradient(self):
        depths = torch.full((1, 2, 4, 1, 2), 80.0, requires_grad=True)
        lidar = torch.tensor(
            [[[[10.0, 0.0], [float("nan"), 90.0]], [[20.0, 0.5], [1.0, 80.0]]]]
        )
        result = module.replace_depth_with_lidar(
            depths, lidar, torch.ones(1, 2), torch.full((1, 2), 80.0)
        )
        expected = torch.tensor([[[10.0, 80.0, 80.0, 80.0], [20.0, 80.0, 1.0, 80.0]]])
        torch.testing.assert_close(result[..., 0, 0], expected)
        torch.testing.assert_close(result[..., 0, 1], expected)
        result.sum().backward()
        expected_grad = torch.tensor([[[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]]])
        torch.testing.assert_close(depths.grad[..., 0, 0], expected_grad)

    def test_batch_and_view_order_and_empty_map(self):
        depths = torch.full((2, 2, 4, 1, 1), 30.0)
        lidar = torch.zeros(2, 2, 2, 2)
        torch.testing.assert_close(
            module.replace_depth_with_lidar(
                depths, lidar, torch.ones(2, 2), torch.full((2, 2), 80.0)
            ),
            depths,
        )
        lidar[1, 0, 1, 0] = 12.0
        result = module.replace_depth_with_lidar(
            depths, lidar, torch.ones(2, 2), torch.full((2, 2), 80.0)
        )
        self.assertEqual(result[1, 0, 2, 0, 0].item(), 12.0)
        self.assertEqual((result != depths).sum().item(), 1)

    def test_mismatched_resolution_rejected(self):
        with self.assertRaises(ValueError):
            module.replace_depth_with_lidar(
                torch.zeros(1, 2, 4, 1, 1),
                torch.zeros(1, 2, 3, 3),
                torch.ones(1, 2),
                torch.full((1, 2), 80.0),
            )


if __name__ == "__main__":
    unittest.main()
