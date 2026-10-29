"""Check direct inverse-depth supervision independently of the CUDA stack."""

import ast
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

path = Path(__file__).resolve().parents[1] / "src/loss/loss_lidar_inverse_depth.py"
tree = ast.parse(path.read_text())
function = next(
    n
    for n in tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "sparse_inverse_depth_loss"
)
namespace = {"torch": torch, "F": F}
exec(
    compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace
)
loss_fn = namespace["sparse_inverse_depth_loss"]


class LidarInverseDepthTests(unittest.TestCase):
    def test_gradient_recovers_both_clamp_boundaries(self):
        raw = torch.tensor([[[[[-0.7], [1.4]]]]], requires_grad=True)
        lidar = torch.full((1, 1, 1, 2), 10.0, requires_grad=True)
        near, far = torch.ones(1, 1), torch.full((1, 1), 80.0)
        # Both predictions would have zero clamp derivative.
        clamped = raw.clamp(1 / 80, 1)
        clamp_grad = torch.autograd.grad(clamped.sum(), raw, retain_graph=True)[0]
        torch.testing.assert_close(clamp_grad, torch.zeros_like(raw))
        loss = loss_fn(raw, lidar, near, far, 0.01)
        loss.backward()
        self.assertLess(raw.grad[0, 0, 0, 0, 0].item(), 0)
        self.assertGreater(raw.grad[0, 0, 0, 1, 0].item(), 0)
        self.assertIsNone(lidar.grad)

    def test_missing_invalid_and_out_of_range_points_have_no_gradient(self):
        raw = torch.full((1, 1, 1, 6, 1), -0.2, requires_grad=True)
        lidar = torch.tensor([[[[0.0, float("nan"), float("inf"), 0.5, 90.0, 10.0]]]])
        loss = loss_fn(raw, lidar, torch.ones(1, 1), torch.full((1, 1), 80.0), 0.01)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(
            raw.grad[..., :5, :], torch.zeros_like(raw.grad[..., :5, :])
        )
        self.assertLess(raw.grad[..., 5, :].item(), 0)

    def test_empty_map_and_multiple_samples(self):
        raw = torch.full((2, 2, 2, 2, 2), 0.1, requires_grad=True)
        lidar = torch.zeros(2, 2, 2, 2)
        near, far = torch.ones(2, 2), torch.full((2, 2), 80.0)
        loss = loss_fn(raw, lidar, near, far, 0.01)
        loss.backward()
        self.assertEqual(loss.item(), 0)
        torch.testing.assert_close(raw.grad, torch.zeros_like(raw))
        lidar[1, 0, 1, 1] = 5.0
        loss = loss_fn(raw, lidar, near, far, 0.01)
        self.assertAlmostEqual(loss.item(), 0.095, places=6)


if __name__ == "__main__":
    unittest.main()
