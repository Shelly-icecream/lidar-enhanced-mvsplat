"""Numerical tests without importing the optional rendering/dataset stack."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from dataclasses import fields
from typing import Generic, TypeVar

import torch


def load_loss():
    package = types.ModuleType('_depth_loss_test')
    package.__path__ = []
    sys.modules[package.__name__] = package
    base = types.ModuleType('_depth_loss_test.loss')
    T, U = TypeVar('T'), TypeVar('U')

    class Loss(torch.nn.Module, Generic[T, U]):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = getattr(cfg, fields(cfg)[0].name)

    base.Loss = Loss
    sys.modules[base.__name__] = base
    path = Path(__file__).resolve().parents[1] / 'src/loss/loss_cross_visual_depth.py'
    spec = importlib.util.spec_from_file_location('_depth_loss_test.cross', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = load_loss()


class CrossDepthTests(unittest.TestCase):
    def setUp(self):
        self.loss = module.LossCrossVisualDepth(module.LossCrossVisualDepthCfgWrapper(
            module.LossCrossVisualDepthCfg(print_every_n_steps=0)))
        self.k = torch.tensor([[[1., 0., .5], [0., 1., .5], [0., 0., 1.]]])
        self.pose = torch.eye(4)[None]
        self.image = torch.zeros(1, 3, 8, 8)

    def direction(self, a, b, image_b=None, pose_b=None):
        return self.loss.direction(a, b, self.image, self.image if image_b is None else image_b,
                                   self.k, self.k, self.pose, self.pose if pose_b is None else pose_b)

    def test_identity(self):
        a = torch.full((1, 8, 8), 10., requires_grad=True)
        loss, ratio = self.direction(a, a)
        self.assertAlmostEqual(loss.item(), 0.)
        self.assertAlmostEqual(ratio.item(), 1.)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_relative_residual_and_both_gradients(self):
        a = torch.full((1, 8, 8), 10., requires_grad=True)
        b = torch.full((1, 8, 8), 10.2, requires_grad=True)
        loss, ratio = self.direction(a, b)
        expected = abs((10 - 10.2) / 10.2) - .005
        self.assertAlmostEqual(loss.item(), expected, places=6)
        self.assertEqual(ratio.item(), 1.)
        self.assertFalse(ratio.requires_grad)
        loss.backward()
        self.assertTrue((a.grad < 0).all())
        self.assertTrue((b.grad > 0).all())
        scaled, _ = self.direction(a.detach() * 10, b.detach() * 10)
        self.assertAlmostEqual(loss.item(), scaled.item(), places=6)

    def test_rejected_points_safe_zero(self):
        for b, color in ((100., None), (10., torch.ones_like(self.image)), (float('nan'), None), (0., None)):
            a = torch.full((1, 8, 8), 10., requires_grad=True)
            target = torch.full((1, 8, 8), b, requires_grad=True)
            loss, ratio = self.direction(a, target, color)
            self.assertEqual(loss.item(), 0.)
            self.assertEqual(ratio.item(), 0.)
            loss.backward()
            self.assertTrue(torch.isfinite(a.grad).all())
            self.assertTrue(torch.isfinite(target.grad).all())

    def test_translated_plane_and_roundtrip_filter(self):
        pose_b = self.pose.clone()
        pose_b[:, 0, 3] = 1.
        a = torch.full((1, 8, 8), 10.)
        loss, ratio = self.direction(a, a, pose_b=pose_b)
        self.assertAlmostEqual(loss.item(), 0., places=6)
        self.assertGreater(ratio.item(), 0.)
        self.assertLess(ratio.item(), 1.)
        self.loss.cfg.reprojection_threshold_px = .001
        loss, ratio = self.direction(a, a * 1.02, pose_b=pose_b)
        self.assertEqual(ratio.item(), 0.)

    def test_symmetric_weight(self):
        depth = torch.stack((torch.full((1, 8, 8), 10.), torch.full((1, 8, 8), 10.2)), 1)
        self.loss.cfg.weight = 2.
        batch = {'context': {'image': self.image[:, None].expand(-1, 2, -1, -1, -1),
                             'intrinsics': self.k[:, None].expand(-1, 2, -1, -1),
                             'extrinsics': self.pose[:, None].expand(-1, 2, -1, -1)}}
        result = self.loss(None, batch, types.SimpleNamespace(context_depth=depth), 1)
        ab, _ = self.direction(depth[:, 0], depth[:, 1])
        ba, _ = self.direction(depth[:, 1], depth[:, 0])
        self.assertAlmostEqual(result.item(), (ab + ba).item(), places=6)

    def test_context_masks_filter_both_directions(self):
        depth = torch.stack((torch.full((1, 8, 8), 10.),
                             torch.full((1, 8, 8), 10.25)), 1).requires_grad_()
        masks = torch.zeros(1, 2, 1, 8, 8)
        masks[:, 0, :, :2] = 1
        masks[:, 1, :, 2:4] = 1
        batch = {'context': {'image': self.image[:, None].expand(-1, 2, -1, -1, -1),
                             'intrinsics': self.k[:, None].expand(-1, 2, -1, -1),
                             'extrinsics': self.pose[:, None].expand(-1, 2, -1, -1),
                             'dynamic_mask': masks}}
        result = self.loss(None, batch, types.SimpleNamespace(context_depth=depth), 1)
        self.assertAlmostEqual(self.loss.diagnostics['valid_ratio'].item(), .5)
        result.backward()
        self.assertTrue((depth.grad[:, :, :4] == 0).all())
        self.assertTrue((depth.grad[:, :, 4:] != 0).all())
        masks.fill_(1)
        result = self.loss(None, batch, types.SimpleNamespace(context_depth=depth), 1)
        self.assertEqual(result.item(), 0.)
        self.assertEqual(self.loss.diagnostics['valid_ratio'].item(), 0.)

    def test_target_mask_filters_bilinear_sampling_region(self):
        depth = torch.full((1, 8, 8), 10.)
        target_pose = self.pose.clone()
        target_pose[:, 0, 3] = .625  # Half a pixel horizontally.
        target_mask = torch.zeros(1, 1, 8, 8)
        target_mask[:, :, :, 3] = 1
        _, ratio = self.loss.direction(
            depth, depth, self.image, self.image, self.k, self.k,
            self.pose, target_pose, torch.zeros_like(target_mask), target_mask)
        # Seven columns are in bounds; two sample the dynamic column.
        self.assertAlmostEqual(ratio.item(), 5 / 8)


if __name__ == '__main__':
    unittest.main()
