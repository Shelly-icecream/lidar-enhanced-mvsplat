"""Checkpoint and geometry-isolation regression tests for the Stage6 head."""
import unittest

import torch
from torch import nn

from src.model.encoder.costvolume.sh_head import SplitGaussianHead


class SplitSHHeadTest(unittest.TestCase):
    def test_migration_and_geometry_isolation(self):
        for surfaces in (1, 2):
            with self.subTest(surfaces=surfaces):
                joint = nn.Sequential(nn.Conv2d(5, 16, 3, padding=1), nn.GELU(),
                                      nn.Conv2d(16, 21 * surfaces, 3, padding=1))
                head = SplitGaussianHead(joint, 21)
                head.load_state_dict(joint.state_dict(), strict=True)
                features = torch.randn(2, 5, 8, 9)
                torch.testing.assert_close(head(features), joint(features))
                head.geometry.requires_grad_(False)
                before = head(features).detach().reshape(2, surfaces, 21, 8, 9)
                optimizer = torch.optim.Adam(head.sh.parameters(), lr=0.01)
                head(features).square().mean().backward()
                optimizer.step()
                after = head(features).detach().reshape(2, surfaces, 21, 8, 9)
                self.assertTrue(torch.equal(before[:, :, :9], after[:, :, :9]))
                self.assertFalse(torch.equal(before[:, :, 9:], after[:, :, 9:]))
                restored = SplitGaussianHead(joint, 21)
                restored.load_state_dict(head.state_dict(), strict=True)
                torch.testing.assert_close(restored(features), head(features))
