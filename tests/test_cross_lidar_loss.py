import unittest
from types import SimpleNamespace
import torch
from src.loss.loss_cross_lidar_depth import LossCrossLidarDepth, LossCrossLidarDepthCfg, LossCrossLidarDepthCfgWrapper
from src.dataset.shims.augmentation_shim import reflect_views

class CrossLossTest(unittest.TestCase):
    def test_gradient_and_empty_mask(self):
        loss = LossCrossLidarDepth(LossCrossLidarDepthCfgWrapper(LossCrossLidarDepthCfg()))
        ctx = dict(cross_lidar_depth=torch.tensor([[[[[10.,float('nan')]]]]]), cross_lidar_mask=torch.tensor([[[[[1.,0.]]]]]), near=torch.ones(1,1), far=torch.full((1,1),80.))
        v=torch.full((1,1,2,1,2),.2,requires_grad=True)
        value=loss(None,{'context':ctx},SimpleNamespace(visual_disparity=v),0)
        value.backward()
        self.assertAlmostEqual(value.item(),.0095,places=6)
        self.assertTrue((v.grad[...,0]>0).all())
        self.assertEqual(v.grad[...,1].abs().sum().item(),0)
        ctx['cross_lidar_mask'].zero_(); v.grad=None
        value=loss(None,{'context':ctx},SimpleNamespace(visual_disparity=v),0); value.backward()
        self.assertEqual(value.item(),0); self.assertEqual(v.grad.abs().sum().item(),0)

    def test_flip_labels(self):
        x=torch.tensor([[[[1.,2.]]]])
        views=dict(image=x,extrinsics=torch.eye(4)[None],cross_lidar_depth=x,cross_lidar_mask=x)
        out=reflect_views(views)
        self.assertTrue(torch.equal(out['cross_lidar_depth'],x.flip(-1)))
        self.assertTrue(torch.equal(out['cross_lidar_mask'],x.flip(-1)))
