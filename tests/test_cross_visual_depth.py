import unittest
from types import SimpleNamespace
import torch
from src.loss.loss_cross_visual_depth import LossCrossVisualDepth, LossCrossVisualDepthCfg, LossCrossVisualDepthCfgWrapper

class VisualLossTest(unittest.TestCase):
    def test_masks_and_gradients(self):
        v=torch.tensor([.1,.1,.1,.1,.102,.102,.102,.102]).reshape(1,2,1,2,2).requires_grad_()
        mask=torch.zeros_like(v); mask[...,0,0]=1
        ctx=dict(near=torch.ones(1,2),far=torch.full((1,2),80.),intrinsics=torch.eye(3).repeat(1,2,1,1),extrinsics=torch.eye(4).repeat(1,2,1,1),lidar_mask=mask,cross_lidar_mask=torch.zeros_like(mask))
        loss=LossCrossVisualDepth(LossCrossVisualDepthCfgWrapper(LossCrossVisualDepthCfg(samples=5000)))
        value=loss(None,{'context':ctx},SimpleNamespace(visual_disparity=v),0)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertEqual(v.grad[...,0,0].abs().sum(),0)
        # Both directions are geometrically consistent and supervise their target.
        self.assertGreater(v.grad[:,0].abs().sum(),0)
        self.assertGreater(v.grad[:,1].abs().sum(),0)
        ctx['lidar_mask'].fill_(1); v.grad=None
        zero=loss(None,{'context':ctx},SimpleNamespace(visual_disparity=v),0); zero.backward()
        self.assertEqual(zero.item(),0); self.assertEqual(v.grad.abs().sum(),0)

    def test_identical_views(self):
        v=torch.full((1,2,1,3,3),.1,requires_grad=True)
        ctx=dict(near=torch.ones(1,2),far=torch.full((1,2),80.),intrinsics=torch.eye(3).repeat(1,2,1,1),extrinsics=torch.eye(4).repeat(1,2,1,1))
        loss=LossCrossVisualDepth(LossCrossVisualDepthCfgWrapper(LossCrossVisualDepthCfg()))
        self.assertAlmostEqual(loss(None,{'context':ctx},SimpleNamespace(visual_disparity=v),0).item(),0)

    def test_color_rejection_and_matching_gradients(self):
        v=torch.tensor([.1]*4+[.102]*4).reshape(1,2,1,2,2).requires_grad_()
        rgb=torch.zeros(1,2,3,2,2)
        rgb[:,1,:,:,1]=1  # mismatched right column, identical left column
        ctx=dict(image=rgb,near=torch.ones(1,2),far=torch.full((1,2),80.),intrinsics=torch.eye(3).repeat(1,2,1,1),extrinsics=torch.eye(4).repeat(1,2,1,1))
        loss=LossCrossVisualDepth(LossCrossVisualDepthCfgWrapper(LossCrossVisualDepthCfg(color_threshold=.1)))
        result=loss(None,{'context':ctx},SimpleNamespace(visual_disparity=v),0)
        result.backward()
        self.assertGreater(v.grad[:,0,:,:,:1].abs().sum().item(),0)
        self.assertEqual(v.grad[:,:,:,:,1].abs().sum().item(),0)
        self.assertEqual(loss.diagnostics['cross_visual_color_removed'].item(),2)
        rgb[:,1]=1; v.grad=None
        result=loss(None,{'context':ctx},SimpleNamespace(visual_disparity=v),0)
        result.backward()
        self.assertEqual(result.item(),0)
        self.assertEqual(v.grad.abs().sum().item(),0)

    def test_independent_depth_cycle(self):
        from src.loss.loss_cross_visual_depth import geometry_errors
        k=torch.eye(3); transform=torch.eye(4); transform[0,3]=1
        source_uv=torch.tensor([[.5],[.5]])
        uv=torch.tensor([[.4],[.5]])
        args=(torch.tensor([10.]),torch.tensor([10.]),source_uv,k,k,transform,100,100)
        pix,dep=geometry_errors(uv,torch.tensor([10.]),*args)
        self.assertLess(pix.item(),1e-4); self.assertLess(dep.item(),1e-5)
        pix,dep=geometry_errors(uv,torch.tensor([5.]),*args)
        self.assertAlmostEqual(pix.item(),10.,places=4)
        self.assertAlmostEqual(dep.item(),.5,places=5)

    def test_large_depth_disagreement_rejected(self):
        v=torch.tensor([.1]*4+[.2]*4).reshape(1,2,1,2,2).requires_grad_()
        ctx=dict(near=torch.ones(1,2),far=torch.full((1,2),80.),intrinsics=torch.eye(3).repeat(1,2,1,1),extrinsics=torch.eye(4).repeat(1,2,1,1))
        loss=LossCrossVisualDepth(LossCrossVisualDepthCfgWrapper(LossCrossVisualDepthCfg()))
        value=loss(None,{'context':ctx},SimpleNamespace(visual_disparity=v),0); value.backward()
        self.assertEqual(value.item(),0)
        self.assertEqual(v.grad.abs().sum().item(),0)
        self.assertEqual(loss.diagnostics['cross_visual_depth_removed'].item(),4)
