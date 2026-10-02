import unittest
import torch
from src.diagnostics.visual_cross_depth import project_samples

class ProjectionTest(unittest.TestCase):
    def test_identity_pixel_centers(self):
        depth=torch.full((4,6),10.)
        k=torch.tensor([[.8,0,.5],[0,.8,.5],[0,0,1.]])
        pose=torch.eye(4); idx=torch.arange(24)
        uv,z=project_samples(depth,k,k,pose,pose,idx)
        expected=torch.stack(((idx%6+.5)/6,(idx//6+.5)/4))
        torch.testing.assert_close(uv,expected)
        torch.testing.assert_close(z,torch.full((24,),10.))

    def test_translation(self):
        depth=torch.full((4,6),10.)
        k=torch.eye(3); a=torch.eye(4); b=torch.eye(4); b[0,3]=1
        uv,z=project_samples(depth,k,k,a,b,torch.tensor([14]))
        torch.testing.assert_close(uv[:,0],torch.tensor([2.5/6-.1,2.5/4]))
        self.assertEqual(z.item(),10.)
