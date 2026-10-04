"""CPU geometry regressions without importing the optional CUDA/dataset stack."""
import ast
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name):
    tree = ast.parse((ROOT / path).read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
        ast.alias(name='annotations')], level=0), function], type_ignores=[])
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(module), path, 'exec'), namespace)
    return namespace[name]


projection = load_function('src/model/decoder/cuda_splatting.py', 'get_projection_matrix')
crop = load_function('src/dataset/shims/crop_shim.py', 'center_crop')
patch = load_function('src/dataset/shims/patch_shim.py', 'apply_patch_shim_to_views')


class CameraGeometryTests(unittest.TestCase):
    def test_off_center_projection_matches_pinhole(self):
        k = torch.tensor([[[.79, 0., .516], [0., 1.44, .547], [0., 0., 1.]],
                          [[1.1, 0., .43], [0., .9, .48], [0., 0., 1.]]])
        near, far = torch.tensor([1., 2.]), torch.tensor([80., 100.])
        fov_x, fov_y = 2 * (.5 / k[:, 0, 0]).atan(), 2 * (.5 / k[:, 1, 1]).atan()
        p = projection(near, far, fov_x, fov_y, k)
        points = torch.tensor([[0., 0., 5., 1.], [2., -1., 10., 1.],
                               [-3., 2., 20., 1.]])
        clip = torch.einsum('bij,nj->bni', p, points)
        actual = (clip[..., :2] / clip[..., 3:] + 1) / 2
        expected = torch.einsum('bij,nj->bni', k, points[:, :3])
        expected = expected[..., :2] / expected[..., 2:]
        torch.testing.assert_close(actual, expected)
        # Near/far mapping must remain unchanged.
        z_points = torch.stack([torch.zeros_like(near), torch.zeros_like(near),
                                near, torch.ones_like(near)], dim=-1)
        clip_near = torch.einsum('bij,bj->bi', p, z_points)
        torch.testing.assert_close(clip_near[:, 2], torch.zeros_like(near))
        z_points[:, 2] = far
        clip_far = torch.einsum('bij,bj->bi', p, z_points)
        torch.testing.assert_close(clip_far[:, 2] / clip_far[:, 3], torch.ones_like(far))

    def test_centered_projection_preserves_legacy_behavior(self):
        k = torch.tensor([[[.8, 0., .5], [0., 1.4, .5], [0., 0., 1.]]])
        near, far = torch.tensor([1.]), torch.tensor([80.])
        fx, fy = 2 * (.5 / k[:, 0, 0]).atan(), 2 * (.5 / k[:, 1, 1]).atan()
        torch.testing.assert_close(projection(near, far, fx, fy, k),
                                   projection(near, far, fx, fy))

    def test_crop_preserves_pixel_projection(self):
        # Includes odd crop margins and a principal point away from center.
        image = torch.zeros(2, 3, 181, 323)
        k = torch.tensor([[[.8, 0., .516], [0., 1.4, .547], [0., 0., 1.]]]).repeat(2, 1, 1)
        original = k.clone()
        result, updated = crop(image, k, (176, 320))
        xy = torch.tensor([.1, -.2, 1.])
        old_pixel = (original @ xy)[..., :2] * torch.tensor([323., 181.])
        new_pixel = (updated @ xy)[..., :2] * torch.tensor([320., 176.])
        torch.testing.assert_close(new_pixel, old_pixel - torch.tensor([1., 2.]))
        torch.testing.assert_close(k, original)
        self.assertEqual(result.shape[-2:], (176, 320))

    def test_patch_crop_preserves_pixel_projection(self):
        image = torch.zeros(1, 2, 3, 182, 326)
        k = torch.tensor([[[[.8, 0., .516], [0., 1.4, .547], [0., 0., 1.]]]]).repeat(1, 2, 1, 1)
        result = patch({'image': image, 'intrinsics': k}, 16)
        xy = torch.tensor([.1, -.2, 1.])
        old_pixel = (k @ xy)[..., :2] * torch.tensor([326., 182.])
        new_pixel = (result['intrinsics'] @ xy)[..., :2] * torch.tensor([320., 176.])
        torch.testing.assert_close(new_pixel, old_pixel - 3)


if __name__ == '__main__':
    unittest.main()
