"""Check RGB iteration with samples containing camera data only."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class NuScenesRgbOnlyTests(unittest.TestCase):
    def test_train_and_test_without_pointcloud_records(self):
        path = Path(__file__).resolve().parents[1] / 'src/dataset/dataset_nuscenes.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DatasetNuScenes')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__iter__')
        namespace = {
            'torch': torch,
            'apply_crop_shim': lambda example, shape: example,
            'apply_augmentation_shim': lambda example: example,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        camera_ids = {'a': 1., 'b': 2., 'c': 3.}

        def get(table, token):
            self.assertEqual(table, 'sample')
            return {'data': {'CAM_FRONT': token}}

        for stage in ('train', 'test'):
            masked_samples = []

            def mask(**kwargs):
                masked_samples.append(kwargs['sample_token'])
                return torch.zeros(1, 2, 3)

            dataset = SimpleNamespace(
                stage=stage,
                items=[dict(scene_name='scene', camera_name='CAM_FRONT',
                            context_sample_tokens=['a', 'b'], target_sample_tokens=['c'])],
                cfg=SimpleNamespace(num_context_views=2, num_target_views=1,
                                    skip_bad_shape=True, camera_names=['CAM_FRONT'],
                                    image_shape=(2, 3), augment=False),
                nusc=SimpleNamespace(get=get),
                _load_camera=lambda token: (torch.full((3, 2, 3), camera_ids[token]), torch.eye(4), torch.eye(3)),
                get_bound=lambda name, count: torch.full((count,), 1. if name == 'near' else 80.),
                _project_dynamic_mask_to_camera=mask,
            )
            example, = list(namespace['__iter__'](dataset))
            torch.testing.assert_close(example['context']['image'][:, 0, 0, 0], torch.tensor([1., 2.]))
            torch.testing.assert_close(example['target']['image'], torch.full((1, 3, 2, 3), 3.))
            self.assertEqual(masked_samples, ['a', 'b'])
            self.assertNotIn('dynamic_mask', example['target'])
            self.assertEqual(set(example['context']), {'image', 'extrinsics', 'intrinsics', 'near', 'far', 'index', 'dynamic_mask'})


if __name__ == '__main__':
    unittest.main()
