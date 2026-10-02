import unittest

import numpy as np

from src.diagnostics.cross_lidar import classify, rasterize, rates, summarize


class CrossLidarTest(unittest.TestCase):
    def test_translation_zbuffer_and_out_of_view(self):
        camera = np.eye(4)
        camera[0, 3] = 1
        k = np.array([[.5, 0, .5], [0, .5, .5], [0, 0, 1]])
        points = np.array([[1, 1, 100, 1], [0, 0, 0, 0], [5, 10, 5, -2]])
        depth = rasterize(points, camera, k, (10, 10), 1, 80)
        self.assertEqual(np.count_nonzero(depth), 1)
        self.assertEqual(depth[5, 5], 5)

    def test_conflicts_and_new_points_have_distinct_denominators(self):
        own = np.array([[10., 20., 0., 0.]])
        cross = np.array([[10.5, 25., 30., 0.]])
        overlap, conflict, added, _ = classify(own, cross, 1., .05)
        self.assertEqual(overlap.sum(), 2)
        self.assertEqual(conflict.sum(), 1)
        self.assertEqual(added.sum(), 1)
        all_distances = rates(summarize(own, cross, 1., .05)[0])
        self.assertEqual(all_distances['overlap_mae_m'], 2.75)
        self.assertEqual(all_distances['conflict_rate_of_overlap'], .5)
        self.assertEqual(all_distances['new_coverage_of_image'], .25)
        empty = rates(summarize(np.zeros((2, 2)), np.zeros((2, 2)), 1., .05)[0])
        self.assertIsNone(empty['overlap_mae_m'])


if __name__ == '__main__':
    unittest.main()
