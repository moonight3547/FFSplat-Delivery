"""Meaningful convention, score-boundary and input-isolation checks (CPU only)."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.select_views import _write_synthetic_scene, select
from research.visualize_cameras import (estimate_frame, geometry_diagnostics, lpips_status,
                                       nearest_context_edges, policy_names, validate_case,
                                       work_snapshot)


class CameraVisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _write_synthetic_scene(self.root)
        self.case = work_snapshot(self.root, "synthetic", None)["cases"][0]

    def tearDown(self):
        self.temp.cleanup()

    def test_metric_boundaries_unknown_is_not_zero(self):
        self.assertIsNone(lpips_status({})["points"])
        self.assertFalse(lpips_status({})["zero"])
        self.assertEqual(lpips_status({"lpips": 0})["points"], 40)
        self.assertFalse(lpips_status({"lpips": .2499})["zero"])
        self.assertTrue(lpips_status({"lpips": .25})["zero"])
        self.assertTrue(lpips_status({"lpips": .2993})["zero"])

    def test_frame_rotates_with_scene_and_preserves_distances(self):
        views = self.case["views"]
        origin, basis, info = estimate_frame(views)
        self.assertEqual(info["method"], "mean_camera_up")
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-10)
        self.assertGreater(np.linalg.det(basis), .9999)
        arbitrary_rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        rotated = copy.deepcopy(views)
        for v in rotated:
            v["center"] = (arbitrary_rotation @ np.asarray(v["center"])).tolist()
            v["R_c2w"] = (arbitrary_rotation @ np.asarray(v["R_c2w"])).tolist()
        _, b2, _ = estimate_frame(rotated)
        np.testing.assert_allclose(b2[:, 2], arbitrary_rotation @ basis[:, 2], atol=1e-10)
        raw = np.asarray([v["center"] for v in views])
        projected = (raw - origin) @ basis
        np.testing.assert_allclose(np.linalg.norm(raw[1:] - raw[0], axis=1),
                                   np.linalg.norm(projected[1:] - projected[0], axis=1))

    def test_selector_matches_prepared_work(self):
        policies, errors = policy_names(self.case)
        self.assertFalse(errors)
        for policy in ("uniform", "fps", "coverage"):
            self.assertEqual(policies[policy], select(self.root, policy)["context"])

    def test_test_pose_cannot_affect_selection(self):
        before = policy_names(self.case)[0]
        changed = copy.deepcopy(self.case)
        test = set(changed["split"]["test"])
        for view in changed["views"]:
            if view["name"] in test:
                view["center"] = [1e7, -2e7, 3e7]
        self.assertEqual(before, policy_names(changed)[0])

    def test_reject_test_context_and_bad_rotation(self):
        validate_case(self.case)
        bad = copy.deepcopy(self.case)
        bad["split"]["context"][0] = bad["split"]["test"][0]
        with self.assertRaisesRegex(ValueError, "training"):
            validate_case(bad)
        bad = copy.deepcopy(self.case)
        bad["views"][0]["R_c2w"][0][0] += .2
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            validate_case(bad)

    def test_network_neighbor_graph_excludes_self(self):
        centers = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0], [10, 0, 0]])
        edges = nearest_context_edges(centers)
        self.assertEqual(len(edges), 8)
        self.assertNotIn((0, 0), edges)
        self.assertEqual(edges[:2], [(0, 1), (0, 2)])

    def test_diagnostics_independent_of_global_scale(self):
        names = self.case["split"]["context"]
        before = geometry_diagnostics(self.case, names)
        scaled = copy.deepcopy(self.case)
        for v in scaled["views"]:
            v["center"] = (np.asarray(v["center"]) * 123.4).tolist()
        after = geometry_diagnostics(scaled, names)
        for key in ("max_train_to_context_distance_normalized", "min_context_pair_distance_normalized"):
            self.assertAlmostEqual(before[key], after[key])
        self.assertEqual(before["pivot_name"], names[8])


if __name__ == "__main__":
    unittest.main()
