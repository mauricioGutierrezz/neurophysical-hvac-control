# SPDX-License-Identifier: Apache-2.0
"""Structural contracts using synthetic inputs only."""

import unittest

import numpy as np

from neurophysical_hvac import DenseEncoder, HierarchicalEncoder, pool_by_ahu


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(13)
        self.model = HierarchicalEncoder.random(4, 3, 2, seed=7)
        self.zones = self.rng.normal(size=(5, 4))
        self.ahus = self.rng.normal(size=(3, 3))
        self.history = np.array([0.2, -0.1])
        self.mapping = np.array([0, 0, 2, 2, 2])  # AHU 1 is empty.

    def encode(self, **changes):
        inputs = dict(zone_features=self.zones, ahu_features=self.ahus,
                      water_previous=self.history, zone_to_ahu=self.mapping)
        inputs.update(changes)
        return self.model.encode(**inputs)

    def test_pool_has_known_means_maxima_counts_and_zero_empty_group(self):
        actual = pool_by_ahu([[1., -2.], [3., 4.], [8., 0.]], [0, 0, 2], 3)
        expected = [[2., 1., 3., 4., np.log(3)], [0., 0., 0., 0., 0.],
                    [8., 0., 8., 0., np.log(2)]]
        np.testing.assert_allclose(actual, expected)

    def test_zone_reordering_only_reorders_zone_embeddings(self):
        before = self.encode()
        order = np.array([4, 2, 0, 3, 1])
        after = self.encode(zone_features=self.zones[order], zone_to_ahu=self.mapping[order])
        np.testing.assert_allclose(after.zones, before.zones[order], atol=1e-14)
        np.testing.assert_allclose(after.ahus, before.ahus, atol=1e-14)
        np.testing.assert_allclose(after.water, before.water, atol=1e-14)

    def test_ahu_relabeling_preserves_global_context_and_reorders_ahus(self):
        before = self.encode()
        order = np.array([2, 0, 1])
        inverse = np.argsort(order)
        after = self.encode(ahu_features=self.ahus[order], zone_to_ahu=inverse[self.mapping])
        np.testing.assert_allclose(after.zones, before.zones, atol=1e-14)
        np.testing.assert_allclose(after.ahus, before.ahus[order], atol=1e-14)
        np.testing.assert_allclose(after.water, before.water, atol=1e-14)

    def test_same_model_handles_another_building_without_parameter_changes(self):
        first = self.encode()
        second = self.encode(zone_features=np.zeros((1, 4)), ahu_features=np.zeros((1, 3)),
                             water_previous=np.zeros(2), zone_to_ahu=np.array([0]))
        self.assertEqual(first.zones.shape, (5, 8))
        self.assertEqual(first.ahus.shape, (3, 8))
        self.assertEqual(second.zones.shape, (1, 8))
        self.assertEqual(second.ahus.shape, (1, 8))
        self.assertEqual(second.water.shape, (6,))
        self.assertTrue(np.isfinite(second.water).all())

    def test_bad_mapping_dimensions_and_nonfinite_inputs_are_rejected(self):
        cases = [dict(zone_to_ahu=[0, 0, 2, 2, 3]), dict(zone_to_ahu=[0, 0, -1, 2, 2]),
                 dict(zone_to_ahu=[0., 0., 2., 2., 2.]), dict(zone_to_ahu=[0]),
                 dict(zone_features=np.zeros((5, 3))), dict(zone_features=np.empty((0, 4))),
                 dict(ahu_features=np.empty((0, 3))), dict(water_previous=np.zeros(3)),
                 dict(zone_features=np.full((5, 4), np.nan)),
                 dict(ahu_features=np.full((3, 3), np.inf)),
                 dict(water_previous=np.array([0., np.inf]))]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.encode(**case)

    def test_initialization_does_not_change_numpy_global_random_state(self):
        np.random.seed(21)
        expected = np.random.normal(size=4)
        np.random.seed(21)
        other = HierarchicalEncoder.random(4, 3, 2, seed=7)
        np.testing.assert_array_equal(np.random.normal(size=4), expected)
        np.testing.assert_array_equal(other.zone_encoder.weight, self.model.zone_encoder.weight)

    def test_encoding_is_stateless_and_does_not_modify_caller_inputs(self):
        zone_copy, ahu_copy, history_copy = self.zones.copy(), self.ahus.copy(), self.history.copy()
        first = self.encode()
        self.encode(zone_features=self.zones + 1, water_previous=np.ones(2))
        again = self.encode()
        np.testing.assert_array_equal(first.water, again.water)
        np.testing.assert_array_equal(self.zones, zone_copy)
        np.testing.assert_array_equal(self.ahus, ahu_copy)
        np.testing.assert_array_equal(self.history, history_copy)

    def test_parameter_validation_and_ownership(self):
        weight, bias = np.ones((2, 3)), np.zeros(3)
        encoder = DenseEncoder(weight, bias)
        weight[:] = 0
        bias[:] = 1
        np.testing.assert_array_equal(encoder.weight, np.ones((2, 3)))
        np.testing.assert_array_equal(encoder.bias, np.zeros(3))
        with self.assertRaises(ValueError):
            encoder.weight[0, 0] = 10
        for bad_weight, bad_bias in [(np.zeros((2, 3)), np.zeros(2)),
                                     (np.full((2, 3), np.nan), np.zeros(3))]:
            with self.assertRaises(ValueError):
                DenseEncoder(bad_weight, bad_bias)
        for scale in [0, -1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                HierarchicalEncoder.random(4, 3, 2, count_scale=scale)
        for size in [0, -1, 2.5, True]:
            with self.assertRaises(ValueError):
                HierarchicalEncoder.random(size, 3, 2)

    def test_finite_values_that_overflow_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "numerical range"):
            pool_by_ahu([[1.]], [0], 1, count_scale=1e-320)
        with self.assertRaisesRegex(ValueError, "numerical range"):
            pool_by_ahu([[1e308], [1e308]], [0, 0], 1)
        encoder = DenseEncoder(np.array([[1e308], [1e308]]), np.zeros(1))
        with self.assertRaisesRegex(ValueError, "numerical range"):
            encoder(np.array([[2., -2.]]))


if __name__ == "__main__":
    unittest.main()
