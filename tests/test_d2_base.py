# SPDX-License-Identifier: Apache-2.0
"""D2-BASE structural checks using synthetic metadata and observations."""

import unittest

import numpy as np

from neurophysical_hvac import D2Policy
from neurophysical_hvac.d2_base import ARM, forward, initial_weights


def synthetic_policy():
    fields = ["VAV_1_zone_air_temperature_sensor", "hod_sin_000", "hod_cos_000",
              "dow_sin_000", "dow_cos_000", "comfort_mode_now", "comfort_mode_soon"]
    metadata = {
        "observation_field_names": fields,
        "n_continuous_action": 7,
        "n_discrete_action": 2,
        "ahu_zone_assignments": {"ahu_1": ["room_1"]},
        "temp_indices": [2],
        "pressure_indices": [3, 6],
        "recirculation_indices": [4],
        "water_indices": [5],
        "zone_action_indices": {"VAV_1": {
            "supply_air_damper_percentage_command": 0,
            "reheat_valve_percentage_command": 1,
        }},
        "observation_normalization_metadata": {
            "zone_air_temperature_sensor": {"sample_mean": 295., "sample_std": 1.}
        },
        "comfort_temp_window": [294., 296.],
        "eco_temp_window": [293., 297.],
        "morning_start_hour": 8,
        "evening_start_hour": 18,
    }
    action_spec = {"continuous_action": np.zeros(7), "discrete_action": np.zeros(2)}
    return D2Policy(metadata, action_spec)


class D2BaseTests(unittest.TestCase):
    def test_release_contains_only_base_arm(self):
        self.assertEqual(ARM, "BASE")
        self.assertEqual(D2Policy.arm, "BASE")

    def test_forward_has_expected_policy_head_shapes(self):
        weights = initial_weights(seed=17)
        zone = np.zeros((1, 2, 17))
        ahu = np.zeros((1, 1, 12))
        outputs, _ = forward(weights, zone, ahu, np.zeros((1, 3)),
                             [np.array([0, 1])], np.array([0, 0]))
        self.assertEqual([x.shape for x in outputs], [(1, 2, 2), (1, 1, 4), (1, 3)])
        self.assertTrue(all(np.isfinite(x).all() for x in outputs))

    def test_weights_are_explicit_and_shape_checked(self):
        policy = synthetic_policy()
        observation = np.zeros(len(policy.fields))
        with self.assertRaisesRegex(RuntimeError, "explicit model weights"):
            policy.get_action(observation)
        weights = initial_weights(seed=22)
        policy.set_weights(weights)
        policy.set_weights({key: value.copy() for key, value in weights.items()})
        with self.assertRaisesRegex(ValueError, "Invalid D2 weights"):
            policy.set_weights({"wrong": np.zeros(1)})

    def test_untrained_structural_action_obeys_shapes_and_projection(self):
        policy = synthetic_policy()
        policy.set_weights(initial_weights(seed=23))
        action = policy.get_action(np.zeros(len(policy.fields)))
        self.assertEqual(action["continuous_action"].shape, (7,))
        self.assertEqual(action["discrete_action"].shape, (2,))
        self.assertTrue(np.isfinite(action["continuous_action"]).all())
        self.assertTrue(np.isin(action["continuous_action"], [-1., 1.]).all())
        np.testing.assert_array_equal(action["discrete_action"], [0, 0])

    def test_action_projection_respects_heating_cooling_and_off_gates(self):
        policy = synthetic_policy()
        zone_proposal = np.array([[0.4, 0.6]])
        ahu_proposal = np.array([[0.2, -0.3, 0.5, 0.8]])
        water_proposal = np.array([0.7, -0.2, 0.9])

        cooling = policy.project((zone_proposal, ahu_proposal, water_proposal),
                                 np.array([-1.]))
        np.testing.assert_array_equal(cooling["discrete_action"], [1, 1])
        np.testing.assert_allclose(cooling["continuous_action"],
                                   [0.4, -1., 0.2, -0.3, 0.5, -1., -0.2])

        heating = policy.project((zone_proposal, ahu_proposal, water_proposal),
                                 np.array([1.]))
        np.testing.assert_array_equal(heating["discrete_action"], [1, 1])
        np.testing.assert_allclose(heating["continuous_action"],
                                   [0.4, 0.6, 0.2, -0.3, 0.5, 0.7, -0.2])

        off_ahu = policy.project((zone_proposal, np.array([[0.2, -0.3, 0.5, -0.8]]),
                                  water_proposal), np.array([1.]))
        np.testing.assert_array_equal(off_ahu["discrete_action"], [0, 0])
        np.testing.assert_allclose(off_ahu["continuous_action"], [0.4, 0.6, -1., -1., 1., -1., -1.])

    def test_comfort_now_is_authoritative_over_soon(self):
        policy = synthetic_policy()
        observation = np.zeros(len(policy.fields))
        observation[policy.index["comfort_mode_now"]] = 0.
        observation[policy.index["comfort_mode_soon"]] = 1.
        zone_eco, *_ = policy.features(observation)
        self.assertAlmostEqual(zone_eco[0, 15], (293. - 295.) / 10.)
        self.assertEqual(zone_eco[0, 9], 0.)
        self.assertEqual(zone_eco[0, 10], 1.)

        observation[policy.index["comfort_mode_now"]] = 1.
        observation[policy.index["comfort_mode_soon"]] = 0.
        policy.reset()
        zone_comfort, *_ = policy.features(observation)
        self.assertAlmostEqual(zone_comfort[0, 15], (294. - 295.) / 10.)
        self.assertEqual(zone_comfort[0, 9], 1.)
        self.assertEqual(zone_comfort[0, 10], 0.)

    def test_clock_fallback_switches_at_causal_five_minute_boundary(self):
        policy = synthetic_policy()
        policy.fields = ["VAV_1_zone_air_temperature_sensor", "hod_sin_000", "hod_cos_000",
                         "dow_sin_000", "dow_cos_000"]
        policy.index = {field: index for index, field in enumerate(policy.fields)}
        policy.outside_obs = [None]

        def observation(hour, minute):
            phase = 2 * np.pi * (hour + minute / 60) / 24
            return np.array([0., np.sin(phase), np.cos(phase), 0., 1.])

        before, *_ = policy.features(observation(7, 55))
        policy.reset()
        after, *_ = policy.features(observation(8, 0))
        self.assertAlmostEqual(before[0, 15], (293. - 295.) / 10.)
        self.assertAlmostEqual(after[0, 15], (294. - 295.) / 10.)

    def test_executed_action_updates_history_and_reset_clears_it(self):
        policy = synthetic_policy()
        observation = np.zeros(len(policy.fields))
        policy.features(observation)
        action = {"continuous_action": np.array([0.25, -0.5, 0., 0., 1., -1., 0.]),
                  "discrete_action": np.array([1, 0])}
        policy.remember(action)
        observation[0] = 0.5
        zone, *_ = policy.features(observation)
        self.assertAlmostEqual(zone[0, 4], 0.25)
        self.assertAlmostEqual(zone[0, 6], 0.25)
        self.assertAlmostEqual(zone[0, 7], -0.5)

        policy.reset()
        zone, *_ = policy.features(observation)
        self.assertEqual(zone[0, 4], 0.)
        self.assertEqual(zone[0, 6], -1.)
        self.assertEqual(zone[0, 7], -1.)


if __name__ == "__main__":
    unittest.main()
