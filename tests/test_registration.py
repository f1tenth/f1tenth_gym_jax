import unittest

from f1tenth_gym_jax.registration import _parse_scenario


class TestRegistration(unittest.TestCase):
    def test_parse_scenario(self):
        scenario = (
            "Spielberg_2_scan_collision_progress+alive_"
            "velocity+steeringangle_10_500_v0"
        )

        self.assertEqual(
            _parse_scenario(scenario),
            (
                "Spielberg",
                2,
                True,
                True,
                "progress+alive",
                ["velocity", "steeringangle"],
                10,
                500,
            ),
        )

    def test_parse_scenario_with_underscored_map_name(self):
        scenario = (
            "Spielberg_blank_1_noscan_nocollision_progress_"
            "acceleration+steeringvelocity_1_v0_v0"
        )

        self.assertEqual(
            _parse_scenario(scenario),
            (
                "Spielberg_blank",
                1,
                False,
                False,
                "progress",
                ["acceleration", "steeringvelocity"],
                1,
                None,
            ),
        )

    def test_parse_scenario_rejects_invalid_fields(self):
        invalid_scenarios = [
            "Spielberg_0_noscan_nocollision_progress_acceleration+steeringvelocity_1_v0_v0",
            "Spielberg_1_depth_nocollision_progress_acceleration+steeringvelocity_1_v0_v0",
            "Spielberg_1_noscan_bump_progress_acceleration+steeringvelocity_1_v0_v0",
            "Spielberg_1_noscan_nocollision_progress_throttle+steeringvelocity_1_v0_v0",
            "Spielberg_1_noscan_nocollision_progress_acceleration+wheel_1_v0_v0",
            "Spielberg_1_noscan_nocollision_score_acceleration+steeringvelocity_1_v0_v0",
            "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_0_v0_v0",
            "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_0_v0",
            "Spielberg_1_noscan_nocollision_progress_acceleration+steeringvelocity_1_v0_v1",
            "Spielberg_1_noscan",
        ]

        for scenario in invalid_scenarios:
            with self.subTest(scenario=scenario):
                with self.assertRaises(ValueError):
                    _parse_scenario(scenario)
