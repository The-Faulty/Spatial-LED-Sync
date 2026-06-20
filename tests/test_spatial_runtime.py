from __future__ import annotations

import unittest
import unittest.mock

import numpy as np

from config import EngineConfig
from event_detector import LightEvent
from spatial_config import parse_spatial_config, spatial_config_to_dict, default_spatial_dict, validate_spatial_config
from spatial_renderer import VectorizedSpatialRenderer
from spatial_topology import SpatialRoomTopology
from wled_output import WLEDOutput


def spatial_config() -> dict:
    spatial = default_spatial_dict(12)
    spatial["enabled"] = True
    spatial["room"] = {"width": 4.0, "depth": 3.0, "height": 2.4, "unit": "m"}
    spatial["tv"] = {
        "wall": "front",
        "center_u": 2.0,
        "center_v": 1.2,
        "width": 1.4,
        "height": 0.8,
        "mirror_horizontal": False,
    }
    spatial["devices"] = [
        {"id": "left", "name": "Left", "ip": "192.0.2.10", "led_count": 8, "segment_id": None, "enabled": True},
        {"id": "right", "name": "Right", "ip": "192.0.2.11", "led_count": 8, "segment_id": 1, "enabled": True},
    ]
    spatial["strips"] = [
        {
            "id": "front",
            "name": "Front",
            "wall": "front",
            "start_u": 0.5,
            "start_v": 1.9,
            "end_u": 3.5,
            "end_v": 1.9,
            "led_count": 4,
            "direction": "forward",
            "device_id": "left",
            "device_start": 0,
            "sync_mode": "blend",
            "blend": 0.5,
        },
        {
            "id": "right",
            "name": "Right",
            "wall": "right",
            "start_u": 0.0,
            "start_v": 1.5,
            "end_u": 3.0,
            "end_v": 1.5,
            "led_count": 4,
            "direction": "forward",
            "device_id": "right",
            "device_start": 2,
            "sync_mode": "spatial",
            "blend": 0.5,
        },
    ]
    return spatial


class FakeResponse:
    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict, float]] = []

    def post(self, url: str, json: dict, timeout: float) -> FakeResponse:
        self.posts.append((url, json, timeout))
        return FakeResponse()

    def close(self) -> None:
        return None


class SpatialRuntimeTests(unittest.TestCase):
    def test_spatial_validation_catches_device_overlap(self) -> None:
        spatial = spatial_config()
        spatial["strips"][1]["device_id"] = "left"
        spatial["strips"][1]["device_start"] = 2
        errors = validate_spatial_config(spatial, 12)
        self.assertTrue(any("overlap" in error for error in errors))

    def test_topology_expands_led_coordinates_and_routes(self) -> None:
        config = EngineConfig(total_leds=8, spatial=spatial_config())
        topology = SpatialRoomTopology(config)
        self.assertEqual(topology.total, 8)
        self.assertEqual(topology.positions.shape, (8, 3))
        self.assertEqual(topology.wall_for_led(0), "front")
        routes = topology.device_routes()
        self.assertEqual(len(routes), 2)
        self.assertEqual(routes[0].global_indices.tolist(), [0, 1, 2, 3])
        self.assertEqual(routes[1].device_indices.tolist(), [2, 3, 4, 5])

    def test_left_and_right_wall_coordinates_match_room_sides(self) -> None:
        config = EngineConfig(total_leds=8, spatial=spatial_config())
        topology = SpatialRoomTopology(config)
        room = topology.spatial.room
        left = topology.wall_point("left", 0.0, 1.0)
        right = topology.wall_point("right", 0.0, 1.0)
        self.assertAlmostEqual(float(left[0]), room.width)
        self.assertAlmostEqual(float(right[0]), 0.0)

    def test_vectorized_renderer_produces_led_buffer(self) -> None:
        config = EngineConfig(total_leds=8, spatial=spatial_config(), brightness=1.0, color_smoothing=0.0)
        topology = SpatialRoomTopology(config)
        renderer = VectorizedSpatialRenderer(config, topology)
        renderer.add_events([LightEvent(edge="top", intensity=1.0, color=(255, 80, 20), kind="explosion")])
        leds = renderer.step(0.1)
        self.assertEqual(leds.shape, (8, 3))
        self.assertGreater(int(leds.max()), 0)

    def test_tv_role_sampling_uses_explicit_side(self) -> None:
        spatial = spatial_config()
        spatial["strips"] = [
            dict(spatial["strips"][0], id="top", tv_role="top", sync_mode="tv_image", led_count=2, device_start=0),
            dict(spatial["strips"][0], id="bottom", tv_role="bottom", sync_mode="tv_image", led_count=2, device_start=2),
        ]
        config = EngineConfig(total_leds=4, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[:2, :] = (0, 0, 255)
        frame[-2:, :] = (0, 255, 0)
        colors = topology.tv_sample_colors(frame)
        self.assertGreater(float(colors[0, 0]), 0.9)
        self.assertGreater(float(colors[2, 1]), 0.9)

    def test_same_tv_side_strips_sample_independently(self) -> None:
        spatial = spatial_config()
        top_strip = dict(
            spatial["strips"][0],
            start_u=1.3,
            end_u=2.7,
            start_v=1.65,
            end_v=1.65,
            led_count=3,
            direction="forward",
            tv_role="top",
            sync_mode="tv_image",
        )
        spatial["strips"] = [
            dict(top_strip, id="top-a", device_id="left", device_start=0),
            dict(top_strip, id="top-b", device_id="right", device_start=0),
        ]
        config = EngineConfig(total_leds=6, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((6, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        colors = topology.tv_sample_colors(frame)
        self.assertLess(float(colors[0, 0]), 0.05)
        self.assertGreater(float(colors[2, 0]), 0.95)
        self.assertLess(float(colors[3, 0]), 0.05)
        self.assertGreater(float(colors[5, 0]), 0.95)

    def test_same_side_strip_direction_reverses_sampling(self) -> None:
        spatial = spatial_config()
        bottom_strip = dict(
            spatial["strips"][0],
            start_u=1.3,
            end_u=2.7,
            start_v=0.75,
            end_v=0.75,
            led_count=3,
            tv_role="bottom",
            sync_mode="tv_image",
        )
        spatial["strips"] = [
            dict(bottom_strip, id="bottom-forward", direction="forward", device_id="left", device_start=0),
            dict(bottom_strip, id="bottom-reverse", direction="reverse", device_id="right", device_start=0),
        ]
        config = EngineConfig(total_leds=6, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((6, 5, 3), dtype=np.uint8)
        frame[-1, :, 2] = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        colors = topology.tv_sample_colors(frame)
        self.assertLess(float(colors[0, 0]), 0.05)
        self.assertGreater(float(colors[2, 0]), 0.95)
        self.assertGreater(float(colors[3, 0]), 0.95)
        self.assertLess(float(colors[5, 0]), 0.05)

    def test_mirrored_tv_sampling_matches_mirrored_preview_orientation(self) -> None:
        spatial = spatial_config()
        spatial["tv"]["mirror_horizontal"] = True
        spatial["strips"] = [
            dict(
                spatial["strips"][0],
                id="top",
                start_u=1.3,
                end_u=2.7,
                start_v=1.65,
                end_v=1.65,
                led_count=3,
                direction="forward",
                tv_role="top",
                sync_mode="tv_image",
                device_start=0,
            ),
            dict(
                spatial["strips"][0],
                id="bottom",
                start_u=1.3,
                end_u=2.7,
                start_v=0.75,
                end_v=0.75,
                led_count=3,
                direction="forward",
                tv_role="bottom",
                sync_mode="tv_image",
                device_start=3,
            ),
        ]
        config = EngineConfig(total_leds=6, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((6, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        frame[-1, :, 2] = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        colors = topology.tv_sample_colors(frame)
        self.assertGreater(float(colors[0, 0]), 0.95)
        self.assertLess(float(colors[2, 0]), 0.05)
        self.assertGreater(float(colors[3, 0]), 0.95)
        self.assertLess(float(colors[5, 0]), 0.05)

    def test_left_and_right_tv_roles_sample_vertical_edges(self) -> None:
        spatial = spatial_config()
        side_strip = dict(
            spatial["strips"][0],
            start_u=1.3,
            end_u=1.3,
            start_v=0.8,
            end_v=1.6,
            led_count=3,
            direction="forward",
            sync_mode="tv_image",
        )
        spatial["strips"] = [
            dict(side_strip, id="left-side", tv_role="left", device_id="left", device_start=0),
            dict(side_strip, id="right-side", tv_role="right", device_id="right", device_start=0),
        ]
        config = EngineConfig(total_leds=6, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((5, 6, 3), dtype=np.uint8)
        frame[:, 0, 1] = np.array([255, 192, 128, 64, 0], dtype=np.uint8)
        frame[:, -1, 2] = np.array([255, 192, 128, 64, 0], dtype=np.uint8)
        colors = topology.tv_sample_colors(frame)
        self.assertLess(float(colors[0, 1]), 0.05)
        self.assertGreater(float(colors[2, 1]), 0.95)
        self.assertLess(float(colors[3, 0]), 0.05)
        self.assertGreater(float(colors[5, 0]), 0.95)

    def test_out_of_bounds_tv_side_strip_falls_back_to_strip_position(self) -> None:
        spatial = spatial_config()
        spatial["strips"] = [
            dict(
                spatial["strips"][0],
                id="top-offset",
                start_u=3.0,
                end_u=3.8,
                start_v=2.2,
                end_v=2.2,
                led_count=3,
                tv_role="top",
                sync_mode="tv_image",
                device_start=0,
            )
        ]
        config = EngineConfig(total_leds=3, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        colors = topology.tv_sample_colors(frame)
        self.assertLess(float(colors[0, 0]), 0.05)
        self.assertGreater(float(colors[2, 0]), 0.95)

    def test_extension_inherits_parent_tv_role(self) -> None:
        spatial = spatial_config()
        spatial["strips"][0]["tv_role"] = "left"
        spatial["strips"][1]["extends_strip_id"] = spatial["strips"][0]["id"]
        config = EngineConfig(total_leds=8, spatial=spatial)
        topology = SpatialRoomTopology(config)
        self.assertEqual(topology.effective_tv_role_for_strip(spatial["strips"][1]["id"]), "left")

    def test_tv_image_extension_samples_parent_side_independently(self) -> None:
        spatial = spatial_config()
        parent = dict(
            spatial["strips"][0],
            id="parent",
            start_u=1.3,
            end_u=2.7,
            led_count=3,
            tv_role="top",
            sync_mode="tv_image",
            device_id="left",
            device_start=0,
        )
        extension = dict(
            parent,
            id="extension",
            tv_role="none",
            extends_strip_id="parent",
            extension_mode="edge_reach",
            extension_strength=1.0,
            extension_softness=0.0,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=6, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
        colors = topology.tv_sample_colors(frame)
        self.assertLess(float(colors[0, 0]), 0.05)
        self.assertGreater(float(colors[2, 0]), 0.95)
        self.assertLess(float(colors[3, 0]), 0.05)
        self.assertGreater(float(colors[5, 0]), 0.95)

    def test_extension_defaults_are_added_to_old_configs(self) -> None:
        spatial = spatial_config()
        spatial["tv"].pop("mirror_horizontal", None)
        for strip in spatial["strips"]:
            strip.pop("extension_mode", None)
            strip.pop("extension_strength", None)
            strip.pop("extension_softness", None)
        parsed = parse_spatial_config(spatial, 8)
        saved = spatial_config_to_dict(parsed)
        self.assertTrue(saved["tv"]["mirror_horizontal"])
        self.assertEqual(saved["strips"][0]["extension_mode"], "soft_spill")
        self.assertEqual(saved["strips"][0]["extension_strength"], 0.45)
        self.assertEqual(saved["strips"][0]["extension_softness"], 0.65)

    def test_soft_spill_extension_is_lower_strength_than_parent(self) -> None:
        spatial = spatial_config()
        parent = dict(
            spatial["strips"][0],
            id="parent",
            start_u=1.3,
            end_u=2.7,
            led_count=3,
            tv_role="top",
            sync_mode="tv_image",
            device_id="left",
            device_start=0,
        )
        extension = dict(
            parent,
            id="soft-extension",
            tv_role="none",
            extends_strip_id="parent",
            sync_mode="tv_image",
            extension_mode="soft_spill",
            extension_strength=0.45,
            extension_softness=0.65,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=6, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = 255
        colors = topology.tv_sample_colors(frame)
        self.assertGreater(float(colors[:3, 0].max()), 0.95)
        self.assertLess(float(colors[3:, 0].max()), 0.50)
        self.assertGreater(float(colors[3:, 0].max()), 0.30)

    def test_edge_reach_extension_gates_by_distance_from_tv_edge(self) -> None:
        spatial = spatial_config()
        parent = dict(
            spatial["strips"][0],
            id="parent",
            start_u=1.3,
            end_u=2.7,
            start_v=0.8,
            end_v=0.8,
            led_count=3,
            tv_role="bottom",
            sync_mode="tv_image",
            device_id="left",
            device_start=0,
        )
        extension = dict(
            parent,
            id="edge-reach-extension",
            tv_role="none",
            extends_strip_id="parent",
            start_u=1.3,
            end_u=1.3,
            start_v=0.8,
            end_v=0.1,
            sync_mode="tv_image",
            extension_mode="edge_reach",
            extension_strength=1.0,
            extension_softness=0.65,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=6, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[-1, :, 2] = 128
        colors = topology.tv_sample_colors(frame)
        near = float(colors[3, 0])
        far = float(colors[5, 0])
        self.assertGreater(near, 0.45)
        self.assertLess(far, near * 0.25)

    def test_spatial_extension_inherits_role_but_does_not_sample_tv_image(self) -> None:
        spatial = spatial_config()
        parent = dict(
            spatial["strips"][0],
            id="parent",
            start_u=1.3,
            end_u=2.7,
            led_count=3,
            tv_role="top",
            sync_mode="tv_image",
            device_id="left",
            device_start=0,
        )
        extension = dict(
            parent,
            id="effects-extension",
            tv_role="none",
            extends_strip_id="parent",
            sync_mode="spatial",
            extension_mode="effects_only",
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=6, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = 255
        colors = topology.tv_sample_colors(frame)
        self.assertEqual(topology.effective_tv_role_for_strip("effects-extension"), "top")
        self.assertGreater(float(colors[:3, 0].max()), 0.95)
        self.assertTrue(np.allclose(colors[3:], 0.0))

    def test_blend_extension_receives_inherited_tv_side(self) -> None:
        spatial = spatial_config()
        parent = dict(
            spatial["strips"][0],
            id="parent",
            start_u=1.3,
            end_u=2.7,
            led_count=3,
            tv_role="top",
            sync_mode="tv_image",
            device_id="left",
            device_start=0,
        )
        extension = dict(
            parent,
            id="blend-extension",
            tv_role="none",
            extends_strip_id="parent",
            sync_mode="blend",
            blend=0.5,
            extension_mode="soft_spill",
            extension_strength=0.45,
            extension_softness=0.65,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=6, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0)
        topology = SpatialRoomTopology(config)
        renderer = VectorizedSpatialRenderer(config, topology)
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = 255
        renderer.set_tv_frame(frame)
        leds = renderer.step(0.1)
        self.assertGreater(int(leds[:3, 0].max()), 240)
        self.assertGreater(int(leds[3:, 0].max()), 40)
        self.assertLess(int(leds[3:, 0].max()), 120)

    def test_effects_only_extension_with_tv_image_sync_receives_no_tv_color(self) -> None:
        spatial = spatial_config()
        parent = dict(
            spatial["strips"][0],
            id="parent",
            start_u=1.3,
            end_u=2.7,
            led_count=3,
            tv_role="top",
            sync_mode="tv_image",
            device_id="left",
            device_start=0,
        )
        extension = dict(
            parent,
            id="effects-only-extension",
            tv_role="none",
            extends_strip_id="parent",
            sync_mode="tv_image",
            extension_mode="effects_only",
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=6, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = 255
        colors = topology.tv_sample_colors(frame)
        self.assertGreater(float(colors[:3, 0].max()), 0.95)
        self.assertTrue(np.allclose(colors[3:], 0.0))

    def test_invalid_extension_parent_is_rejected(self) -> None:
        spatial = spatial_config()
        spatial["strips"][1]["extends_strip_id"] = "missing"
        errors = validate_spatial_config(spatial, 8)
        self.assertTrue(any("extends_strip_id" in error for error in errors))

    def test_wled_output_routes_by_spatial_device(self) -> None:
        config = EngineConfig(total_leds=8, spatial=spatial_config(), send_to_wled=True, brightness=1.0)
        topology = SpatialRoomTopology(config)
        output = WLEDOutput(config, unittest.mock.Mock(), topology)
        left = FakeSession()
        right = FakeSession()
        output.device_sessions = {"left": left, "right": right}
        leds = np.arange(24, dtype=np.uint8).reshape(8, 3)
        self.assertTrue(output.send(leds))
        self.assertEqual(len(left.posts), 1)
        self.assertEqual(len(right.posts), 1)
        self.assertEqual(left.posts[0][1]["seg"][0]["i"][0], [0, 1, 2])
        self.assertEqual(right.posts[0][1]["seg"][0]["i"][2], [12, 13, 14])


if __name__ == "__main__":
    unittest.main()
