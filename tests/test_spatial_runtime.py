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

    def test_tv_capable_effects_only_extension_still_samples_tv_image(self) -> None:
        spatial = spatial_config()
        parent = dict(
            spatial["strips"][0],
            id="tv-top",
            tv_role="top",
            sync_mode="tv_image",
            device_id="left",
            device_start=0,
        )
        extension = dict(
            spatial["strips"][0],
            id="ceiling-front",
            tv_role="none",
            sync_mode="blend",
            blend=0.5,
            extends_strip_id="tv-top",
            extension_mode="effects_only",
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=8, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0)
        topology = SpatialRoomTopology(config)
        renderer = VectorizedSpatialRenderer(config, topology)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[0, :, 2] = 255
        renderer.set_tv_frame(frame)
        leds = renderer.step(0.1)
        parent_start, parent_end = topology.strip_ranges["tv-top"]
        extension_start, extension_end = topology.strip_ranges["ceiling-front"]
        self.assertGreater(int(leds[parent_start:parent_end, 0].max()), 240)
        self.assertGreater(int(leds[extension_start:extension_end, 0].max()), 110)

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
        self.assertLess(float(colors[3, 0]), 0.20)
        self.assertGreater(float(colors[5, 0]), 0.95)

    def test_tv_fill_zero_uses_tv_width_and_remaps_full_edge(self) -> None:
        spatial = spatial_config()
        spatial["strips"] = [
            dict(
                spatial["strips"][0],
                id="wide-top",
                start_u=0.0,
                end_u=4.0,
                led_count=9,
                tv_role="top",
                sync_mode="tv_image",
                tv_fill=0.0,
                device_id="left",
                device_start=0,
            )
        ]
        config = EngineConfig(total_leds=9, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = np.array([32, 80, 128, 192, 255], dtype=np.uint8)
        colors = topology.tv_sample_colors(frame)
        lit = np.nonzero(np.max(colors, axis=1) > 0.01)[0].tolist()
        self.assertEqual(lit, [3, 4, 5])
        self.assertLess(float(colors[3, 0]), 0.20)
        self.assertGreater(float(colors[5, 0]), 0.95)

    def test_tv_fill_one_maps_full_tv_edge_across_whole_strip(self) -> None:
        spatial = spatial_config()
        spatial["strips"] = [
            dict(
                spatial["strips"][0],
                id="wide-top",
                start_u=0.0,
                end_u=4.0,
                led_count=9,
                tv_role="top",
                sync_mode="tv_image",
                tv_fill=1.0,
                device_id="left",
                device_start=0,
            )
        ]
        config = EngineConfig(total_leds=9, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = np.array([32, 80, 128, 192, 255], dtype=np.uint8)
        colors = topology.tv_sample_colors(frame)
        lit = np.nonzero(np.max(colors, axis=1) > 0.01)[0].tolist()
        self.assertEqual(lit, list(range(9)))
        self.assertGreater(float(colors[0, 0]), 0.10)
        self.assertGreater(float(colors[-1, 0]), 0.95)

    def test_tv_fill_spatial_uses_tv_center_relative_to_strip(self) -> None:
        spatial = spatial_config()
        spatial["tv"]["center_u"] = 1.0
        spatial["tv"]["width"] = 1.0
        spatial["strips"] = [
            dict(
                spatial["strips"][0],
                id="wide-top",
                start_u=0.0,
                end_u=4.0,
                led_count=9,
                tv_role="top",
                sync_mode="tv_image",
                tv_fill=0.0,
                tv_fill_spatial=True,
                device_id="left",
                device_start=0,
            )
        ]
        config = EngineConfig(total_leds=9, spatial=spatial)
        topology = SpatialRoomTopology(config)
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = 255
        colors = topology.tv_sample_colors(frame)
        lit = np.nonzero(np.max(colors, axis=1) > 0.01)[0].tolist()
        self.assertEqual(lit, [1, 2, 3])

    def test_event_origin_uses_spatial_aware_tv_fill_projection(self) -> None:
        spatial = spatial_config()
        spatial["strips"] = [
            dict(
                spatial["strips"][0],
                id="wide-top",
                start_u=0.0,
                end_u=4.0,
                start_v=2.2,
                end_v=2.2,
                led_count=9,
                tv_role="top",
                sync_mode="tv_image",
                tv_fill=1.0,
                tv_fill_spatial=True,
                device_id="left",
                device_start=0,
            )
        ]
        topology = SpatialRoomTopology(EngineConfig(total_leds=9, spatial=spatial))
        origin = topology.event_origin_point("top")
        self.assertAlmostEqual(float(origin[1]), 2.2, places=3)

    def test_event_origin_ignores_non_spatial_aware_tv_fill(self) -> None:
        spatial = spatial_config()
        spatial["strips"] = [
            dict(
                spatial["strips"][0],
                id="wide-top",
                start_u=0.0,
                end_u=4.0,
                start_v=2.2,
                end_v=2.2,
                led_count=9,
                tv_role="top",
                sync_mode="tv_image",
                tv_fill=1.0,
                tv_fill_spatial=False,
                device_id="left",
                device_start=0,
            )
        ]
        topology = SpatialRoomTopology(EngineConfig(total_leds=9, spatial=spatial))
        origin = topology.event_origin_point("top")
        expected = spatial["tv"]["center_v"] + spatial["tv"]["height"] / 2.0
        self.assertAlmostEqual(float(origin[1]), expected, places=3)

    def test_extension_defaults_are_added_to_old_configs(self) -> None:
        spatial = spatial_config()
        spatial["tv"].pop("mirror_horizontal", None)
        for strip in spatial["strips"]:
            strip.pop("extension_mode", None)
            strip.pop("extension_strength", None)
            strip.pop("extension_softness", None)
            strip.pop("tv_fill", None)
            strip.pop("tv_fill_spatial", None)
        parsed = parse_spatial_config(spatial, 8)
        saved = spatial_config_to_dict(parsed)
        self.assertTrue(saved["tv"]["mirror_horizontal"])
        self.assertEqual(saved["strips"][0]["extension_mode"], "soft_spill")
        self.assertEqual(saved["strips"][0]["extension_strength"], 0.45)
        self.assertEqual(saved["strips"][0]["extension_softness"], 0.65)
        self.assertEqual(saved["strips"][0]["tv_fill"], 1.0)
        self.assertFalse(saved["strips"][0]["tv_fill_spatial"])

    def test_spatial_validation_rejects_invalid_tv_fill(self) -> None:
        spatial = spatial_config()
        spatial["strips"][0]["tv_fill"] = 1.4
        errors = validate_spatial_config(spatial, 8)
        self.assertTrue(any("tv_fill" in error for error in errors))

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

    def test_ambient_side_spill_extends_rendered_front_ambient_edge_only(self) -> None:
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
            id="ambient-extension",
            tv_role="none",
            extends_strip_id="parent",
            start_u=2.7,
            end_u=3.4,
            sync_mode="spatial",
            extension_mode="soft_spill",
            extension_strength=0.6,
            extension_softness=0.65,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=6, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient")
        topology = SpatialRoomTopology(config)
        renderer = VectorizedSpatialRenderer(config, topology)
        renderer.set_front_ambient_strip(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32), 1.0)
        leds = renderer.step(0.1)
        near = leds[3] / 255.0
        far = leds[5] / 255.0
        self.assertGreater(float(near[0]), 0.05)
        self.assertLess(float(near[2]), float(near[0]) * 0.25)
        self.assertLess(float(far.max()), float(near.max()) * 0.05)

    def test_ambient_side_spill_boosts_when_uniform_scene_brightens(self) -> None:
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
            id="ambient-extension",
            tv_role="none",
            extends_strip_id="parent",
            start_u=2.7,
            end_u=3.8,
            led_count=5,
            sync_mode="spatial",
            extension_mode="soft_spill",
            extension_strength=0.6,
            extension_softness=0.65,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=8, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient")
        topology = SpatialRoomTopology(config)

        steady = VectorizedSpatialRenderer(config, topology)
        steady._front_ambient_scene_luma = 0.2126
        red = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1))
        steady.set_front_ambient_strip(red, 1.0)
        steady_leds = steady.step(0.1)

        rising = VectorizedSpatialRenderer(config, topology)
        rising.set_front_ambient_strip(red, 0.2)
        rising.step(0.1)
        rising.set_front_ambient_strip(red, 1.0)
        rising_leds = rising.step(0.1)

        self.assertGreater(int(rising_leds[3, 0]), int(steady_leds[3, 0]))
        self.assertGreater(int(rising_leds[5, 0]), int(steady_leds[5, 0]) + 4)
        self.assertLess(int(rising_leds[5, 0]), int(rising_leds[3, 0]))

    def test_ambient_side_spill_boost_requires_uniform_color(self) -> None:
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
            id="ambient-extension",
            tv_role="none",
            extends_strip_id="parent",
            start_u=2.7,
            end_u=3.8,
            led_count=5,
            sync_mode="spatial",
            extension_mode="soft_spill",
            extension_strength=0.6,
            extension_softness=0.65,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=8, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient")
        topology = SpatialRoomTopology(config)
        renderer = VectorizedSpatialRenderer(config, topology)
        mixed = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        renderer.set_front_ambient_strip(mixed, 0.2)
        renderer.step(0.1)
        renderer.set_front_ambient_strip(mixed, 1.0)
        renderer.step(0.1)
        self.assertLess(renderer._ambient_side_spill_boost, 0.1)

    def test_scene_bloom_boost_increases_ambient_side_spill_reach(self) -> None:
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
            id="ambient-extension",
            tv_role="none",
            extends_strip_id="parent",
            start_u=2.7,
            end_u=3.8,
            led_count=5,
            sync_mode="spatial",
            extension_mode="soft_spill",
            extension_strength=0.6,
            extension_softness=0.65,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=8, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient")
        topology = SpatialRoomTopology(config)
        cyan = (170, 245, 255)
        red_edge = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1))

        baseline = VectorizedSpatialRenderer(config, topology)
        baseline.set_front_ambient_strip(red_edge, 1.0)
        baseline_leds = baseline.step(0.1)

        boosted = VectorizedSpatialRenderer(config, topology)
        boosted.set_front_ambient_strip(red_edge, 1.0)
        boosted.set_ambient_spill_scene_boost(0.8, cyan)
        boosted_leds = boosted.step(0.1)

        self.assertGreater(int(boosted_leds[5].max()), int(baseline_leds[5].max()) + 4)
        self.assertGreater(int(boosted_leds[3, 2]), int(boosted_leds[3, 0]))
        self.assertGreater(int(boosted_leds[3, 1]), int(boosted_leds[3, 0]))

    def test_ambient_side_spill_base_intensity_controls_normal_spill(self) -> None:
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
            id="ambient-extension",
            tv_role="none",
            extends_strip_id="parent",
            sync_mode="spatial",
            extension_mode="soft_spill",
            extension_strength=0.6,
            extension_softness=0.65,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        low_config = EngineConfig(total_leds=6, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient", ambient_side_spill_base_intensity=0.2)
        high_config = EngineConfig(total_leds=6, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient", ambient_side_spill_base_intensity=0.8)
        sample = np.ones((3, 3), dtype=np.float32)
        low = VectorizedSpatialRenderer(low_config, SpatialRoomTopology(low_config))
        low.set_front_ambient_strip(sample, 1.0)
        low_leds = low.step(0.1)
        high = VectorizedSpatialRenderer(high_config, SpatialRoomTopology(high_config))
        high.set_front_ambient_strip(sample, 1.0)
        high_leds = high.step(0.1)
        self.assertGreater(int(high_leds[3].max()), int(low_leds[3].max()) * 2)

    def test_ambient_side_spill_boost_intensity_controls_boosted_reach(self) -> None:
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
            id="ambient-extension",
            tv_role="none",
            extends_strip_id="parent",
            start_u=2.7,
            end_u=3.8,
            led_count=5,
            sync_mode="spatial",
            extension_mode="soft_spill",
            extension_strength=0.6,
            extension_softness=0.65,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        off_config = EngineConfig(total_leds=8, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient", ambient_side_spill_boost_intensity=0.0)
        high_config = EngineConfig(total_leds=8, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient", ambient_side_spill_boost_intensity=2.0)
        sample = np.ones((3, 3), dtype=np.float32)
        off = VectorizedSpatialRenderer(off_config, SpatialRoomTopology(off_config))
        off.set_front_ambient_strip(sample, 1.0)
        off.set_ambient_spill_scene_boost(0.8, (170, 245, 255))
        off_leds = off.step(0.1)
        high = VectorizedSpatialRenderer(high_config, SpatialRoomTopology(high_config))
        high.set_front_ambient_strip(sample, 1.0)
        high.set_ambient_spill_scene_boost(0.8, (170, 245, 255))
        high_leds = high.step(0.1)
        self.assertGreater(int(high_leds[5].max()), int(off_leds[5].max()) + 10)

    def test_scene_bloom_boost_respects_disabled_ambient_side_spill(self) -> None:
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
            id="ambient-extension",
            tv_role="none",
            extends_strip_id="parent",
            sync_mode="spatial",
            extension_mode="soft_spill",
            extension_strength=0.6,
            extension_softness=0.65,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=6, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient")
        config.enabled_effects["ambient_side_spill"] = False
        renderer = VectorizedSpatialRenderer(config, SpatialRoomTopology(config))
        renderer.set_front_ambient_strip(np.ones((3, 3), dtype=np.float32), 1.0)
        renderer.set_ambient_spill_scene_boost(0.9, (170, 245, 255))
        leds = renderer.step(0.1)
        self.assertEqual(int(leds[3:].max()), 0)

    def test_scene_bloom_boost_respects_effects_only_extension(self) -> None:
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
            id="ambient-extension",
            tv_role="none",
            extends_strip_id="parent",
            sync_mode="spatial",
            extension_mode="effects_only",
            extension_strength=1.0,
            extension_softness=0.0,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=6, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient")
        renderer = VectorizedSpatialRenderer(config, SpatialRoomTopology(config))
        renderer.set_front_ambient_strip(np.ones((3, 3), dtype=np.float32), 1.0)
        renderer.set_ambient_spill_scene_boost(0.9, (170, 245, 255))
        leds = renderer.step(0.1)
        self.assertEqual(int(leds[3:].max()), 0)

    def test_ambient_side_spill_can_be_disabled(self) -> None:
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
            id="ambient-extension",
            tv_role="none",
            extends_strip_id="parent",
            sync_mode="spatial",
            extension_mode="soft_spill",
            extension_strength=0.6,
            extension_softness=0.65,
            device_id="right",
            device_start=0,
        )
        spatial["strips"] = [parent, extension]
        config = EngineConfig(total_leds=6, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0)
        config.enabled_effects["ambient_side_spill"] = False
        topology = SpatialRoomTopology(config)
        renderer = VectorizedSpatialRenderer(config, topology)
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        frame[0, :, 2] = 255
        renderer.set_tv_frame(frame)
        leds = renderer.step(0.1)
        self.assertGreater(int(leds[:3, 0].max()), 240)
        self.assertEqual(int(leds[3:].max()), 0)

    def test_front_ambient_coverage_controls_spatial_front_span(self) -> None:
        spatial = spatial_config()
        spatial["strips"] = [
            dict(
                spatial["strips"][0],
                id="front-wide",
                start_u=0.0,
                end_u=4.0,
                led_count=9,
                tv_role="top",
                sync_mode="spatial",
                device_id="left",
                device_start=0,
            )
        ]
        config = EngineConfig(total_leds=9, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient", front_ambient_coverage=1.0)
        topology = SpatialRoomTopology(config)
        renderer = VectorizedSpatialRenderer(config, topology)
        renderer.set_front_ambient_strip(np.ones((9, 3), dtype=np.float32), 1.0)
        full = renderer.step(0.1)
        self.assertGreater(int(full.min()), 200)

        config.front_ambient_coverage = 0.0
        topology = SpatialRoomTopology(config)
        renderer = VectorizedSpatialRenderer(config, topology)
        renderer.set_front_ambient_strip(np.ones((1, 3), dtype=np.float32), 1.0)
        none = renderer.step(0.1)
        self.assertEqual(int(none.max()), 0)

    def test_spatial_renderer_accepts_front_ambient_strip_samples(self) -> None:
        config = EngineConfig(total_leds=12, spatial=spatial_config(), lighting_mode="front_ambient")
        renderer = VectorizedSpatialRenderer(config, SpatialRoomTopology(config))
        self.assertTrue(renderer.accepts_front_ambient_strip_sample)

    def test_front_ambient_coverage_updates_live_for_blend_front_strip(self) -> None:
        spatial = spatial_config()
        spatial["strips"] = [
            dict(
                spatial["strips"][0],
                id="front-wide",
                start_u=0.0,
                end_u=4.0,
                led_count=9,
                tv_role="none",
                sync_mode="blend",
                blend=0.5,
                device_id="left",
                device_start=0,
            )
        ]
        config = EngineConfig(
            total_leds=9,
            spatial=spatial,
            brightness=1.0,
            gamma=1.0,
            color_smoothing=0.0,
            lighting_mode="front_ambient",
            front_ambient_coverage=1.0,
        )
        topology = SpatialRoomTopology(config)
        renderer = VectorizedSpatialRenderer(config, topology)
        renderer.set_tv_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        renderer.set_front_ambient_strip(np.ones((9, 3), dtype=np.float32), 1.0)
        full = renderer.step(0.1)
        self.assertGreater(np.count_nonzero(np.max(full, axis=1)), 7)

        config.front_ambient_coverage = 0.25
        renderer.front_ambient_intensity = 1.0
        renderer.set_front_ambient_strip(np.ones((renderer.front_ambient_led_count(), 3), dtype=np.float32), 1.0)
        partial = renderer.step(0.1)
        self.assertLess(np.count_nonzero(np.max(partial, axis=1)), np.count_nonzero(np.max(full, axis=1)))

    def test_front_ambient_does_not_overwrite_tv_image_strip(self) -> None:
        spatial = spatial_config()
        spatial["strips"] = [
            dict(
                spatial["strips"][0],
                id="front-tv",
                start_u=0.0,
                end_u=4.0,
                led_count=5,
                tv_role="top",
                sync_mode="tv_image",
                device_id="left",
                device_start=0,
            )
        ]
        config = EngineConfig(total_leds=5, spatial=spatial, brightness=1.0, gamma=1.0, color_smoothing=0.0, lighting_mode="front_ambient")
        renderer = VectorizedSpatialRenderer(config, SpatialRoomTopology(config))
        renderer.set_tv_frame(np.zeros((10, 10, 3), dtype=np.uint8))
        renderer.set_front_ambient_strip(np.ones((5, 3), dtype=np.float32), 1.0)
        leds = renderer.step(0.1)
        self.assertEqual(int(leds.max()), 0)

    def test_ambient_side_spill_respects_effects_only_extension(self) -> None:
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
            sync_mode="spatial",
            extension_mode="effects_only",
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
        self.assertEqual(int(leds[3:].max()), 0)

    def test_effects_only_extension_with_tv_image_sync_receives_tv_color(self) -> None:
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
        self.assertGreater(float(colors[3:, 0].max()), 0.95)

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
