from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import EngineConfig, WallConfig
from spatial_config import SpatialConfig, SpatialDevice, SpatialStrip, parse_spatial_config, spatial_config_to_dict


@dataclass(frozen=True)
class DeviceRoute:
    device: SpatialDevice
    global_indices: np.ndarray
    device_indices: np.ndarray


class SpatialRoomTopology:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.spatial: SpatialConfig = parse_spatial_config(config.spatial, config.total_leds)
        self.config.spatial = spatial_config_to_dict(self.spatial)
        self.total = sum(strip.led_count for strip in self.spatial.strips)
        self.global_indices = np.arange(self.total, dtype=np.int32)
        self.positions = np.zeros((self.total, 3), dtype=np.float32)
        self.wall_u = np.zeros(self.total, dtype=np.float32)
        self.wall_v = np.zeros(self.total, dtype=np.float32)
        self.strip_blend = np.zeros(self.total, dtype=np.float32)
        self.wall_names: list[str] = []
        self.strip_ids: list[str] = []
        self.sync_modes: list[str] = []
        self.effective_tv_roles: list[str] = []
        self.device_ids: list[str] = []
        self.device_led_indices = np.zeros(self.total, dtype=np.int32)
        self.strip_ranges: dict[str, tuple[int, int]] = {}
        self.wall_ranges: dict[str, list[int]] = {"front": [], "left": [], "rear": [], "right": []}
        self.devices = {device.id: device for device in self.spatial.devices}
        self.strips = {strip.id: strip for strip in self.spatial.strips}
        self._effective_role_by_strip = self._resolve_effective_tv_roles()
        self._routes: list[DeviceRoute] | None = None
        self._build()

    @property
    def enabled(self) -> bool:
        return self.spatial.enabled

    def apply_legacy_fields(self, config: EngineConfig) -> None:
        if self.total <= 0:
            return
        config.total_leds = self.total
        for wall in ("front", "left", "rear", "right"):
            indices = self.wall_ranges[wall]
            if indices:
                setattr(config, f"{wall}_wall", WallConfig(indices[0], indices[-1]))
            else:
                setattr(config, f"{wall}_wall", WallConfig(0, 0))

        tv_indices = self.tv_wall_indices()
        if tv_indices:
            config.tv_left_boundary = tv_indices[0]
            config.tv_center_led = tv_indices[len(tv_indices) // 2]
            config.tv_right_boundary = tv_indices[-1]

    def range_inclusive(self, start: int, end: int) -> list[int]:
        start %= self.total
        end %= self.total
        if start <= end:
            return list(range(start, end + 1))
        return list(range(start, self.total)) + list(range(0, end + 1))

    def circular_distance(self, a: int, b: int) -> int:
        diff = abs((a % self.total) - (b % self.total))
        return min(diff, self.total - diff)

    def signed_distance(self, origin: float, position: int, direction: int) -> float:
        position %= self.total
        origin %= self.total
        if direction >= 0:
            return (position - origin) % self.total
        return (origin - position) % self.total

    def wall_for_led(self, led: int) -> str | None:
        led %= self.total
        return self.wall_names[led] if 0 <= led < len(self.wall_names) else None

    def adjacent_walls(self, wall: str) -> tuple[str, str]:
        order = self.config.clockwise_order
        idx = order.index(wall)
        return order[(idx - 1) % len(order)], order[(idx + 1) % len(order)]

    def direction_from_to(self, origin: int, target_wall: str) -> int:
        indices = self.wall_ranges[target_wall]
        target = indices[len(indices) // 2] if indices else origin
        cw = (target - origin) % self.total
        ccw = (origin - target) % self.total
        return 1 if cw <= ccw else -1

    def origin_for_edge(self, edge: str, intensity: float):
        from topology import Origin

        led = self.nearest_led_to_point(self.event_origin_point(edge))
        radius = self.config.normal_spill_radius
        if intensity >= self.config.level2_threshold:
            radius = self.total // 2
        elif intensity >= self.config.level1_threshold:
            radius = self.config.strong_spill_radius
        return Origin(led, (-1, 1), radius, f"screen-{edge}")

    def event_origin_point(self, edge: str) -> np.ndarray:
        tv = self.spatial.tv
        left_u = tv.center_u - tv.width / 2.0
        right_u = tv.center_u + tv.width / 2.0
        top_v = tv.center_v + tv.height / 2.0
        bottom_v = tv.center_v - tv.height / 2.0
        if edge == "left":
            return self.wall_point(tv.wall, left_u, tv.center_v)
        if edge == "right":
            return self.wall_point(tv.wall, right_u, tv.center_v)
        if edge == "bottom":
            return self.wall_point(tv.wall, tv.center_u, bottom_v)
        return self.wall_point(tv.wall, tv.center_u, top_v)

    def nearest_led_to_point(self, point: np.ndarray) -> int:
        if self.total <= 0:
            return 0
        distances = np.linalg.norm(self.positions - point.reshape(1, 3), axis=1)
        return int(np.argmin(distances))

    def room_center(self) -> np.ndarray:
        room = self.spatial.room
        return np.array([room.width / 2.0, room.height / 2.0, room.depth / 2.0], dtype=np.float32)

    def room_diagonal(self) -> float:
        room = self.spatial.room
        return float(np.linalg.norm([room.width, room.height, room.depth]))

    def device_routes(self) -> list[DeviceRoute]:
        if self._routes is not None:
            return self._routes
        routes: list[DeviceRoute] = []
        for device in self.spatial.devices:
            if not device.enabled:
                continue
            indices = np.array([idx for idx, did in enumerate(self.device_ids) if did == device.id], dtype=np.int32)
            if indices.size == 0:
                continue
            routes.append(DeviceRoute(device, indices, self.device_led_indices[indices]))
        self._routes = routes
        return routes

    def tv_wall_indices(self) -> list[int]:
        tv = self.spatial.tv
        left = tv.center_u - tv.width / 2.0
        right = tv.center_u + tv.width / 2.0
        bottom = tv.center_v - tv.height / 2.0
        top = tv.center_v + tv.height / 2.0
        result = [
            idx
            for idx, wall in enumerate(self.wall_names)
            if wall == tv.wall and left <= self.wall_u[idx] <= right and bottom <= self.wall_v[idx] <= top
        ]
        return result or self.wall_ranges.get(tv.wall, [])

    def tv_sample_colors(self, frame_bgr: np.ndarray | None) -> np.ndarray:
        colors = np.zeros((self.total, 3), dtype=np.float32)
        if frame_bgr is None or frame_bgr.size == 0 or self.total <= 0:
            return colors
        frame = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
        for strip in self.spatial.strips:
            role = self.effective_tv_role_for_strip(strip.id)
            if role == "none" or strip.sync_mode not in {"tv_image", "blend"}:
                continue
            start, end = self.strip_ranges.get(strip.id, (0, 0))
            idx = np.arange(start, end, dtype=np.int32)
            if idx.size:
                colors[idx] = self._sample_tv_role(frame, idx, role)

        roles = np.array(self.effective_tv_roles, dtype=object)
        tv_capable = np.array([mode in {"tv_image", "blend"} for mode in self.sync_modes], dtype=bool)
        unmapped = np.nonzero((roles == "none") & tv_capable & self._same_wall_tv_mask())[0].astype(np.int32)
        if unmapped.size:
            colors[unmapped] = self._sample_tv_rect(frame, unmapped)
        return colors

    def effective_tv_role_for_strip(self, strip_id: str) -> str:
        return self._effective_role_by_strip.get(strip_id, "none")

    def wall_point(self, wall: str, u: float, v: float) -> np.ndarray:
        room = self.spatial.room
        u = float(u)
        v = float(v)
        if wall == "front":
            return np.array([u, v, 0.0], dtype=np.float32)
        if wall == "rear":
            return np.array([room.width - u, v, room.depth], dtype=np.float32)
        if wall == "left":
            return np.array([room.width, v, u], dtype=np.float32)
        return np.array([0.0, v, room.depth - u], dtype=np.float32)

    def _build(self) -> None:
        cursor = 0
        for strip in self.spatial.strips:
            start = cursor
            self._build_strip(strip, cursor)
            cursor += strip.led_count
            self.strip_ranges[strip.id] = (start, cursor)

    def _build_strip(self, strip: SpatialStrip, cursor: int) -> None:
        count = strip.led_count
        t = np.linspace(0.0, 1.0, count, dtype=np.float32)
        if strip.direction == "reverse":
            t = 1.0 - t
        us = strip.start_u + (strip.end_u - strip.start_u) * t
        vs = strip.start_v + (strip.end_v - strip.start_v) * t
        for offset in range(count):
            idx = cursor + offset
            self.positions[idx] = self.wall_point(strip.wall, float(us[offset]), float(vs[offset]))
            self.wall_u[idx] = us[offset]
            self.wall_v[idx] = vs[offset]
            self.strip_blend[idx] = strip.blend
            self.wall_names.append(strip.wall)
            self.strip_ids.append(strip.id)
            self.sync_modes.append(strip.sync_mode)
            self.effective_tv_roles.append(self.effective_tv_role_for_strip(strip.id))
            self.device_ids.append(strip.device_id)
            self.device_led_indices[idx] = strip.device_start + offset
            self.wall_ranges[strip.wall].append(idx)

    def _resolve_effective_tv_roles(self) -> dict[str, str]:
        by_id = {strip.id: strip for strip in self.spatial.strips}
        resolved: dict[str, str] = {}

        def resolve(strip: SpatialStrip, seen: set[str]) -> str:
            if strip.id in resolved:
                return resolved[strip.id]
            if strip.id in seen:
                return "none"
            if strip.tv_role != "none":
                resolved[strip.id] = strip.tv_role
                return strip.tv_role
            parent_id = strip.extends_strip_id.strip()
            parent = by_id.get(parent_id)
            if parent is None:
                resolved[strip.id] = "none"
                return "none"
            role = resolve(parent, seen | {strip.id})
            resolved[strip.id] = role
            return role

        for strip in self.spatial.strips:
            resolve(strip, set())
        return resolved

    def _same_wall_tv_mask(self) -> np.ndarray:
        tv = self.spatial.tv
        left = tv.center_u - tv.width / 2.0
        right = tv.center_u + tv.width / 2.0
        bottom = tv.center_v - tv.height / 2.0
        top = tv.center_v + tv.height / 2.0
        return np.array(
            [
                wall == tv.wall and left <= self.wall_u[idx] <= right and bottom <= self.wall_v[idx] <= top
                for idx, wall in enumerate(self.wall_names)
            ],
            dtype=bool,
        )

    def _sample_tv_rect(self, frame: np.ndarray, idx: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        tv = self.spatial.tv
        left = tv.center_u - tv.width / 2.0
        bottom = tv.center_v - tv.height / 2.0
        tv_w = max(0.001, tv.width)
        tv_h = max(0.001, tv.height)
        u_norm = np.clip((self.wall_u[idx] - left) / tv_w, 0.0, 1.0)
        v_norm = np.clip((self.wall_v[idx] - bottom) / tv_h, 0.0, 1.0)
        xs = np.clip((u_norm * (width - 1)).astype(np.int32), 0, width - 1)
        ys = np.clip(((1.0 - v_norm) * (height - 1)).astype(np.int32), 0, height - 1)
        return frame[ys, xs]

    def _sample_tv_role(self, frame: np.ndarray, idx: np.ndarray, role: str) -> np.ndarray:
        height, width = frame.shape[:2]
        if role in {"top", "bottom"}:
            progress = self._tv_edge_progress(idx, role)
            xs = np.clip(np.rint(progress * (width - 1)).astype(np.int32), 0, width - 1)
            edge_h = max(1, int(round(height * 0.04)))
            band = frame[:edge_h, :] if role == "top" else frame[height - edge_h :, :]
            sampled = band[:, xs].mean(axis=0)
        else:
            progress = self._tv_edge_progress(idx, role)
            ys = np.clip(np.rint((1.0 - progress) * (height - 1)).astype(np.int32), 0, height - 1)
            edge_w = max(1, int(round(width * 0.04)))
            band = frame[:, :edge_w] if role == "left" else frame[:, width - edge_w :]
            sampled = band[ys, :].mean(axis=1)
        return sampled

    def _tv_edge_progress(self, idx: np.ndarray, role: str) -> np.ndarray:
        tv = self.spatial.tv
        if role in {"top", "bottom"}:
            axis = self.wall_u[idx]
            tv_min = tv.center_u - tv.width / 2.0
            tv_max = tv.center_u + tv.width / 2.0
        else:
            axis = self.wall_v[idx]
            tv_min = tv.center_v - tv.height / 2.0
            tv_max = tv.center_v + tv.height / 2.0
        span = max(0.001, tv_max - tv_min)
        overlaps_tv_axis = np.any((axis >= tv_min) & (axis <= tv_max))
        if overlaps_tv_axis:
            return np.clip((axis - tv_min) / span, 0.0, 1.0).astype(np.float32)
        axis_min = float(np.min(axis))
        axis_max = float(np.max(axis))
        axis_span = axis_max - axis_min
        if axis_span > 0.0001:
            return np.clip((axis - axis_min) / axis_span, 0.0, 1.0).astype(np.float32)
        return np.linspace(0.0, 1.0, idx.size, dtype=np.float32)
