from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


WALLS = {"front", "left", "rear", "right"}
SYNC_MODES = {"spatial", "tv_image", "blend"}
TV_ROLES = {"none", "top", "bottom", "left", "right"}
EXTENSION_MODES = {"soft_spill", "edge_reach", "effects_only"}


@dataclass
class RoomDimensions:
    width: float = 4.0
    depth: float = 3.0
    height: float = 2.4
    unit: str = "m"


@dataclass
class TVPlacement:
    wall: str = "front"
    center_u: float = 2.0
    center_v: float = 1.25
    width: float = 1.4
    height: float = 0.8
    mirror_horizontal: bool = True


@dataclass
class SpatialDevice:
    id: str
    name: str
    ip: str = ""
    led_count: int = 300
    segment_id: int | None = None
    enabled: bool = True


@dataclass
class SpatialStrip:
    id: str
    name: str
    wall: str
    start_u: float
    start_v: float
    end_u: float
    end_v: float
    led_count: int
    direction: str = "forward"
    device_id: str = "main"
    device_start: int = 0
    sync_mode: str = "spatial"
    blend: float = 0.5
    tv_fill: float = 1.0
    tv_fill_spatial: bool = False
    tv_role: str = "none"
    extends_strip_id: str = ""
    extension_mode: str = "soft_spill"
    extension_strength: float = 0.45
    extension_softness: float = 0.65


@dataclass
class SpatialConfig:
    enabled: bool = False
    room: RoomDimensions = field(default_factory=RoomDimensions)
    tv: TVPlacement = field(default_factory=TVPlacement)
    devices: list[SpatialDevice] = field(default_factory=list)
    strips: list[SpatialStrip] = field(default_factory=list)


def default_spatial_dict(total_leds: int = 240) -> dict[str, Any]:
    per_wall = max(1, total_leds // 4)
    remainder = max(0, total_leds - per_wall * 4)
    counts = [per_wall, per_wall, per_wall, per_wall + remainder]
    room = RoomDimensions()
    return {
        "enabled": False,
        "room": room.__dict__,
        "tv": TVPlacement().__dict__,
        "devices": [
            {
                "id": "main",
                "name": "Main WLED",
                "ip": "",
                "led_count": total_leds,
                "segment_id": None,
                "enabled": True,
            }
        ],
        "strips": [
            _default_strip("front", counts[0], 0, room.width, 1.9, 1.9, "front-strip"),
            _default_strip("left", counts[1], counts[0], room.depth, 1.9, 1.9, "left-strip"),
            _default_strip("rear", counts[2], counts[0] + counts[1], room.width, 1.9, 1.9, "rear-strip"),
            _default_strip("right", counts[3], counts[0] + counts[1] + counts[2], room.depth, 1.9, 1.9, "right-strip"),
        ],
    }


def _default_strip(
    wall: str,
    led_count: int,
    device_start: int,
    wall_length: float,
    start_v: float,
    end_v: float,
    strip_id: str,
) -> dict[str, Any]:
    return {
        "id": strip_id,
        "name": wall.title(),
        "wall": wall,
        "start_u": 0.0,
        "start_v": start_v,
        "end_u": wall_length,
        "end_v": end_v,
        "led_count": led_count,
        "direction": "forward",
        "device_id": "main",
        "device_start": device_start,
        "sync_mode": "spatial",
        "blend": 0.5,
        "tv_fill": 1.0,
        "tv_fill_spatial": False,
        "tv_role": "none",
        "extends_strip_id": "",
        "extension_mode": "soft_spill",
        "extension_strength": 0.45,
        "extension_softness": 0.65,
    }


def parse_spatial_config(raw: dict[str, Any] | None, total_leds: int = 240) -> SpatialConfig:
    data = dict(default_spatial_dict(total_leds))
    if isinstance(raw, dict):
        data.update(raw)
    room = RoomDimensions(**_known(data.get("room"), RoomDimensions))
    tv = TVPlacement(**_known(data.get("tv"), TVPlacement))
    devices = [SpatialDevice(**_known(item, SpatialDevice)) for item in data.get("devices", []) if isinstance(item, dict)]
    strips = [SpatialStrip(**_known(item, SpatialStrip)) for item in data.get("strips", []) if isinstance(item, dict)]
    return SpatialConfig(bool(data.get("enabled", False)), room, tv, devices, strips)


def spatial_config_to_dict(config: SpatialConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "room": dict(config.room.__dict__),
        "tv": dict(config.tv.__dict__),
        "devices": [dict(device.__dict__) for device in config.devices],
        "strips": [dict(strip.__dict__) for strip in config.strips],
    }


def pack_spatial_device_ranges(raw: dict[str, Any] | None, total_leds: int = 240) -> dict[str, Any]:
    spatial = spatial_config_to_dict(parse_spatial_config(raw, total_leds))
    devices = spatial["devices"]
    strips = spatial["strips"]
    if not devices:
        return spatial
    devices_by_id = {device["id"]: device for device in devices if device.get("id")}
    if not devices_by_id:
        return spatial
    cursors: dict[str, int] = {}
    fallback_id = devices[0]["id"]
    for strip in strips:
        device_id = strip.get("device_id")
        if device_id not in devices_by_id:
            device_id = fallback_id
            strip["device_id"] = device_id
        led_count = max(1, int(strip.get("led_count", 1)))
        strip["led_count"] = led_count
        start = cursors.get(device_id, 0)
        strip["device_start"] = start
        cursors[device_id] = start + led_count
    for device_id, used in cursors.items():
        device = devices_by_id.get(device_id)
        if device is not None:
            device["led_count"] = max(int(device.get("led_count", 0)), used)
    return spatial


def validate_spatial_config(raw: dict[str, Any] | None, total_leds: int = 240) -> list[str]:
    spatial = parse_spatial_config(raw, total_leds)
    if not spatial.enabled:
        return []

    errors: list[str] = []
    room = spatial.room
    if room.width <= 0 or room.depth <= 0 or room.height <= 0:
        errors.append("spatial.room dimensions must be greater than 0")
    if spatial.tv.wall not in WALLS:
        errors.append("spatial.tv.wall must be front, left, rear, or right")
    else:
        wall_length = _wall_length(room, spatial.tv.wall)
        if spatial.tv.width <= 0 or spatial.tv.height <= 0:
            errors.append("spatial.tv width and height must be greater than 0")
        if not 0 <= spatial.tv.center_u <= wall_length:
            errors.append("spatial.tv.center_u must be inside the selected wall")
        if not 0 <= spatial.tv.center_v <= room.height:
            errors.append("spatial.tv.center_v must be inside the room height")

    device_ids: set[str] = set()
    for device in spatial.devices:
        if not device.id.strip():
            errors.append("spatial.devices id is required")
        if device.id in device_ids:
            errors.append(f"spatial.devices duplicate id {device.id}")
        device_ids.add(device.id)
        if device.led_count <= 0:
            errors.append(f"spatial.devices.{device.id}.led_count must be greater than 0")

    strip_ids: set[str] = set()
    device_ranges: dict[str, list[tuple[int, int, str]]] = {}
    for strip in spatial.strips:
        if strip.id in strip_ids:
            errors.append(f"spatial.strips duplicate id {strip.id}")
        strip_ids.add(strip.id)
        if strip.wall not in WALLS:
            errors.append(f"spatial.strips.{strip.id}.wall is invalid")
            continue
        if strip.sync_mode not in SYNC_MODES:
            errors.append(f"spatial.strips.{strip.id}.sync_mode must be spatial, tv_image, or blend")
        if strip.tv_role not in TV_ROLES:
            errors.append(f"spatial.strips.{strip.id}.tv_role must be none, top, bottom, left, or right")
        if strip.extension_mode not in EXTENSION_MODES:
            errors.append(f"spatial.strips.{strip.id}.extension_mode must be soft_spill, edge_reach, or effects_only")
        if strip.tv_role != "none" and strip.extends_strip_id:
            errors.append(f"spatial.strips.{strip.id} cannot set both tv_role and extends_strip_id")
        if strip.led_count <= 0:
            errors.append(f"spatial.strips.{strip.id}.led_count must be greater than 0")
        if strip.device_id not in device_ids:
            errors.append(f"spatial.strips.{strip.id}.device_id must reference a device")
        wall_length = _wall_length(room, strip.wall)
        for attr in ("start_u", "end_u"):
            if not 0 <= getattr(strip, attr) <= wall_length:
                errors.append(f"spatial.strips.{strip.id}.{attr} must be inside the wall length")
        for attr in ("start_v", "end_v"):
            if not 0 <= getattr(strip, attr) <= room.height:
                errors.append(f"spatial.strips.{strip.id}.{attr} must be inside the room height")
        if not 0.0 <= strip.blend <= 1.0:
            errors.append(f"spatial.strips.{strip.id}.blend must be between 0.0 and 1.0")
        if not 0.0 <= strip.tv_fill <= 1.0:
            errors.append(f"spatial.strips.{strip.id}.tv_fill must be between 0.0 and 1.0")
        if not 0.0 <= strip.extension_strength <= 1.0:
            errors.append(f"spatial.strips.{strip.id}.extension_strength must be between 0.0 and 1.0")
        if not 0.0 <= strip.extension_softness <= 1.0:
            errors.append(f"spatial.strips.{strip.id}.extension_softness must be between 0.0 and 1.0")
        end = strip.device_start + strip.led_count
        device = next((item for item in spatial.devices if item.id == strip.device_id), None)
        if strip.device_start < 0 or (device and end > device.led_count):
            errors.append(f"spatial.strips.{strip.id} device LED range exceeds device length")
        device_ranges.setdefault(strip.device_id, []).append((strip.device_start, end, strip.id))

    for device_id, ranges in device_ranges.items():
        ordered = sorted(ranges)
        for prev, curr in zip(ordered, ordered[1:]):
            if prev[1] > curr[0]:
                errors.append(f"spatial.strips {prev[2]} and {curr[2]} overlap on device {device_id}")
    errors.extend(_validate_extension_chains(spatial.strips))
    return errors


def _validate_extension_chains(strips: list[SpatialStrip]) -> list[str]:
    errors: list[str] = []
    by_id = {strip.id: strip for strip in strips}
    for strip in strips:
        parent_id = strip.extends_strip_id.strip()
        if not parent_id:
            continue
        if parent_id not in by_id:
            errors.append(f"spatial.strips.{strip.id}.extends_strip_id must reference a strip")
            continue
        seen = {strip.id}
        current = by_id[parent_id]
        while current.extends_strip_id.strip():
            if current.id in seen:
                errors.append(f"spatial.strips.{strip.id}.extends_strip_id creates a cycle")
                break
            seen.add(current.id)
            next_id = current.extends_strip_id.strip()
            if next_id not in by_id:
                errors.append(f"spatial.strips.{current.id}.extends_strip_id must reference a strip")
                break
            current = by_id[next_id]
        if current.tv_role == "none":
            errors.append(f"spatial.strips.{strip.id}.extends_strip_id must inherit from a TV-side strip")
    return errors


def _known(raw: Any, cls: type) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    names = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return {key: value for key, value in raw.items() if key in names}


def _wall_length(room: RoomDimensions, wall: str) -> float:
    return room.width if wall in {"front", "rear"} else room.depth
