from __future__ import annotations

from dataclasses import dataclass

from config import EngineConfig, WallConfig


@dataclass(frozen=True)
class Origin:
    led: int
    directions: tuple[int, ...]
    radius: int
    label: str


class RoomTopology:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.total = config.total_leds
        self.walls: dict[str, WallConfig] = {
            "front": config.front_wall,
            "left": config.left_wall,
            "rear": config.rear_wall,
            "right": config.right_wall,
        }
        self.clockwise_order = config.clockwise_order
        self._wall_leds = {name: self.range_inclusive(w.start, w.end) for name, w in self.walls.items()}

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
        for name, leds in self._wall_leds.items():
            if led in leds:
                return name
        return None

    def adjacent_walls(self, wall: str) -> tuple[str, str]:
        idx = self.clockwise_order.index(wall)
        return (
            self.clockwise_order[(idx - 1) % len(self.clockwise_order)],
            self.clockwise_order[(idx + 1) % len(self.clockwise_order)],
        )

    def direction_from_to(self, origin: int, target_wall: str) -> int:
        leds = self._wall_leds[target_wall]
        target = leds[len(leds) // 2]
        cw = (target - origin) % self.total
        ccw = (origin - target) % self.total
        return 1 if cw <= ccw else -1

    def origin_for_edge(self, edge: str, intensity: float) -> Origin:
        radius = self.config.normal_spill_radius
        if intensity >= self.config.level2_threshold:
            radius = self.total // 2
        elif intensity >= self.config.level1_threshold:
            radius = self.config.strong_spill_radius

        if edge == "left":
            direction = self.direction_from_to(self.config.tv_left_boundary, "left")
            return Origin(self.config.tv_left_boundary, (direction,), radius, "screen-left")
        if edge == "right":
            direction = self.direction_from_to(self.config.tv_right_boundary, "right")
            return Origin(self.config.tv_right_boundary, (direction,), radius, "screen-right")
        if edge == "bottom" and self.config.bottom_edge_mode != "room":
            return Origin(self.config.tv_center_led, (-1, 1), self.config.normal_spill_radius, "screen-bottom")
        return Origin(self.config.tv_center_led, (-1, 1), radius, "screen-top")
