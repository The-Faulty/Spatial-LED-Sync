from __future__ import annotations

from dataclasses import replace

from event_detector import LightEvent


EVENT_PRIORITIES = {
    "lightning": 100,
    "explosion": 95,
    "shockwave": 90,
    "impact_pulse": 85,
    "negative_wave": 80,
    "scene_wipe": 70,
    "directional_sweep": 65,
    "portal_vortex": 60,
    "camera_pan": 55,
    "energy_trail": 45,
    "top_color_exit": 40,
    "spill": 35,
    "flame_shimmer": 25,
    "underwater": 20,
    "color_bloom": 15,
}

BASE_EFFECTS = {"front_ambient"}
HIGH_ENERGY_EFFECTS = {"lightning", "explosion", "shockwave", "impact_pulse", "negative_wave"}
MOTION_PRIMARY_EFFECTS = {"scene_wipe", "directional_sweep", "portal_vortex", "camera_pan"}
EDGE_ACCENT_EFFECTS = {"energy_trail", "top_color_exit", "spill"}
MOOD_EFFECTS = {"flame_shimmer", "underwater", "color_bloom"}
SECONDARY_EFFECTS = {"ember_particles"}
SECONDARY_PARENTS = {"explosion", "lightning", "flame_shimmer"}
MOOD_TIE_BREAK = {"flame_shimmer": 3, "underwater": 2, "color_bloom": 1}


class EventMixer:
    def mix(self, events: list[LightEvent]) -> list[LightEvent]:
        if not events:
            return []
        base = [replace(event, primary=False, suppressed_by="") for event in events if event.kind in BASE_EFFECTS]
        candidates = [event for event in events if event.kind not in BASE_EFFECTS and event.kind not in SECONDARY_EFFECTS]
        secondary = [event for event in events if event.kind in SECONDARY_EFFECTS or event.secondary]
        primary = self._select_primary(candidates)
        if primary is None:
            return base + self._mix_without_primary(candidates)

        primary_event = replace(primary, primary=True, suppressed_by="")
        if primary.kind in HIGH_ENERGY_EFFECTS:
            return base + [primary_event] + self._allowed_secondary(primary_event, secondary)
        if primary.kind in MOTION_PRIMARY_EFFECTS:
            companion = self._best_motion_companion(primary_event, candidates)
            return base + [primary_event] + ([companion] if companion else [])
        return base + self._mix_without_primary(candidates)

    def _select_primary(self, events: list[LightEvent]) -> LightEvent | None:
        primary_candidates = [event for event in events if EVENT_PRIORITIES.get(event.effect_id or event.kind, 0) >= 55]
        if not primary_candidates:
            return None
        return max(primary_candidates, key=lambda event: (self._priority(event), event.intensity))

    def _priority(self, event: LightEvent) -> int:
        return EVENT_PRIORITIES.get(event.effect_id or event.kind, EVENT_PRIORITIES.get(event.kind, 0))

    def _allowed_secondary(self, primary: LightEvent, secondary: list[LightEvent]) -> list[LightEvent]:
        if primary.kind not in SECONDARY_PARENTS:
            return []
        return [replace(event, primary=False, suppressed_by="") for event in secondary if event.effect_id in SECONDARY_EFFECTS or event.kind in SECONDARY_EFFECTS]

    def _best_motion_companion(self, primary: LightEvent, events: list[LightEvent]) -> LightEvent | None:
        accents = [event for event in events if (event.effect_id or event.kind) in EDGE_ACCENT_EFFECTS]
        if not accents:
            return None

        def score(event: LightEvent) -> tuple[int, float]:
            same_direction = int(event.edge == primary.edge or (event.direction_hint != 0 and event.direction_hint == primary.direction_hint))
            return same_direction, event.intensity

        return replace(max(accents, key=score), primary=False, suppressed_by="")

    def _mix_without_primary(self, events: list[LightEvent]) -> list[LightEvent]:
        edge_accents = [event for event in events if (event.effect_id or event.kind) in EDGE_ACCENT_EFFECTS]
        mood = [event for event in events if (event.effect_id or event.kind) in MOOD_EFFECTS]
        result = [replace(event, primary=False, suppressed_by="") for event in sorted(edge_accents, key=lambda event: event.intensity, reverse=True)[:2]]
        selected_mood = self._select_mood(mood)
        if selected_mood:
            result.append(selected_mood)
        return result

    def _select_mood(self, events: list[LightEvent]) -> LightEvent | None:
        if not events:
            return None
        strongest = max(events, key=lambda event: event.intensity)
        close = [event for event in events if strongest.intensity - event.intensity <= 0.08]
        chosen = max(close, key=lambda event: (MOOD_TIE_BREAK.get(event.effect_id or event.kind, 0), event.intensity))
        return replace(chosen, primary=False, suppressed_by="")


def is_high_energy_primary(kind: str) -> bool:
    return kind in HIGH_ENERGY_EFFECTS


def priority_for_kind(kind: str) -> int:
    return EVENT_PRIORITIES.get(kind, 0)


def should_duck_wave(primary_kind: str, wave_kind: str) -> bool:
    if wave_kind == primary_kind:
        return False
    if wave_kind == "ember_particles" and primary_kind in SECONDARY_PARENTS:
        return False
    return priority_for_kind(wave_kind) < priority_for_kind(primary_kind)
