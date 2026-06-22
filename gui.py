from __future__ import annotations

import copy
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from config import EngineConfig
from logger import setup_logging
from effects_engine import HeadlessEffectsEngine, RuntimeSnapshot
from topology import RoomTopology
from visualization import Visualizer


class ConfigVar:
    def __init__(self, widget_var: tk.Variable, cast: type | str):
        self.var = widget_var
        self.cast = cast

    def get(self) -> object:
        value = self.var.get()
        if self.cast is bool:
            return bool(value)
        if self.cast is int:
            return int(value)
        if self.cast is float:
            return float(value)
        if self.cast == "csv":
            return [item.strip() for item in str(value).split(",") if item.strip()]
        return str(value)

    def set(self, value: object) -> None:
        if isinstance(value, list):
            self.var.set(", ".join(str(item) for item in value))
        else:
            self.var.set(value)


class CinematicSpillGUI:
    def __init__(self, root: tk.Tk, config_path: str = "config.json"):
        self.root = root
        self.root.title("HyperHDR Cinematic Light Spill Engine")
        self.root.geometry("1320x880")
        self.config_path = Path(config_path)
        self.config = EngineConfig.load(self.config_path)
        self.config.send_to_wled = False
        self.logger = setup_logging(self.config.log_level, self.config.log_file)
        self.runtime = HeadlessEffectsEngine(copy.deepcopy(self.config), self.logger)
        self.vars: dict[str, ConfigVar] = {}
        self.slider_vars: dict[str, ConfigVar] = {}
        self.slider_dirty: set[str] = set()
        self.loading_vars = False
        self.preview_images: dict[str, ImageTk.PhotoImage] = {}
        self.status_var = tk.StringVar(value="Stopped")
        self.stats_var = tk.StringVar(value="FPS 0.0 | Waves 0 | Buffer 0")
        self.events_var = tk.StringVar(value="Events: none")
        self.trigger_status_var = tk.StringVar(value="Currently triggering: none")
        self.validation_var = tk.StringVar(value="")

        self._build()
        self._load_config_to_vars()
        self._schedule_preview()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(toolbar, text="Start", command=self.start).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Load", command=self.load_config).pack(side=tk.LEFT, padx=(18, 0))
        ttk.Button(toolbar, text="Save", command=self.save_config).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Save As", command=self.save_config_as).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Reset Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT)

        ttk.Label(main, textvariable=self.validation_var, foreground="#a33").pack(fill=tk.X, pady=(0, 6))

        notebook = ttk.Notebook(main)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.connection_tab = ttk.Frame(notebook, padding=10)
        self.room_tab = ttk.Frame(notebook, padding=10)
        self.motion_tab = ttk.Frame(notebook, padding=10)
        self.wave_tab = ttk.Frame(notebook, padding=10)
        self.preview_tab = ttk.Frame(notebook, padding=10)
        notebook.add(self.connection_tab, text="Connection")
        notebook.add(self.room_tab, text="Room")
        notebook.add(self.motion_tab, text="Motion")
        notebook.add(self.wave_tab, text="Waves & Color")
        notebook.add(self.preview_tab, text="Preview")

        self._build_connection_tab()
        self._build_room_tab()
        self._build_motion_tab()
        self._build_wave_tab()
        self._build_preview_tab()

    def _field(self, parent: ttk.Frame, row: int, label: str, key: str, cast: type | str = str, width: int = 16) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        var = tk.StringVar()
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky=tk.W, pady=4)
        self.vars[key] = ConfigVar(var, cast)

    def _checkbox(self, parent: ttk.Frame, row: int, label: str, key: str) -> None:
        var = tk.BooleanVar()

        def changed() -> None:
            if key.startswith("enabled_effects."):
                self._apply_live_effect_toggle(key, bool(var.get()))

        ttk.Checkbutton(parent, text=label, variable=var, command=changed).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=4)
        self.vars[key] = ConfigVar(var, bool)

    def _combo(self, parent: ttk.Frame, row: int, label: str, key: str, values: list[str]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        var = tk.StringVar()
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=18)
        combo.grid(row=row, column=1, sticky=tk.W, pady=4)
        self.vars[key] = ConfigVar(var, str)

    def _section(self, parent: ttk.Frame, text: str, column: int) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=text, padding=10)
        frame.grid(row=0, column=column, sticky=tk.NW, padx=(0, 14), pady=(0, 12))
        return frame

    def _build_connection_tab(self) -> None:
        hyper = self._section(self.connection_tab, "HyperHDR Input", 0)
        self._field(hyper, 0, "Host", "hyperhdr_host", str, 22)
        self._combo(hyper, 1, "Input Mode", "hyperhdr_input_mode", ["websocket", "flatbuffers"])
        self._field(hyper, 2, "FlatBuffers Port", "hyperhdr_port", int)
        self._field(hyper, 3, "WebSocket Port", "hyperhdr_ws_port", int)
        self._field(hyper, 4, "WebSocket Path", "hyperhdr_ws_path", str, 18)
        self._field(hyper, 5, "Instance", "hyperhdr_instance", int)
        self._field(hyper, 6, "API Token", "hyperhdr_token", str, 38)
        self._field(hyper, 7, "Frame Timeout", "hyperhdr_frame_timeout", float)
        self._checkbox(hyper, 8, "Use simulation input", "simulate_input")

        wled = self._section(self.connection_tab, "WLED Output", 1)
        self._field(wled, 0, "WLED IP", "wled_ip", str, 22)
        self._field(wled, 1, "Timeout", "wled_timeout", float)
        self._field(wled, 2, "Segment ID (blank for all)", "wled_segment_id", str)
        self._field(wled, 3, "WLED FPS", "wled_fps", int)
        self._checkbox(wled, 4, "Enable WLED output", "send_to_wled")

        perf = self._section(self.connection_tab, "Performance", 2)
        self._field(perf, 0, "Target FPS", "target_fps", int)
        self._field(perf, 1, "Frame Buffer", "frame_buffer_size", int)
        self._field(perf, 2, "Analysis Width", "analysis_width", int)
        self._field(perf, 3, "Analysis Height", "analysis_height", int)
        self._field(perf, 4, "Max Waves", "max_active_waves", int)

    def _build_room_tab(self) -> None:
        values = ttk.Frame(self.room_tab)
        values.pack(side=tk.LEFT, fill=tk.Y)
        room = ttk.LabelFrame(values, text="Room & TV", padding=10)
        room.pack(anchor=tk.NW, pady=(0, 12))
        self._field(room, 0, "Total LEDs", "total_leds", int)
        self._field(room, 1, "TV Center", "tv_center_led", int)
        self._field(room, 2, "TV Left Boundary", "tv_left_boundary", int)
        self._field(room, 3, "TV Right Boundary", "tv_right_boundary", int)
        self._field(room, 4, "Clockwise Order", "clockwise_order", "csv", 34)

        walls = ttk.LabelFrame(values, text="Wall Ranges", padding=10)
        walls.pack(anchor=tk.NW)
        ttk.Label(walls, text="Wall").grid(row=0, column=0, sticky=tk.W, padx=(0, 12), pady=(0, 6))
        ttk.Label(walls, text="Start").grid(row=0, column=1, sticky=tk.W, padx=(0, 8), pady=(0, 6))
        ttk.Label(walls, text="End").grid(row=0, column=2, sticky=tk.W, pady=(0, 6))
        for wall in ("front", "left", "rear", "right"):
            row = {"front": 1, "left": 2, "rear": 3, "right": 4}[wall]
            ttk.Label(walls, text=wall.title()).grid(row=row, column=0, sticky=tk.W, padx=(0, 12), pady=4)
            start_var = tk.StringVar()
            end_var = tk.StringVar()
            ttk.Entry(walls, textvariable=start_var, width=10).grid(row=row, column=1, sticky=tk.W, padx=(0, 8), pady=4)
            ttk.Entry(walls, textvariable=end_var, width=10).grid(row=row, column=2, sticky=tk.W, pady=4)
            self.vars[f"{wall}_wall.start"] = ConfigVar(start_var, int)
            self.vars[f"{wall}_wall.end"] = ConfigVar(end_var, int)

        preview = ttk.LabelFrame(self.room_tab, text="Topology Preview", padding=10)
        preview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(20, 0))
        self.topology_label = ttk.Label(preview)
        self.topology_label.pack(fill=tk.BOTH, expand=True)
        ttk.Button(preview, text="Refresh Preview", command=self.refresh_topology_preview).pack(anchor=tk.CENTER, pady=(8, 0))

    def _build_motion_tab(self) -> None:
        core = self._section(self.motion_tab, "Motion Detection", 0)
        self._combo(core, 0, "Algorithm", "motion_algorithm", ["optical_flow", "frame_difference"])
        self._field(core, 1, "Motion History", "motion_history", int)
        self._field(core, 2, "Minimum Motion", "min_motion", float)
        self._checkbox(core, 3, "Enable Bottom Edge", "bottom_edge_enabled")
        self._combo(core, 4, "Bottom Edge Mode", "bottom_edge_mode", ["local", "room"])
        self._field(core, 5, "Flash Changed Fraction", "flash_changed_fraction", float)
        self._field(core, 6, "Explosion Brightness Spike", "explosion_brightness_spike", float)
        self._field(core, 7, "Camera Pan Confidence", "camera_pan_confidence", float)

        weights = self._section(self.motion_tab, "Event Weights", 1)
        for row, key in enumerate(("motion", "direction", "brightness", "saturation", "coverage", "edge", "change")):
            self._field(weights, row, key.title(), f"weights.{key}", float)

        effects = self._section(self.motion_tab, "Enabled Effects", 2)
        self._checkbox(effects, 0, "Front ambient extension", "enabled_effects.front_ambient")
        self._checkbox(effects, 1, "Ambient side spill", "enabled_effects.ambient_side_spill")
        self._checkbox(effects, 2, "Motion spill waves", "enabled_effects.spill")
        self._checkbox(effects, 3, "Top color exit", "enabled_effects.top_color_exit")
        self._checkbox(effects, 4, "Screen flash pulses", "enabled_effects.flash")
        self._checkbox(effects, 5, "Explosion waves", "enabled_effects.explosion")
        self._checkbox(effects, 6, "Camera pan waves", "enabled_effects.camera_pan")
        self._checkbox(effects, 7, "Energy trails", "enabled_effects.energy_trail")
        self._checkbox(effects, 8, "Shockwaves", "enabled_effects.shockwave")
        self._checkbox(effects, 9, "Directional sweeps", "enabled_effects.directional_sweep")
        self._checkbox(effects, 10, "Lightning bursts", "enabled_effects.lightning")
        self._checkbox(effects, 11, "Impact pulses", "enabled_effects.impact_pulse")
        self._checkbox(effects, 12, "Color blooms", "enabled_effects.color_bloom")
        self._checkbox(effects, 13, "Flame shimmer", "enabled_effects.flame_shimmer")
        self._checkbox(effects, 14, "Underwater caustics", "enabled_effects.underwater")
        self._checkbox(effects, 15, "Portal vortex", "enabled_effects.portal_vortex")
        self._checkbox(effects, 16, "Scene wipes", "enabled_effects.scene_wipe")
        self._checkbox(effects, 17, "Ember particles", "enabled_effects.ember_particles")
        self._checkbox(effects, 18, "Negative waves", "enabled_effects.negative_wave")

    def _build_wave_tab(self) -> None:
        waves = self._section(self.wave_tab, "Spill & Waves", 0)
        self._combo(waves, 0, "Lighting Mode", "lighting_mode", ["cinematic", "front_ambient"])
        self._field(waves, 1, "Normal Radius", "normal_spill_radius", int)
        self._field(waves, 2, "Strong Radius", "strong_spill_radius", int)
        self._field(waves, 3, "Level 1 Threshold", "level1_threshold", float)
        self._field(waves, 4, "Level 2 Threshold", "level2_threshold", float)
        self._field(waves, 5, "Room Fill Threshold", "room_fill_threshold", float)
        self._field(waves, 6, "Side Major Threshold", "side_major_threshold", float)
        self._field(waves, 7, "Front Ambient Min Brightness", "front_ambient_min_brightness", float)
        self._field(waves, 8, "Front Ambient Intensity", "front_ambient_intensity", float)
        self._combo(waves, 9, "Front Ambient Source", "front_ambient_source", ["top_strip", "average"])
        self._field(waves, 10, "Top Strip Height", "front_ambient_top_height", float)
        self._field(waves, 11, "Top Strip Blur", "front_ambient_blur", int)
        self._field(waves, 12, "Wave Speed", "wave_speed", float)
        self._field(waves, 13, "Wave Decay", "wave_decay", float)
        self._field(waves, 14, "Wave Spread", "wave_spread", float)
        self._field(waves, 15, "Min Wave Intensity", "wave_min_intensity", float)
        self._field(waves, 16, "Color Velocity Speed", "color_velocity_speed_boost", float)
        self._field(waves, 17, "Color Velocity Decay", "color_velocity_decay_boost", float)
        self._field(waves, 18, "Top Exit Color Velocity", "top_color_exit_velocity_threshold", float)
        self._field(waves, 19, "Top Exit Motion", "top_color_exit_motion_threshold", float)
        self._field(waves, 20, "Top Exit Coverage", "top_color_exit_coverage_threshold", float)

        color = self._section(self.wave_tab, "Color Processing", 1)
        self._field(color, 0, "Brightness", "brightness", float)
        self._field(color, 1, "Gamma", "gamma", float)
        self._field(color, 2, "Saturation", "saturation", float)
        self._field(color, 3, "Color Smoothing", "color_smoothing", float)
        self._field(color, 4, "White Balance R", "white_balance.0", float)
        self._field(color, 5, "White Balance G", "white_balance.1", float)
        self._field(color, 6, "White Balance B", "white_balance.2", float)

    def _build_preview_tab(self) -> None:
        top = ttk.Frame(self.preview_tab)
        top.pack(fill=tk.X)
        ttk.Label(top, textvariable=self.stats_var).pack(side=tk.LEFT)
        ttk.Label(top, textvariable=self.events_var).pack(side=tk.RIGHT)
        trigger_status = ttk.LabelFrame(self.preview_tab, text="Current Trigger", padding=8)
        trigger_status.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(trigger_status, textvariable=self.trigger_status_var, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)

        grid = ttk.Frame(self.preview_tab)
        grid.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        controls = ttk.LabelFrame(grid, text="Trigger Controls", padding=6)
        controls.grid(row=0, column=3, rowspan=2, sticky=tk.NSEW, padx=6, pady=6)

        threshold_frame = ttk.LabelFrame(controls, text="Thresholds", padding=6)
        threshold_frame.pack(fill=tk.X, pady=(0, 8))
        self._slider(threshold_frame, 0, "Local Spill", "level1_threshold", 0.0, 1.0)
        self._slider(threshold_frame, 1, "Strong Spill", "level2_threshold", 0.0, 1.0)
        self._slider(threshold_frame, 2, "Room Fill", "room_fill_threshold", 0.0, 1.0)
        self._slider(threshold_frame, 3, "Side Major", "side_major_threshold", 0.0, 1.0)
        self._slider(threshold_frame, 4, "Front Ambient", "front_ambient_intensity", 0.0, 1.0)
        self._slider(threshold_frame, 5, "Flash Area", "flash_changed_fraction", 0.0, 1.0)
        self._slider(threshold_frame, 6, "Explosion Spike", "explosion_brightness_spike", 0.0, 1.0)
        self._slider(threshold_frame, 7, "Camera Pan", "camera_pan_confidence", 0.0, 1.0)
        self._slider(threshold_frame, 8, "Min Motion", "min_motion", 0.0, 0.1)

        triggers = ttk.LabelFrame(controls, text="Simulation Triggers", padding=6)
        triggers.pack(fill=tk.X)
        for row, (label, effect) in enumerate(
            (
                ("Left Exit", "left_exit"),
                ("Right Exit", "right_exit"),
                ("Top Spill", "top_spill"),
                ("Top Color Exit Left", "top_color_exit_left"),
                ("Top Color Exit Right", "top_color_exit_right"),
                ("Explosion", "explosion"),
                ("Screen Flash", "flash"),
                ("Shockwave", "shockwave"),
                ("Camera Pan Left", "pan_left"),
                ("Camera Pan Right", "pan_right"),
                ("Energy Trail", "energy_trail"),
                ("Directional Sweep", "directional_sweep"),
                ("Lightning", "lightning"),
                ("Impact Pulse", "impact_pulse"),
                ("Color Bloom", "color_bloom"),
                ("Flame Shimmer", "flame_shimmer"),
                ("Underwater", "underwater"),
                ("Portal Vortex", "portal_vortex"),
                ("Scene Wipe", "scene_wipe"),
                ("Ember Particles", "ember_particles"),
                ("Negative Wave", "negative_wave"),
            )
        ):
            ttk.Button(triggers, text=label, command=lambda name=effect: self.trigger_simulation(name)).grid(row=row, column=0, sticky=tk.EW, pady=3)
        triggers.columnconfigure(0, weight=1)

        self.preview_labels: dict[str, ttk.Label] = {}
        for idx, name in enumerate(("Frame", "Optical Flow", "Motion Heatmap", "LED Strip", "Square Room")):
            panel = ttk.LabelFrame(grid, text=name, padding=6)
            panel.grid(row=idx // 3, column=idx % 3, sticky=tk.NSEW, padx=6, pady=6)
            label = ttk.Label(panel)
            label.pack(fill=tk.BOTH, expand=True)
            self.preview_labels[name] = label
        history_panel = ttk.LabelFrame(grid, text="Incoming Frame History", padding=6)
        history_panel.grid(row=2, column=0, columnspan=4, sticky=tk.EW, padx=6, pady=6)
        self.frame_history_label = ttk.Label(history_panel)
        self.frame_history_label.pack(fill=tk.X)
        for col in range(3):
            grid.columnconfigure(col, weight=1)
        grid.columnconfigure(3, weight=0)
        for row in range(3):
            grid.rowconfigure(row, weight=1)

    def _slider(self, parent: ttk.Frame, row: int, label: str, key: str, start: float, end: float) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 8), pady=3)
        var = tk.DoubleVar()
        value_var = tk.StringVar(value="0.00")

        def changed(value: str, *, config_key: str = key, display: tk.StringVar = value_var) -> None:
            number = float(value)
            display.set(f"{number:.3f}" if end <= 0.1 else f"{number:.2f}")
            if not self.loading_vars:
                self.slider_dirty.add(config_key)
            self._apply_live_threshold(config_key, number)

        slider = ttk.Scale(parent, from_=start, to=end, orient=tk.HORIZONTAL, variable=var, command=changed, length=150)
        slider.grid(row=row, column=1, sticky=tk.EW, pady=3)
        ttk.Label(parent, textvariable=value_var, width=6).grid(row=row, column=2, sticky=tk.E, pady=3)
        parent.columnconfigure(1, weight=1)
        self.slider_vars[key] = ConfigVar(var, float)

    def _load_config_to_vars(self) -> None:
        data = self.config.to_dict()
        self.loading_vars = True
        try:
            for key, config_var in self.vars.items():
                if key.startswith("weights."):
                    config_var.set(data["weights"][key.split(".", 1)[1]])
                elif key.startswith("enabled_effects."):
                    config_var.set(data["enabled_effects"][key.split(".", 1)[1]])
                elif key.startswith("white_balance."):
                    config_var.set(data["white_balance"][int(key.rsplit(".", 1)[1])])
                elif "_wall." in key:
                    wall, attr = key.split(".", 1)
                    config_var.set(data[wall][attr])
                elif key == "wled_segment_id":
                    config_var.set("" if data[key] is None else data[key])
                else:
                    config_var.set(data[key])
            for key, config_var in self.slider_vars.items():
                config_var.set(data[key])
            self.slider_dirty.clear()
        finally:
            self.loading_vars = False
        self.refresh_topology_preview()

    def _config_from_vars(self) -> EngineConfig:
        data = self.config.to_dict()
        for key, config_var in self.vars.items():
            value = config_var.get()
            if key == "wled_segment_id":
                data[key] = None if str(value).strip() == "" else int(value)
            elif key.startswith("weights."):
                data["weights"][key.split(".", 1)[1]] = float(value)
            elif key.startswith("enabled_effects."):
                data["enabled_effects"][key.split(".", 1)[1]] = bool(value)
            elif key.startswith("white_balance."):
                data["white_balance"][int(key.rsplit(".", 1)[1])] = float(value)
            elif "_wall." in key:
                wall, attr = key.split(".", 1)
                data[wall][attr] = int(value)
            else:
                data[key] = value
        for key, config_var in self.slider_vars.items():
            if key not in self.slider_dirty:
                continue
            data[key] = config_var.get()
        return EngineConfig.from_dict(data)

    def _apply_live_threshold(self, key: str, value: float) -> None:
        try:
            if key in self.vars:
                self.vars[key].set(value)
            setattr(self.config, key, value)
            if self.runtime.running:
                setattr(self.runtime.config, key, value)
                if self.runtime.event_detector:
                    setattr(self.runtime.event_detector.config, key, value)
                if self.runtime.wave_engine:
                    setattr(self.runtime.wave_engine.config, key, value)
        except Exception:
            pass

    def _apply_live_effect_toggle(self, key: str, value: bool) -> None:
        try:
            effect = key.split(".", 1)[1]
            self.config.enabled_effects[effect] = value
            if self.runtime.running:
                self.runtime.config.enabled_effects[effect] = value
                if self.runtime.event_detector:
                    self.runtime.event_detector.config.enabled_effects[effect] = value
        except Exception:
            pass

    def start(self) -> None:
        try:
            self.config = self._config_from_vars()
            errors = self.config.validate()
            if errors:
                self.validation_var.set("; ".join(errors))
                return
            self.validation_var.set("")
            self.runtime.restart(copy.deepcopy(self.config))
            if not self.runtime.running:
                self.runtime.start()
            self.status_var.set("Running")
        except Exception as exc:
            self.status_var.set("Failed to start")
            messagebox.showerror("Start failed", str(exc))

    def stop(self) -> None:
        self.runtime.stop()
        self.status_var.set("Stopped")

    def trigger_simulation(self, effect: str) -> None:
        try:
            if not self.runtime.running:
                self.vars["simulate_input"].set(True)
                self.config = self._config_from_vars()
                self.config.simulate_input = True
                self.config.send_to_wled = bool(self.vars["send_to_wled"].get())
                errors = self.config.validate()
                if errors:
                    self.validation_var.set("; ".join(errors))
                    return
                self.runtime.restart(copy.deepcopy(self.config))
                if not self.runtime.running:
                    self.runtime.start()
            elif not self.runtime.config.simulate_input:
                self.validation_var.set("Simulation triggers are only available when simulation input is enabled.")
                return
            if not self.runtime.trigger_simulation_effect(effect):
                self.validation_var.set("Unable to trigger simulation effect.")
                return
            self.validation_var.set("")
            self.status_var.set(f"Triggered simulation: {effect}")
        except Exception as exc:
            messagebox.showerror("Simulation trigger failed", str(exc))

    def load_config(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON config", "*.json"), ("All files", "*.*")])
        if not path:
            return
        self.config_path = Path(path)
        self.config = EngineConfig.load(self.config_path)
        self.config.send_to_wled = False
        self._load_config_to_vars()
        self.status_var.set(f"Loaded {self.config_path.name}")

    def save_config(self) -> None:
        try:
            config = self._config_from_vars()
            errors = config.validate()
            if errors:
                self.validation_var.set("; ".join(errors))
                return
            self.validation_var.set("")
            config.save(self.config_path)
            self.config = config
            self.refresh_topology_preview()
            if self.runtime.running:
                self.runtime.restart(copy.deepcopy(self.config))
                self.status_var.set("Saved and restarted")
            else:
                self.status_var.set(f"Saved {self.config_path.name}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def save_config_as(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON config", "*.json")])
        if not path:
            return
        self.config_path = Path(path)
        self.save_config()

    def reset_defaults(self) -> None:
        if not messagebox.askyesno("Reset defaults", "Reset the form to default settings?"):
            return
        self.config = EngineConfig()
        self.config.send_to_wled = False
        self._load_config_to_vars()

    def refresh_topology_preview(self) -> None:
        try:
            config = self._config_from_vars()
            image = self._square_room_image(config, 520, 380)
            self._set_label_image(self.topology_label, image, "topology")
        except Exception as exc:
            self.validation_var.set(str(exc))

    def _schedule_preview(self) -> None:
        snapshot = self.runtime.step_once(timeout=0.001) if self.runtime.running else self.runtime.latest_snapshot()
        self._update_status(snapshot)
        self._update_preview(snapshot)
        delay = int(1000 / max(1, min(30, self.config.target_fps)))
        self.root.after(delay, self._schedule_preview)

    def _update_status(self, snapshot: RuntimeSnapshot) -> None:
        if snapshot.running:
            source = "Simulation" if self.runtime.config.simulate_input else ("HyperHDR connected" if snapshot.hyperhdr_connected else "HyperHDR reconnecting")
            wled = "WLED enabled" if snapshot.wled_enabled else "WLED preview only"
            self.status_var.set(f"Running | {source} | {wled}")
        self.stats_var.set(f"FPS {snapshot.fps:.1f} | Waves {snapshot.active_waves} | Buffer {snapshot.buffered_frames}")
        if snapshot.events:
            self.events_var.set("Events: " + ", ".join(f"{e.kind}:{e.edge}:{e.intensity:.2f}:cv{e.color_velocity:.2f}" for e in snapshot.events[:4]))
            self.trigger_status_var.set("Currently triggering: " + " | ".join(f"{e.kind.upper()} {e.edge} {e.intensity:.2f} cv {e.color_velocity:.2f}" for e in snapshot.events[:3]))
        elif snapshot.last_error:
            self.events_var.set(f"Error: {snapshot.last_error}")
            self.trigger_status_var.set("Currently triggering: error")
        else:
            self.events_var.set("Events: none")
            self.trigger_status_var.set("Currently triggering: none")

    def _update_preview(self, snapshot: RuntimeSnapshot) -> None:
        if snapshot.frame is None or snapshot.leds is None:
            self._set_label_image(self.preview_labels["Square Room"], self._square_room_image(self.config, 360, 260, snapshot.leds), "room_live")
            if snapshot.frame_history:
                self._set_label_image(self.frame_history_label, self._frame_history_image(snapshot.frame_history, 980, 120), "frame_history")
            return
        frame = snapshot.frame
        analysis = snapshot.analysis
        zone_frame = frame.copy()
        Visualizer._draw_zones(zone_frame)
        self._set_label_image(self.preview_labels["Frame"], self._cv_to_image(zone_frame, 360, 210), "frame")
        if analysis is not None:
            visualizer = Visualizer(self.config)
            flow = visualizer._flow_view(frame, analysis)
            heat = visualizer._heatmap_view(frame, analysis)
            self._set_label_image(self.preview_labels["Optical Flow"], self._cv_to_image(flow, 360, 210), "flow")
            self._set_label_image(self.preview_labels["Motion Heatmap"], self._cv_to_image(heat, 360, 210), "heat")
        self._set_label_image(self.preview_labels["LED Strip"], self._led_strip_image(snapshot.leds, 360, 210), "strip")
        self._set_label_image(self.preview_labels["Square Room"], self._square_room_image(self.runtime.config, 360, 260, snapshot.leds), "room_live")
        self._set_label_image(self.frame_history_label, self._frame_history_image(snapshot.frame_history, 980, 120), "frame_history")

    def _cv_to_image(self, bgr: np.ndarray, width: int, height: int) -> Image.Image:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        return image

    def _led_strip_image(self, leds: np.ndarray, width: int, height: int) -> Image.Image:
        image = Image.new("RGB", (width, height), "black")
        draw = ImageDraw.Draw(image)
        count = max(1, len(leds))
        for idx, color in enumerate(leds.astype(int).tolist()):
            x0 = int(idx * width / count)
            x1 = int((idx + 1) * width / count) + 1
            draw.rectangle((x0, 0, x1, height), fill=tuple(color))
        return image

    def _frame_history_image(self, frames: list[np.ndarray], width: int, height: int) -> Image.Image:
        image = Image.new("RGB", (width, height), (12, 14, 18))
        if not frames:
            draw = ImageDraw.Draw(image)
            draw.text((12, 12), "No incoming frames yet", fill=(180, 190, 200))
            return image
        count = min(len(frames), 12)
        gap = 6
        thumb_w = max(20, (width - gap * (count + 1)) // count)
        thumb_h = height - gap * 2
        draw = ImageDraw.Draw(image)
        for idx, frame in enumerate(frames[-count:]):
            thumb = self._cv_to_image(frame, thumb_w, thumb_h)
            x = gap + idx * (thumb_w + gap)
            y = gap + (thumb_h - thumb.height) // 2
            image.paste(thumb, (x, y))
            draw.rectangle((x, y, x + thumb.width, y + thumb.height), outline=(58, 68, 82), width=1)
            draw.text((x + 4, y + 4), str(len(frames) - count + idx + 1), fill=(230, 230, 230))
        return image

    def _square_room_image(self, config: EngineConfig, width: int, height: int, leds: np.ndarray | None = None) -> Image.Image:
        image = Image.new("RGB", (width, height), (18, 20, 24))
        draw = ImageDraw.Draw(image)
        margin = 42
        side = min(width - margin * 2, height - margin * 2)
        left = (width - side) // 2
        top = (height - side) // 2 + 8
        right = left + side
        bottom = top + side
        topology = RoomTopology(config)
        wall_colors = {
            "front": (80, 170, 255),
            "left": (120, 255, 150),
            "rear": (255, 210, 80),
            "right": (255, 110, 130),
        }

        draw.rectangle((left, top, right, bottom), outline=(68, 76, 90), width=2)
        for wall in ("front", "left", "rear", "right"):
            wall_leds = topology.range_inclusive(getattr(config, f"{wall}_wall").start, getattr(config, f"{wall}_wall").end)
            for offset, led in enumerate(wall_leds):
                x, y = self._square_wall_point(config, wall, offset, len(wall_leds), left, top, right, bottom)
                color = tuple(int(v) for v in leds[led]) if leds is not None else wall_colors[wall]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

        for wall in ("front", "left", "rear", "right"):
            label_x, label_y = self._square_wall_point(config, wall, 1, 3, left, top, right, bottom, inset=18)
            draw.text((label_x - 16, label_y - 8), wall.title(), fill=wall_colors[wall])

        for label, led, color in (
            ("L", config.tv_left_boundary, (255, 255, 255)),
            ("C", config.tv_center_led, (255, 255, 255)),
            ("R", config.tv_right_boundary, (255, 255, 255)),
        ):
            wall = topology.wall_for_led(led) or "front"
            wall_leds = topology.range_inclusive(getattr(config, f"{wall}_wall").start, getattr(config, f"{wall}_wall").end)
            try:
                offset = wall_leds.index(led)
            except ValueError:
                offset = len(wall_leds) // 2
            x, y = self._square_wall_point(config, wall, offset, len(wall_leds), left, top, right, bottom, inset=16)
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline=color, width=2)
            draw.text((x - 4, y - 7), label, fill=color)

        draw.text((12, 10), "Square room perimeter", fill=(230, 230, 230))
        if leds is None:
            draw.text((12, 30), "Blue=front Green=left Yellow=rear Red=right", fill=(180, 190, 200))
        return image

    @staticmethod
    def _square_wall_point(
        config: EngineConfig,
        wall: str,
        offset: int,
        count: int,
        left: int,
        top: int,
        right: int,
        bottom: int,
        inset: int = 0,
    ) -> tuple[int, int]:
        denominator = max(1, count - 1)
        t = offset / denominator
        start, end = CinematicSpillGUI._square_wall_segment(config, wall, left, top, right, bottom, inset)
        return (
            int(start[0] + t * (end[0] - start[0])),
            int(start[1] + t * (end[1] - start[1])),
        )

    @staticmethod
    def _square_wall_segment(
        config: EngineConfig,
        wall: str,
        left: int,
        top: int,
        right: int,
        bottom: int,
        inset: int = 0,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        # The configured wall order is the physical LED continuity. For the
        # default front,left,rear,right order, this maps front end -> left start,
        # left end -> rear start, rear end -> right start, and right end -> front start.
        corners = [
            (right - inset, top + inset),
            (left + inset, top + inset),
            (left + inset, bottom - inset),
            (right - inset, bottom - inset),
        ]
        order = config.clockwise_order
        idx = order.index(wall) if wall in order else 0
        return corners[idx], corners[(idx + 1) % len(corners)]

    def _set_label_image(self, label: ttk.Label, image: Image.Image, key: str) -> None:
        photo = ImageTk.PhotoImage(image)
        self.preview_images[key] = photo
        label.configure(image=photo)

    def _on_close(self) -> None:
        self.runtime.stop()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    CinematicSpillGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
