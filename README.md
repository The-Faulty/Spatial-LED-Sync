# HyperHDR Cinematic Light Spill Engine

A Python intermediary between HyperHDR and WLED that treats ceiling LEDs as a spatial room-lighting system, not as a direct Ambilight mirror.

The engine analyzes full image frames, detects motion and edge energy, creates cinematic spill events, and propagates soft additive waves around a configurable circular LED perimeter.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py --simulate
```

Use `q` in the debug window to quit.

Launch the desktop configuration and preview GUI with:

```powershell
python gui.py
```

Launch the desktop 3D room editor with:

```powershell
python spatial_editor_server.py
```

Then open `http://127.0.0.1:8787` from a desktop browser. The editor writes the `spatial`
room/device/strip section back to `config.json`. The Raspberry Pi target is the headless
lighting runtime, not the local 3D editing viewport.

Quick launch scripts are also included:

- `Launch GUI.bat` / `Launch GUI.ps1`: desktop configuration and preview GUI.
- `Launch 3D Room Editor.bat` / `Launch 3D Room Editor.ps1`: desktop browser-based room, TV, strip, and device editor.
- `Launch Simulation Preview.bat` / `Launch Simulation Preview.ps1`: CLI simulation with OpenCV debug preview.
- `Launch Live Engine.bat` / `Launch Live Engine.ps1`: live headless engine; WLED output follows `config.json`.
- `Launch Pi Zero Live Engine.bat` / `Launch Pi Zero Live Engine.ps1`: headless runtime using the Pi Zero performance profile; WLED output follows `config.json`.
- `Launch Pi 5 Live Engine.bat` / `Launch Pi 5 Live Engine.ps1`: headless runtime using the Pi 5 performance profile; WLED output follows `config.json`.
- `Run Pi Zero Benchmark.bat` / `Run Pi Zero Benchmark.ps1`: 15-second simulated benchmark using the Pi Zero profile.
- `Run Pi 5 Benchmark.bat` / `Run Pi 5 Benchmark.ps1`: 15-second simulated benchmark using the Pi 5 profile.

In the GUI preview tab, simulation is idle by default. Use the Simulation Triggers buttons to fire specific test effects such as left/right exits, top spill, explosion, screen flash, shockwave, camera pans, and energy trail.

## Configuration

Edit `config.json`.

Important fields:

- `hyperhdr_host`, `hyperhdr_port`: HyperHDR FlatBuffers server.
- `hyperhdr_input_mode`: use `websocket` for HyperHDR live-video frames, or `flatbuffers` only for explicit FlatBuffers diagnostics.
- `hyperhdr_ws_port`, `hyperhdr_ws_path`: HyperHDR WebSocket endpoint, usually port `8090` and path `/json-rpc`.
- `hyperhdr_instance`: HyperHDR instance id, usually `0`.
- `hyperhdr_token`: optional HyperHDR API token used to authorize the WebSocket session.
- `simulate_input`: keep `true` for development without HyperHDR; set `false` for live input.
- `wled_ip`: WLED controller address.
- `send_to_wled`: set `true` to emit LED frames.
- `runtime_profile`: `desktop_dev`, `pi_zero`, or `pi_5`.
- `overload_policy`: `adaptive_quality`, `fixed_quality`, or `output_first`.
- `spatial`: optional 3D room model with room dimensions, TV placement, WLED devices, and wall-mounted strips.
- `total_leds`: total ceiling perimeter LEDs.
- `front_wall`, `left_wall`, `rear_wall`, `right_wall`: inclusive LED ranges; ranges may wrap across index `0`.
- `clockwise_order`: physical wall order.
- `tv_center_led`, `tv_left_boundary`, `tv_right_boundary`: spill origins on the front wall.
- `motion_algorithm`: `optical_flow` or `frame_difference`.
- `lighting_mode`: `cinematic` keeps the original sparse spill behavior; `front_ambient` makes the front wall behave more like a soft TV extension while side/rear walls stay reserved for major effects.
- `front_ambient_source`: `top_strip` mirrors the top band of the incoming video onto the TV span; `average` uses the older single-color ambient wash.
- `front_ambient_top_height`: fraction of the video height sampled for the top ambient strip.
- `enabled_effects`: per-effect switches for front ambient, ambient side spill, spill, top color exit, flash, explosion, camera pan, energy trail, shockwave, directional sweep, lightning, impact pulse, color bloom, flame shimmer, underwater caustics, portal vortex, scene wipe, ember particles, and negative wave events.
- `color_velocity_speed_boost`, `color_velocity_decay_boost`: tune how fast color changes affect wave speed and fade.
- `top_color_exit_velocity_threshold`, `top_color_exit_motion_threshold`, `top_color_exit_coverage_threshold`: control when fast color motion along the top band spills out through the left or right room edge.

Example live run:

```powershell
python main.py --config config.json --wled --no-debug --profile pi_zero --overload-policy adaptive_quality
```

Benchmark the Pi-oriented headless runtime without WLED output:

```powershell
python benchmark.py --profile pi_zero --seconds 15
```

## Architecture

- `hyperhdr_client.py`: connects to HyperHDR FlatBuffers, buffers frames, reconnects automatically, and provides simulation mode.
- `frame_processor.py`: resizes frames, computes color and zone statistics.
- `motion_detector.py`: optical-flow or frame-difference analysis, edge direction, motion coverage, heatmaps.
- `event_detector.py`: converts motion analysis into normalized cinematic events.
- `topology.py`: circular room model with wrapping wall ranges and configurable adjacency.
- `spatial_config.py`: 3D room, TV, device, and strip config model.
- `spatial_topology.py`: precomputed per-LED 3D coordinates, wall coordinates, and device routing.
- `spatial_renderer.py`: vectorized spatial renderer used by spatial-enabled configs.
- `spatial_editor_server.py` / `web_editor/`: desktop browser-based Three.js room editor.
- `benchmark.py`: headless runtime benchmark for Pi profile tuning.
- `wave_engine.py`: multiple simultaneous additive waves with decay, spread, smoothing, color processing, and radius limits.
- `wled_output.py`: rate-limited WLED JSON API output with legacy single-device and spatial multi-device routing.
- `visualization.py`: debug display for frames, flow vectors, heatmap, zones, event stats, and simulated strip.
- `runtime.py`: reusable engine controller shared by the CLI and GUI.
- `gui.py`: Tkinter desktop GUI for configuration, start/stop control, diagnostics, and square-room preview.
- `main.py`: runtime loop, logging, lifecycle, and performance statistics.

## Motion Detection

The screen is divided into center, top, left, right, and bottom zones. Consecutive frames are analyzed with Farneback optical flow by default. Each edge receives:

- motion magnitude
- direction confidence
- outward or edge-bound motion
- parallel motion
- changed area coverage
- dominant edge color
- color velocity, measured as normalized color change from the previous frame

Spill is generated mainly when motion reaches or moves toward a screen edge. Static frames and slow color changes produce little or no ceiling output.

The top-left and top-right corners are analyzed separately. When a color moves horizontally across the top band and reaches a side edge with enough color velocity, the engine emits a side spill from that edge even if the full side zone is otherwise quiet.

## Event Intensity

Events are scored from `0.0` to `1.0` using configurable weights:

- motion magnitude
- direction confidence
- brightness
- saturation
- affected screen percentage
- edge impact strength
- rate of change
- edge color velocity

Level 1 stays close to the TV, Level 2 reaches the front wall and corners, and Level 3 can fill the room.

In `front_ambient` mode, the TV span on the front wall receives a continuous top-strip sample from the current frame, similar to mirroring the TV's top Ambilight/backlight zone. Left, right, and bottom edge propagation is gated by `side_major_threshold`, so side walls remain mostly dark unless a stronger event occurs. Each effect can also be enabled or disabled individually.

Additional cinematic effects include:

- `ambient_side_spill`: subtle renderer-side bleed of the nearest parent strip edge color into spatial extension strips, fading quickly so side strips do not mirror the full TV edge.
- `energy_trail`: comet-like tails from fast bright objects exiting screen edges.
- `shockwave`: thin expanding rings from hard impacts or sudden cuts.
- `directional_sweep`: broad room washes from strong directional motion.
- `lightning`: short high-contrast strobe bursts.
- `impact_pulse`: whole-room blooms from sudden motion or brightness spikes.
- `color_bloom`: slow atmospheric washes from saturated stable scenes.
- `flame_shimmer`: warm flickering edge shimmer.
- `underwater`: blue/green caustic wavelets for calm scenes.
- `portal_vortex`: spiral movement from complex center-heavy flow.
- `scene_wipe`: room-wide directional wipes from fast scene transitions.
- `ember_particles`: small fading sparks after explosions, lightning, or flame.
- `negative_wave`: traveling dimming masks for dark transitions.

## Wave Propagation

Each event creates one or more `LightWave` objects with color, intensity, position, velocity, decay, spread, age, and direction. The physical room preview is square, while the propagation engine treats LEDs as a circular perimeter so waves can move clockwise or counterclockwise across index `0`.

Top-edge motion expands from the TV center in cinematic mode. In `front_ambient` mode, top-origin waves leave from `tv_left_boundary` and `tv_right_boundary` so they spill out of the ambient TV extension instead of painting over it. Left-edge motion starts near `tv_left_boundary` and propagates into the left wall. Right-edge motion starts near `tv_right_boundary` and propagates into the right wall. Major flashes and explosions create broader bidirectional waves.

Color velocity modifies each wave after detection. Fast color changes increase wave speed and slightly increase fade rate, giving color wipes and energetic transitions a quicker, sharper environmental response than stable colors.

## Performance Notes

- Lower `analysis_width` and `analysis_height` for Raspberry Pi targets.
- Use `frame_difference` if optical flow is too expensive.
- Keep `target_fps` and `wled_fps` near 25-30 for smooth output without flooding WLED.
- Limit `max_active_waves` to bound CPU cost.
- Run with `--no-debug` for live deployments.
- Disabled effects reduce analysis work. Ambient/bloom-only modes skip optical flow, flash/pulse-only modes use frame differencing without edge flow, and fully disabled event effects skip motion/event analysis while TV-image sync still works.

## HyperHDR Input Note

HyperHDR's FlatBuffers server is usually an input receiver for external grabbers, so it may accept a TCP connection and then wait for this app to send frames. For live captured frames, this app defaults to HyperHDR's WebSocket JSON-RPC endpoint at `/json-rpc` and sends:

```json
{"command":"authorize","subcommand":"login","token":"...","tan":100}
```

followed by:

```json
{"command":"ledcolors","subcommand":"imagestream-start","tan":1}
```

Incoming image frames may arrive as binary JPEG data, base64 strings, or JSON-wrapped image fields; the client handles all three.

The legacy FlatBuffers decoder remains available with `hyperhdr_input_mode: "flatbuffers"` for builds that provide a pushed frame stream or for diagnostics.

## Future Enhancements

- Add a bundled HyperHDR FlatBuffers schema generator.
- Learn per-room propagation timing from calibration patterns.
- Add audio-reactive secondary energy for impacts.
- Support multiple WLED controllers for large rooms.
- Add a web dashboard for live tuning thresholds and topology.
