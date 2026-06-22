import { createLightPreview3d } from "./scene3d.js?v=three165-live10";

const PREVIEW_MAX_FPS = 90;
const PREVIEW_POLL_INTERVAL_MS = Math.max(1, Math.round(1000 / PREVIEW_MAX_FPS));

const statusEl = document.querySelector("#status");
const canvas = document.querySelector("#scene");
const ctx = canvas.getContext("2d");
const canvas3d = document.querySelector("#scene3d");
const webglFallback = document.querySelector("#webgl-fallback");

let config = null;
let spatial = null;
let previewColors = [];
let yaw = -0.72;
let pitch = -0.48;
let zoom = 130;
const projectionMode = "original";
let dragging = false;
let lastPointer = null;
let selectedStripId = null;
let stripHitTargets = [];
let editDrag = null;
let saveTimer = null;
let saveInFlight = false;
let saveAgain = false;
let previewRunning = false;
let previewTimer = null;
let drawScheduled = false;
let lastLedRevision = 0;
let previewFrameRevision = 0;
let previewFrameRequestRevision = 0;
let previewFrameLoading = false;
let previewImage = null;
let emptyDragStart = null;
let activeView = "2d";
let effectSaveTimer = null;
let simulationPatterns = [];
let loopingEffect = "";
let effectSaveInFlight = false;
let effectSaveAgain = false;
let pendingEffectOverrides = {};
let pendingSensitivityOverrides = {};
let pendingAmbientSpillSettings = {};
let previewRenderStats = { avgMs: 0, samples: 0, maxFps: PREVIEW_MAX_FPS };

const EFFECT_LABELS = {
  front_ambient: "Front ambient",
  ambient_side_spill: "Ambient side spill",
  ambient_side_spill_boost: "Ambient spill boost",
  spill: "Motion spill",
  top_color_exit: "Top color exit",
  flash: "Flash",
  explosion: "Explosion",
  camera_pan: "Camera pan",
  energy_trail: "Energy trail",
  shockwave: "Shockwave",
  directional_sweep: "Directional sweep",
  lightning: "Lightning",
  impact_pulse: "Impact pulse",
  color_bloom: "Color bloom",
  flame_shimmer: "Flame shimmer",
  underwater: "Underwater",
  portal_vortex: "Portal vortex",
  scene_wipe: "Scene wipe",
  ember_particles: "Ember particles",
  negative_wave: "Negative wave",
};

const TRIGGER_EFFECTS = [
  "spill",
  "top_color_exit",
  "flash",
  "explosion",
  "camera_pan",
  "energy_trail",
  "shockwave",
  "directional_sweep",
  "lightning",
  "impact_pulse",
  "color_bloom",
  "flame_shimmer",
  "underwater",
  "portal_vortex",
  "scene_wipe",
  "negative_wave",
];

const fields = {
  roomWidth: document.querySelector("#room-width"),
  roomDepth: document.querySelector("#room-depth"),
  roomHeight: document.querySelector("#room-height"),
  tvWall: document.querySelector("#tv-wall"),
  tvU: document.querySelector("#tv-u"),
  tvV: document.querySelector("#tv-v"),
  tvWidth: document.querySelector("#tv-width"),
  tvHeight: document.querySelector("#tv-height"),
  startPreview: document.querySelector("#start-preview"),
  stopPreview: document.querySelector("#stop-preview"),
  startSimulation: document.querySelector("#start-simulation"),
  stopSimulation: document.querySelector("#stop-simulation"),
  previewStatus: document.querySelector("#preview-status"),
  targetFps: document.querySelector("#target-fps"),
  wledFps: document.querySelector("#wled-fps"),
  renderMode: document.querySelector("#render-mode"),
  edgeBandWidth: document.querySelector("#edge-band-width"),
  edgeBandHeight: document.querySelector("#edge-band-height"),
  edgeBandFraction: document.querySelector("#edge-band-fraction"),
  hybridFullWidth: document.querySelector("#hybrid-full-width"),
  hybridFullHeight: document.querySelector("#hybrid-full-height"),
  effectToggles: document.querySelector("#effect-toggles"),
  effectSensitivity: document.querySelector("#effect-sensitivity"),
  ambientSideSpillBase: document.querySelector("#ambient-side-spill-base-intensity"),
  ambientSideSpillBaseValue: document.querySelector("#ambient-side-spill-base-value"),
  ambientSideSpillBoost: document.querySelector("#ambient-side-spill-boost-intensity"),
  ambientSideSpillBoostValue: document.querySelector("#ambient-side-spill-boost-value"),
  tvImageBlur: document.querySelector("#tv-image-blur"),
  tvImageBlurValue: document.querySelector("#tv-image-blur-value"),
  effectPatterns: document.querySelector("#effect-patterns"),
  stopPattern: document.querySelector("#stop-pattern"),
  triggeredEffects: document.querySelector("#triggered-effects"),
  activeEffects: document.querySelector("#active-effects"),
  view2d: document.querySelector("#view-2d"),
  view3d: document.querySelector("#view-3d"),
  devicesJson: document.querySelector("#devices-json"),
  stripsJson: document.querySelector("#strips-json"),
  stripSelect: document.querySelector("#strip-select"),
  stripName: document.querySelector("#strip-name"),
  stripWall: document.querySelector("#strip-wall"),
  stripStartU: document.querySelector("#strip-start-u"),
  stripStartV: document.querySelector("#strip-start-v"),
  stripEndU: document.querySelector("#strip-end-u"),
  stripEndV: document.querySelector("#strip-end-v"),
  stripLedCount: document.querySelector("#strip-led-count"),
  stripDirection: document.querySelector("#strip-direction"),
  stripDevice: document.querySelector("#strip-device"),
  stripDeviceStart: document.querySelector("#strip-device-start"),
  stripSyncMode: document.querySelector("#strip-sync-mode"),
  stripTvRole: document.querySelector("#strip-tv-role"),
  stripTvFillRow: document.querySelector("#strip-tv-fill-row"),
  stripTvFill: document.querySelector("#strip-tv-fill"),
  stripTvFillSpatialRow: document.querySelector("#strip-tv-fill-spatial-row"),
  stripTvFillSpatial: document.querySelector("#strip-tv-fill-spatial"),
  stripExtends: document.querySelector("#strip-extends"),
  stripExtensionMode: document.querySelector("#strip-extension-mode"),
  stripExtensionStrength: document.querySelector("#strip-extension-strength"),
  stripExtensionSoftness: document.querySelector("#strip-extension-softness"),
  stripBlend: document.querySelector("#strip-blend"),
  stripNudgeLeft: document.querySelector("#strip-nudge-left"),
  stripNudgeRight: document.querySelector("#strip-nudge-right"),
  stripNudgeUp: document.querySelector("#strip-nudge-up"),
  stripNudgeDown: document.querySelector("#strip-nudge-down"),
  deselectStrip: document.querySelector("#deselect-strip"),
  duplicateStrip: document.querySelector("#duplicate-strip"),
  deleteStrip: document.querySelector("#delete-strip"),
  addStrip: document.querySelector("#add-strip"),
  save: document.querySelector("#save"),
};

const lightPreview3d = createLightPreview3d({
  canvas: canvas3d,
  fallback: webglFallback,
  onRenderStats(stats) {
    previewRenderStats = stats;
  },
});

async function loadConfig() {
  statusEl.textContent = "Loading config";
  const response = await fetch("/api/config");
  const payload = await response.json();
  config = payload.config;
  spatial = payload.spatial;
  simulationPatterns = payload.simulation_patterns || [];
  bindForm();
  renderEffectToggles();
  renderEffectSensitivity();
  syncAmbientSpillControls();
  renderEffectPatterns();
  await loadPreviewColors();
  syncLightPreview3d();
  renderSpatial();
  statusEl.textContent = "Loaded";
}

async function loadPreviewColors() {
  const response = await fetch("/api/preview-colors");
  const payload = await response.json();
  previewColors = payload.colors || [];
  syncLightPreview3d();
}

function syncLightPreview3d() {
  lightPreview3d.setState({ spatial, previewColors, previewImage });
}

function effectLabel(effect) {
  return EFFECT_LABELS[effect] || effect.replaceAll("_", " ");
}

function renderEffectToggles() {
  fields.effectToggles.innerHTML = "";
  const enabled = config?.enabled_effects || {};
  for (const effect of Object.keys(enabled)) {
    const label = document.createElement("label");
    label.className = "effect-toggle";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = Boolean(enabled[effect]);
    checkbox.dataset.effect = effect;
    checkbox.addEventListener("change", () => {
      config.enabled_effects[effect] = checkbox.checked;
      pendingEffectOverrides[effect] = checkbox.checked;
      scheduleEffectSave();
    });
    const text = document.createElement("span");
    text.textContent = effectLabel(effect);
    label.append(checkbox, text);
    fields.effectToggles.append(label);
  }
}

function syncEffectToggleControls() {
  const enabled = config?.enabled_effects || {};
  for (const checkbox of fields.effectToggles.querySelectorAll("input[data-effect]")) {
    const effect = checkbox.dataset.effect;
    checkbox.checked = Boolean(enabled[effect]);
    checkbox.disabled = effectSaveInFlight;
  }
  syncEffectSensitivityControls();
  syncAmbientSpillControls();
}

function renderEffectSensitivity() {
  fields.effectSensitivity.innerHTML = "";
  for (const effect of TRIGGER_EFFECTS) {
    const row = document.createElement("label");
    row.className = "effect-sensitivity-row";
    const text = document.createElement("span");
    text.textContent = effectLabel(effect);
    const value = document.createElement("span");
    value.className = "effect-sensitivity-value";
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.05";
    slider.dataset.effectSensitivity = effect;
    slider.addEventListener("input", () => {
      const next = Number(slider.value);
      config.effect_sensitivity[effect] = next;
      pendingSensitivityOverrides[effect] = next;
      value.textContent = next.toFixed(2);
      scheduleEffectSave();
    });
    row.append(text, slider, value);
    fields.effectSensitivity.append(row);
  }
  syncEffectSensitivityControls();
}

function syncEffectSensitivityControls() {
  const sensitivity = config?.effect_sensitivity || {};
  for (const slider of fields.effectSensitivity.querySelectorAll("input[data-effect-sensitivity]")) {
    const effect = slider.dataset.effectSensitivity;
    const value = Number(sensitivity[effect] ?? 0.5);
    slider.value = String(value);
    slider.disabled = effectSaveInFlight;
    const display = slider.parentElement?.querySelector(".effect-sensitivity-value");
    if (display) display.textContent = value.toFixed(2);
  }
}

function syncAmbientSpillControls() {
  if (!config || !fields.ambientSideSpillBase || !fields.ambientSideSpillBoost || !fields.tvImageBlur) return;
  const base = Number(config.ambient_side_spill_base_intensity ?? 0.45);
  const boost = Number(config.ambient_side_spill_boost_intensity ?? 1.0);
  const blur = Math.max(0, Math.round(Number(config.tv_image_blur ?? 0)));
  fields.ambientSideSpillBase.value = String(base);
  fields.ambientSideSpillBoost.value = String(boost);
  fields.tvImageBlur.value = String(blur);
  fields.ambientSideSpillBase.disabled = effectSaveInFlight;
  fields.ambientSideSpillBoost.disabled = effectSaveInFlight;
  fields.tvImageBlur.disabled = effectSaveInFlight;
  fields.ambientSideSpillBaseValue.textContent = base.toFixed(2);
  fields.ambientSideSpillBoostValue.textContent = boost.toFixed(2);
  fields.tvImageBlurValue.textContent = String(blur);
}

function bindAmbientSpillControls() {
  if (!fields.ambientSideSpillBase || !fields.ambientSideSpillBoost || !fields.tvImageBlur) return;
  fields.ambientSideSpillBase.addEventListener("input", () => {
    const next = Number(fields.ambientSideSpillBase.value);
    config.ambient_side_spill_base_intensity = next;
    pendingAmbientSpillSettings.ambient_side_spill_base_intensity = next;
    fields.ambientSideSpillBaseValue.textContent = next.toFixed(2);
    scheduleEffectSave();
  });
  fields.ambientSideSpillBoost.addEventListener("input", () => {
    const next = Number(fields.ambientSideSpillBoost.value);
    config.ambient_side_spill_boost_intensity = next;
    pendingAmbientSpillSettings.ambient_side_spill_boost_intensity = next;
    fields.ambientSideSpillBoostValue.textContent = next.toFixed(2);
    scheduleEffectSave();
  });
  fields.tvImageBlur.addEventListener("input", () => {
    const next = Math.max(0, Math.round(Number(fields.tvImageBlur.value)));
    config.tv_image_blur = next;
    pendingAmbientSpillSettings.tv_image_blur = next;
    fields.tvImageBlurValue.textContent = String(next);
    scheduleEffectSave();
  });
}

function renderEffectPatterns() {
  fields.effectPatterns.innerHTML = "";
  for (const pattern of simulationPatterns) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "effect-pattern";
    button.classList.toggle("active", loopingEffect === pattern.effect);
    button.textContent = pattern.label || effectLabel(pattern.effect);
    button.addEventListener("click", () => togglePreviewPattern(pattern.effect));
    fields.effectPatterns.append(button);
  }
}

async function togglePreviewPattern(effect) {
  await triggerPreviewPattern(loopingEffect === effect ? "idle" : `loop:${effect}`);
}

async function triggerPreviewPattern(effect) {
  if (!previewRunning) {
    fields.previewStatus.textContent = "Start simulation before toggling test patterns";
    return;
  }
  const response = await fetch("/api/preview/trigger", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ effect }),
  });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    fields.previewStatus.textContent = `Pattern failed: ${payload.error || "unknown error"}`;
    return;
  }
  loopingEffect = payload.looping_effect || "";
  renderEffectPatterns();
  fields.previewStatus.textContent = loopingEffect ? `Looping pattern: ${effectLabel(loopingEffect)}` : "Pattern stopped";
  schedulePreviewPoll(PREVIEW_POLL_INTERVAL_MS);
}

function scheduleEffectSave() {
  if (effectSaveTimer) clearTimeout(effectSaveTimer);
  effectSaveTimer = setTimeout(saveEffectToggles, 120);
}

async function saveEffectToggles() {
  if (effectSaveInFlight) {
    effectSaveAgain = true;
    return;
  }
  effectSaveTimer = null;
  effectSaveInFlight = true;
  syncEffectToggleControls();
  const response = await fetch("/api/effects", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      enabled_effects: { ...(config.enabled_effects || {}), ...pendingEffectOverrides },
      effect_sensitivity: { ...(config.effect_sensitivity || {}), ...pendingSensitivityOverrides },
      ...pendingAmbientSpillSettings,
    }),
  });
  const payload = await response.json();
  effectSaveInFlight = false;
  if (!response.ok || !payload.ok) {
    statusEl.textContent = (payload.errors || ["Effect update failed"]).join("; ");
    pendingEffectOverrides = {};
    pendingSensitivityOverrides = {};
    pendingAmbientSpillSettings = {};
    renderEffectToggles();
    renderEffectSensitivity();
    syncAmbientSpillControls();
    return;
  }
  pendingEffectOverrides = {};
  pendingSensitivityOverrides = {};
  pendingAmbientSpillSettings = {};
  config.enabled_effects = payload.enabled_effects || config.enabled_effects;
  config.effect_sensitivity = payload.effect_sensitivity || config.effect_sensitivity;
  config.ambient_side_spill_base_intensity = payload.ambient_side_spill_base_intensity ?? config.ambient_side_spill_base_intensity;
  config.ambient_side_spill_boost_intensity = payload.ambient_side_spill_boost_intensity ?? config.ambient_side_spill_boost_intensity;
  config.tv_image_blur = payload.tv_image_blur ?? config.tv_image_blur;
  syncEffectToggleControls();
  syncAmbientSpillControls();
  statusEl.textContent = "Effects updated";
  if (effectSaveAgain) {
    effectSaveAgain = false;
    scheduleEffectSave();
  }
}

function renderEffectList(element, items, { counts = false } = {}) {
  element.innerHTML = "";
  if (!items.length) {
    element.textContent = "None";
    element.classList.add("empty");
    return;
  }
  element.classList.remove("empty");
  for (const item of items) {
    const pill = document.createElement("span");
    pill.className = "effect-pill";
    if (item.primary) pill.classList.add("primary");
    pill.textContent = counts
      ? `${effectLabel(item.effect)} x${item.count}`
      : `${item.primary ? "Primary: " : ""}${effectLabel(item.effect)} ${item.edge ? `(${item.edge})` : ""}`;
    element.append(pill);
  }
}

function updateLiveEffects(payload) {
  if (payload.enabled_effects && config) {
    config.enabled_effects = { ...payload.enabled_effects, ...pendingEffectOverrides };
    syncEffectToggleControls();
  }
  if (payload.effect_sensitivity && config) {
    config.effect_sensitivity = { ...payload.effect_sensitivity, ...pendingSensitivityOverrides };
    syncEffectSensitivityControls();
  }
  if (config) {
    if (payload.ambient_side_spill_base_intensity !== undefined) {
      config.ambient_side_spill_base_intensity = pendingAmbientSpillSettings.ambient_side_spill_base_intensity ?? payload.ambient_side_spill_base_intensity;
    }
    if (payload.ambient_side_spill_boost_intensity !== undefined) {
      config.ambient_side_spill_boost_intensity = pendingAmbientSpillSettings.ambient_side_spill_boost_intensity ?? payload.ambient_side_spill_boost_intensity;
    }
    syncAmbientSpillControls();
  }
  const triggered = Array.isArray(payload.triggered_effects)
    ? payload.triggered_effects.map((event) => ({
        effect: event.effect_id || event.kind,
        edge: event.edge,
        primary: Boolean(event.primary),
      }))
    : [];
  const activeCounts = payload.active_effect_counts || {};
  const active = Object.entries(activeCounts)
    .filter(([, count]) => Number(count) > 0)
    .map(([effect, count]) => ({ effect, count: Number(count) }))
    .sort((a, b) => b.count - a.count || effectLabel(a.effect).localeCompare(effectLabel(b.effect)));
  renderEffectList(fields.triggeredEffects, triggered);
  renderEffectList(fields.activeEffects, active, { counts: true });
}

function setViewMode(mode) {
  activeView = mode;
  const is3d = mode === "3d";
  canvas.hidden = is3d;
  canvas3d.hidden = !is3d;
  fields.view2d.classList.toggle("active", !is3d);
  fields.view3d.classList.toggle("active", is3d);
  lightPreview3d.setActive(is3d);
  if (is3d) {
    syncLightPreview3d();
    lightPreview3d.render();
  } else {
    requestDraw();
  }
}

function bindForm() {
  fields.roomWidth.value = spatial.room.width;
  fields.roomDepth.value = spatial.room.depth;
  fields.roomHeight.value = spatial.room.height;
  fields.tvWall.value = spatial.tv.wall;
  fields.tvU.value = spatial.tv.center_u;
  fields.tvV.value = spatial.tv.center_v;
  fields.tvWidth.value = spatial.tv.width;
  fields.tvHeight.value = spatial.tv.height;
  bindRenderPerformance();
  refreshEditorState();
}

function readRoomTvForm() {
  spatial.enabled = true;
  spatial.room.width = number(fields.roomWidth.value, 4);
  spatial.room.depth = number(fields.roomDepth.value, 3);
  spatial.room.height = number(fields.roomHeight.value, 2.4);
  spatial.tv.wall = fields.tvWall.value;
  spatial.tv.center_u = number(fields.tvU.value, 2);
  spatial.tv.center_v = number(fields.tvV.value, 1.25);
  spatial.tv.width = number(fields.tvWidth.value, 1.4);
  spatial.tv.height = number(fields.tvHeight.value, 0.8);
}

function bindRenderPerformance() {
  if (!config) return;
  fields.targetFps.value = config.target_fps ?? 90;
  fields.wledFps.value = config.wled_fps ?? 90;
  fields.renderMode.value = config.render_mode || "full_frame";
  fields.edgeBandWidth.value = config.edge_band_width ?? 192;
  fields.edgeBandHeight.value = config.edge_band_height ?? 108;
  fields.edgeBandFraction.value = config.edge_band_fraction ?? 0.12;
  fields.hybridFullWidth.value = config.hybrid_full_width ?? 64;
  fields.hybridFullHeight.value = config.hybrid_full_height ?? 36;
}

function readRenderPerformanceForm() {
  config.render_mode = fields.renderMode.value || "full_frame";
  config.target_fps = Math.max(1, Math.min(90, Math.round(number(fields.targetFps.value, config.target_fps ?? 90))));
  config.wled_fps = Math.max(1, Math.min(90, Math.round(number(fields.wledFps.value, config.wled_fps ?? 90))));
  config.edge_band_width = Math.max(8, Math.round(number(fields.edgeBandWidth.value, config.edge_band_width ?? 192)));
  config.edge_band_height = Math.max(8, Math.round(number(fields.edgeBandHeight.value, config.edge_band_height ?? 108)));
  config.edge_band_fraction = Math.max(0.01, Math.min(0.5, number(fields.edgeBandFraction.value, config.edge_band_fraction ?? 0.12)));
  config.hybrid_full_width = Math.max(8, Math.round(number(fields.hybridFullWidth.value, config.hybrid_full_width ?? 64)));
  config.hybrid_full_height = Math.max(8, Math.round(number(fields.hybridFullHeight.value, config.hybrid_full_height ?? 36)));
}

function refreshEditorState({ syncDevicesJson = true, syncStripsJson = true, redraw = true } = {}) {
  if (selectedStripId && !selectedStrip()) {
    selectedStripId = null;
  }
  if (syncDevicesJson) syncDevicesJsonText();
  if (syncStripsJson) syncStripsJsonText();
  bindStripEditor();
  syncLightPreview3d();
  if (redraw) requestDraw();
}

function syncDevicesJsonText() {
  fields.devicesJson.value = JSON.stringify(spatial.devices, null, 2);
}

function syncStripsJsonText() {
  fields.stripsJson.value = JSON.stringify(spatial.strips, null, 2);
}

function parseDevicesJson() {
  const devices = JSON.parse(fields.devicesJson.value || "[]");
  if (!Array.isArray(devices)) {
    throw new Error("Devices JSON must be an array");
  }
  spatial.devices = devices;
  const validDeviceIds = new Set(spatial.devices.map((device) => device.id));
  for (const strip of spatial.strips) {
    if (!validDeviceIds.has(strip.device_id) && spatial.devices[0]) {
      strip.device_id = spatial.devices[0].id;
    }
  }
  packDeviceRanges();
}

function parseStripsJson() {
  const strips = JSON.parse(fields.stripsJson.value || "[]");
  if (!Array.isArray(strips)) {
    throw new Error("Strips JSON must be an array");
  }
  spatial.strips = strips;
  if (!selectedStrip()) {
    selectedStripId = null;
  }
  packDeviceRanges();
}

function number(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function selectedStrip() {
  if (!spatial) return null;
  return spatial.strips.find((strip) => strip.id === selectedStripId) || null;
}

function bindStripEditor() {
  bindStripSelect();
  bindDeviceSelect();
  bindExtensionSelect();
  const strip = selectedStrip();
  const hasStrip = Boolean(strip);
  for (const input of stripEditorInputs()) {
    input.disabled = !hasStrip;
  }
  fields.stripSelect.disabled = spatial.strips.length === 0;
  fields.deselectStrip.disabled = !hasStrip;
  fields.duplicateStrip.disabled = !hasStrip;
  fields.deleteStrip.disabled = !hasStrip;
  fields.stripNudgeLeft.disabled = !hasStrip;
  fields.stripNudgeRight.disabled = !hasStrip;
  fields.stripNudgeUp.disabled = !hasStrip;
  fields.stripNudgeDown.disabled = !hasStrip;
  if (!strip) {
    clearStripEditor();
    return;
  }
  fields.stripSelect.value = strip.id;
  fields.stripName.value = strip.name || strip.id;
  fields.stripWall.value = strip.wall;
  fields.stripStartU.value = strip.start_u;
  fields.stripStartV.value = strip.start_v;
  fields.stripEndU.value = strip.end_u;
  fields.stripEndV.value = strip.end_v;
  fields.stripLedCount.value = strip.led_count;
  fields.stripDirection.value = strip.direction || "forward";
  fields.stripDevice.value = strip.device_id;
  fields.stripDeviceStart.value = strip.device_start;
  fields.stripSyncMode.value = strip.sync_mode || "spatial";
  fields.stripTvRole.value = strip.tv_role || "none";
  fields.stripTvFill.value = strip.tv_fill ?? 1.0;
  fields.stripTvFillSpatial.checked = Boolean(strip.tv_fill_spatial);
  fields.stripExtends.value = strip.extends_strip_id || "";
  fields.stripExtensionMode.value = strip.extension_mode || "soft_spill";
  fields.stripExtensionStrength.value = strip.extension_strength ?? 0.45;
  fields.stripExtensionSoftness.value = strip.extension_softness ?? 0.65;
  fields.stripBlend.value = strip.blend ?? 0.5;
  bindExtensionControls(strip);
}

function stripEditorInputs() {
  return [
    fields.stripName,
    fields.stripWall,
    fields.stripStartU,
    fields.stripStartV,
    fields.stripEndU,
    fields.stripEndV,
    fields.stripLedCount,
    fields.stripDirection,
    fields.stripDevice,
    fields.stripDeviceStart,
    fields.stripSyncMode,
    fields.stripTvRole,
    fields.stripTvFill,
    fields.stripTvFillSpatial,
    fields.stripExtends,
    fields.stripExtensionMode,
    fields.stripExtensionStrength,
    fields.stripExtensionSoftness,
    fields.stripBlend,
  ];
}

function clearStripEditor() {
  fields.stripSelect.value = "";
  fields.stripName.value = "";
  fields.stripStartU.value = "";
  fields.stripStartV.value = "";
  fields.stripEndU.value = "";
  fields.stripEndV.value = "";
  fields.stripLedCount.value = "";
  fields.stripDeviceStart.value = "";
  fields.stripBlend.value = 0.5;
  fields.stripTvRole.value = "none";
  fields.stripTvFill.value = 1.0;
  fields.stripTvFillRow.hidden = true;
  fields.stripTvFillSpatial.checked = false;
  fields.stripTvFillSpatialRow.hidden = true;
  fields.stripExtends.value = "";
  fields.stripExtensionMode.value = "soft_spill";
  fields.stripExtensionStrength.value = 0.45;
  fields.stripExtensionSoftness.value = 0.65;
}

function bindExtensionControls(strip) {
  const isExtension = Boolean(strip?.extends_strip_id);
  const tvFillCapable = strip?.sync_mode === "tv_image" || strip?.sync_mode === "blend";
  fields.stripTvFillRow.hidden = !tvFillCapable;
  fields.stripTvFill.disabled = !strip || !tvFillCapable;
  fields.stripTvFillSpatialRow.hidden = !tvFillCapable;
  fields.stripTvFillSpatial.disabled = !strip || !tvFillCapable;
  fields.stripExtensionMode.disabled = !strip || !isExtension;
  fields.stripExtensionStrength.disabled = !strip || !isExtension || fields.stripExtensionMode.value === "effects_only";
  fields.stripExtensionSoftness.disabled = !strip || !isExtension || fields.stripExtensionMode.value === "effects_only";
}

function bindStripSelect() {
  const current = selectedStripId;
  fields.stripSelect.replaceChildren();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "none";
  fields.stripSelect.append(none);
  for (const strip of spatial.strips) {
    const option = document.createElement("option");
    option.value = strip.id;
    option.textContent = `${strip.name || strip.id} (${strip.wall})`;
    fields.stripSelect.append(option);
  }
  if (spatial.strips.some((strip) => strip.id === current)) {
    fields.stripSelect.value = current;
  } else {
    fields.stripSelect.value = "";
  }
}

function bindDeviceSelect() {
  const current = selectedStrip()?.device_id;
  fields.stripDevice.replaceChildren();
  for (const device of spatial.devices) {
    const option = document.createElement("option");
    option.value = device.id;
    option.textContent = device.name ? `${device.name} (${device.id})` : device.id;
    fields.stripDevice.append(option);
  }
  if (current) fields.stripDevice.value = current;
}

function bindExtensionSelect() {
  const currentStrip = selectedStrip();
  const current = currentStrip?.extends_strip_id || "";
  fields.stripExtends.replaceChildren();
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "none";
  fields.stripExtends.append(none);
  for (const strip of spatial.strips) {
    if (strip.id === currentStrip?.id) continue;
    const option = document.createElement("option");
    option.value = strip.id;
    option.textContent = `${strip.name || strip.id} (${effectiveTvRole(strip.id)})`;
    fields.stripExtends.append(option);
  }
  fields.stripExtends.value = current;
}

function updateSelectedStripFromEditor() {
  const strip = selectedStrip();
  if (!strip) return;
  strip.name = fields.stripName.value || strip.id;
  strip.wall = fields.stripWall.value;
  strip.start_u = number(fields.stripStartU.value, strip.start_u);
  strip.start_v = number(fields.stripStartV.value, strip.start_v);
  strip.end_u = number(fields.stripEndU.value, strip.end_u);
  strip.end_v = number(fields.stripEndV.value, strip.end_v);
  strip.led_count = Math.max(1, Math.round(number(fields.stripLedCount.value, strip.led_count)));
  strip.direction = fields.stripDirection.value;
  strip.device_id = fields.stripDevice.value || strip.device_id;
  strip.device_start = Math.max(0, Math.round(number(fields.stripDeviceStart.value, strip.device_start)));
  strip.sync_mode = fields.stripSyncMode.value;
  strip.tv_role = fields.stripTvRole.value || "none";
  strip.tv_fill = Math.max(0, Math.min(1, number(fields.stripTvFill.value, strip.tv_fill ?? 1.0)));
  strip.tv_fill_spatial = Boolean(fields.stripTvFillSpatial.checked);
  strip.extends_strip_id = fields.stripExtends.value || "";
  strip.extension_mode = fields.stripExtensionMode.value || "soft_spill";
  strip.extension_strength = Math.max(0, Math.min(1, number(fields.stripExtensionStrength.value, strip.extension_strength ?? 0.45)));
  strip.extension_softness = Math.max(0, Math.min(1, number(fields.stripExtensionSoftness.value, strip.extension_softness ?? 0.65)));
  if (strip.tv_role !== "none") {
    strip.extends_strip_id = "";
  } else if (strip.extends_strip_id) {
    strip.tv_role = "none";
  }
  if (strip.sync_mode === "spatial" && strip.extends_strip_id) {
    strip.extension_mode = "effects_only";
  }
  strip.blend = Math.max(0, Math.min(1, number(fields.stripBlend.value, strip.blend ?? 0.5)));
  clampStripToRoom(strip);
  packDeviceRanges();
  refreshEditorState();
  markDirty();
}

function selectStrip(id) {
  selectedStripId = id || null;
  bindStripEditor();
  requestDraw();
}

function deselectStrip() {
  selectedStripId = null;
  bindStripEditor();
  requestDraw();
}

function syncStripsJson() {
  syncStripsJsonText();
}

function markDirty() {
  statusEl.textContent = "Unsaved changes";
  scheduleAutoSave();
}

function scheduleAutoSave(delay = 650) {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    saveTimer = null;
    saveSpatialConfig({ silent: true });
  }, delay);
}

function wallLength(wall) {
  return wall === "front" || wall === "rear" ? spatial.room.width : spatial.room.depth;
}

function packDeviceRanges() {
  if (!spatial || !Array.isArray(spatial.devices) || !Array.isArray(spatial.strips)) return;
  const devicesById = new Map(spatial.devices.map((device) => [device.id, device]));
  if (!devicesById.size) return;
  const cursors = new Map();
  for (const strip of spatial.strips) {
    if (!devicesById.has(strip.device_id)) {
      strip.device_id = spatial.devices[0].id;
    }
    strip.led_count = Math.max(1, Math.round(number(strip.led_count, 1)));
    const start = cursors.get(strip.device_id) || 0;
    strip.device_start = start;
    cursors.set(strip.device_id, start + strip.led_count);
  }
  for (const [deviceId, used] of cursors.entries()) {
    const device = devicesById.get(deviceId);
    if (device) {
      device.led_count = Math.max(Math.round(number(device.led_count, 0)), used);
    }
  }
}

function clampStripToRoom(strip) {
  const length = wallLength(strip.wall);
  strip.start_u = Math.max(0, Math.min(length, strip.start_u));
  strip.end_u = Math.max(0, Math.min(length, strip.end_u));
  strip.start_v = Math.max(0, Math.min(spatial.room.height, strip.start_v));
  strip.end_v = Math.max(0, Math.min(spatial.room.height, strip.end_v));
}

function nudgeSelected(du, dv) {
  const strip = selectedStrip();
  if (!strip) return;
  strip.start_u += du;
  strip.end_u += du;
  strip.start_v += dv;
  strip.end_v += dv;
  clampStripToRoom(strip);
  refreshEditorState({ syncDevicesJson: false });
  markDirty();
}

function duplicateSelectedStrip() {
  const strip = selectedStrip();
  if (!strip) return;
  const copy = JSON.parse(JSON.stringify(strip));
  copy.id = uniqueStripId(`${strip.id}-copy`);
  copy.name = `${strip.name || strip.id} Copy`;
  copy.start_v = Math.min(spatial.room.height, copy.start_v + 0.1);
  copy.end_v = Math.min(spatial.room.height, copy.end_v + 0.1);
  spatial.strips.push(copy);
  selectedStripId = copy.id;
  packDeviceRanges();
  refreshEditorState();
  markDirty();
}

function deleteSelectedStrip() {
  const strip = selectedStrip();
  if (!strip) return;
  spatial.strips = spatial.strips.filter((item) => item.id !== strip.id);
  selectedStripId = spatial.strips[0]?.id || null;
  packDeviceRanges();
  refreshEditorState();
  markDirty();
}

function uniqueStripId(base) {
  const ids = new Set(spatial.strips.map((strip) => strip.id));
  let id = base;
  let index = 2;
  while (ids.has(id)) {
    id = `${base}-${index}`;
    index += 1;
  }
  return id;
}

function effectiveTvRole(stripId, seen = new Set()) {
  const strip = spatial?.strips.find((item) => item.id === stripId);
  if (!strip || seen.has(stripId)) return "none";
  if (strip.tv_role && strip.tv_role !== "none") return strip.tv_role;
  if (!strip.extends_strip_id) return "none";
  return effectiveTvRole(strip.extends_strip_id, new Set([...seen, stripId]));
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function requestDraw() {
  if (drawScheduled) return;
  drawScheduled = true;
  requestAnimationFrame(() => {
    drawScheduled = false;
    drawScene();
  });
}

function renderSpatial(dirty = false) {
  try {
    readRoomTvForm();
    refreshEditorState({ syncDevicesJson: false, syncStripsJson: false });
    statusEl.textContent = dirty ? "Unsaved changes" : "Loaded";
  } catch (error) {
    statusEl.textContent = "JSON error";
  }
  requestDraw();
  if (dirty) scheduleAutoSave();
}

function drawScene() {
  if (!spatial) return;
  resizeCanvas();
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#161a20";
  ctx.fillRect(0, 0, rect.width, rect.height);
  stripHitTargets = [];

  const room = spatial.room;
  const center = { x: room.width / 2, y: room.height / 2, z: room.depth / 2 };
  const projected = makeProjector(rect.width, rect.height, center);

  drawRoom(room, projected);
  drawTV(spatial.tv, projected);
  drawStrips(spatial.strips, projected);
  drawLegend(rect.width, rect.height);
  drawViewHud(rect.width, rect.height);
}

function makeProjector(width, height, center) {
  return (point) => {
    let x = point.x - center.x;
    let y = point.y - center.y;
    let z = point.z - center.z;
    const cy = Math.cos(yaw);
    const sy = Math.sin(yaw);
    const cp = Math.cos(pitch);
    const sp = Math.sin(pitch);
    const x1 = x * cy - z * sy;
    const z1 = x * sy + z * cy;
    const y1 = y * cp - z1 * sp;
    return {
      x: width / 2 + x1 * zoom,
      y: height / 2 - y1 * zoom,
      depth: z1,
    };
  };
}

function currentProjector() {
  const rect = canvas.getBoundingClientRect();
  const room = spatial.room;
  const center = { x: room.width / 2, y: room.height / 2, z: room.depth / 2 };
  return makeProjector(rect.width, rect.height, center);
}

function drawRoom(room, project) {
  const corners = [
    p(0, 0, 0),
    p(room.width, 0, 0),
    p(room.width, room.height, 0),
    p(0, room.height, 0),
    p(0, 0, room.depth),
    p(room.width, 0, room.depth),
    p(room.width, room.height, room.depth),
    p(0, room.height, room.depth),
  ];
  const floor = [corners[0], corners[1], corners[5], corners[4]].map(project);
  fillPolygon(floor, "rgba(32, 38, 46, 0.92)", "rgba(128, 144, 164, 0.75)");
  const back = [corners[4], corners[5], corners[6], corners[7]].map(project);
  fillPolygon(back, "rgba(22, 28, 36, 0.42)", "rgba(82, 96, 112, 0.65)");
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  for (const [a, b] of edges) {
    line(project(corners[a]), project(corners[b]), "#8090a4", 2);
  }
  label(project(p(room.width / 2, 0, 0)), "front", "#7ab8ff", 0, 18);
  label(project(p(room.width, 0, room.depth / 2)), "left", "#91f0aa", 30, 0);
  label(project(p(room.width / 2, 0, room.depth)), "rear", "#ffd568", 0, 18);
  label(project(p(0, 0, room.depth / 2)), "right", "#ff7e92", -30, 0);
}

function drawTV(tv, project) {
  const left = tv.center_u - tv.width / 2;
  const right = tv.center_u + tv.width / 2;
  const bottom = tv.center_v - tv.height / 2;
  const top = tv.center_v + tv.height / 2;
  const points = [
    wallPoint(tv.wall, left, bottom),
    wallPoint(tv.wall, right, bottom),
    wallPoint(tv.wall, right, top),
    wallPoint(tv.wall, left, top),
  ].map(project);
  fillPolygon(points, "#050607", "#e8eef6");
  if (previewRunning && previewImage?.complete) {
    drawImageOnQuad(previewImage, points, tv.mirror_horizontal !== false);
    strokePolygon(points, "#e8eef6", 2);
  } else {
    const center = project(wallPoint(tv.wall, tv.center_u, tv.center_v));
    label(center, "TV", "#e8eef6", -9, 4);
  }
}

function drawStrips(strips, project) {
  let cursor = 0;
  for (const strip of strips) {
    const count = Math.max(1, Number(strip.led_count) || 1);
    const selected = strip.id === selectedStripId;
    let previous = null;
    let previousColor = null;
    const stripPoints = [];
    for (let i = 0; i < count; i += 1) {
      const rawT = count === 1 ? 0 : i / (count - 1);
      const t = strip.direction === "reverse" ? 1 - rawT : rawT;
      const u = strip.start_u + (strip.end_u - strip.start_u) * t;
      const v = strip.start_v + (strip.end_v - strip.start_v) * t;
      const projected = project(wallPoint(strip.wall, u, v));
      const color = rgb(previewColors[cursor + i] || modeColor(strip.sync_mode));
      if (previous) {
        line(previous, projected, selected ? "rgba(255,255,255,0.72)" : "rgba(255,255,255,0.20)", selected ? 5 : 3);
        line(previous, projected, previousColor || color, selected ? 3 : 2);
        line(previous, projected, color, selected ? 2 : 1);
      }
      dot(projected, selected ? 5.2 : 3.5, selected ? "#ffffff" : color);
      dot(projected, selected ? 3.4 : 2.8, color);
      previous = projected;
      previousColor = color;
      stripPoints.push(projected);
    }
    const start = project(wallPoint(strip.wall, strip.start_u, strip.start_v));
    const end = project(wallPoint(strip.wall, strip.end_u, strip.end_v));
    if (selected) {
      endpoint(start, "#79d8ff");
      endpoint(end, "#ffcc66");
    }
    stripHitTargets.push({ id: strip.id, points: stripPoints, start, end });
    const role = effectiveTvRole(strip.id);
    const extension = strip.extends_strip_id ? ` ${strip.extension_mode || "soft_spill"}` : "";
    const suffix = role !== "none" ? ` [${role}${extension}]` : "";
    label(start, `${strip.name || strip.id}${suffix}`, selected ? "#ffffff" : "#d7e4ef", 8, -8);
    cursor += count;
  }
}

function drawLegend(width, height) {
  ctx.fillStyle = "rgba(12, 16, 22, 0.72)";
  ctx.fillRect(12, height - 86, 360, 64);
  ctx.fillStyle = "#b9c7d4";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("Drag strip body to move. Drag endpoint handles to resize.", 24, height - 61);
  ctx.fillText("Drag empty room to rotate. Mouse wheel to zoom.", 24, height - 43);
  ctx.fillText("Strip colors show the current preview LED output.", 24, height - 25);
}

function drawViewHud(width, height) {
  const x = width - 152;
  const y = 18;
  ctx.fillStyle = "rgba(12, 16, 22, 0.78)";
  ctx.fillRect(x, y, 134, 92);
  ctx.strokeStyle = "rgba(185, 199, 212, 0.45)";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, 134, 92);
  ctx.fillStyle = "#d7e4ef";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(projectionMode, x + 12, y + 20);
  ctx.fillStyle = "#8da3b8";
  ctx.fillText(`yaw ${degrees(yaw)}`, x + 12, y + 38);
  ctx.fillText(`pitch ${degrees(pitch)}`, x + 12, y + 54);

  const origin = { x: x + 70, y: y + 72 };
  const xAxis = viewAxisVector(1, 0);
  const zAxis = viewAxisVector(0, 1);
  hudArrow(origin, xAxis, "#7ab8ff", "X");
  hudArrow(origin, zAxis, "#ffd568", "Z");
}

function viewAxisVector(dx, dz) {
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const sp = Math.sin(pitch);
  const x1 = dx * cy - dz * sy;
  const z1 = dx * sy + dz * cy;
  const y1 = -z1 * sp;
  const length = Math.max(0.001, Math.hypot(x1, y1));
  return { x: x1 / length, y: -y1 / length };
}

function hudArrow(origin, vector, color, text) {
  const end = { x: origin.x + vector.x * 30, y: origin.y + vector.y * 30 };
  line(origin, end, color, 2);
  dot(end, 3.5, color);
  label(end, text, color, 5, 4);
}

function degrees(radians) {
  return `${Math.round((radians * 180) / Math.PI)}deg`;
}

function wallPoint(wall, u, v) {
  const room = spatial.room;
  if (wall === "front") return p(u, v, 0.015);
  if (wall === "rear") return p(room.width - u, v, room.depth - 0.015);
  if (wall === "left") return p(room.width - 0.015, v, u);
  return p(0.015, v, room.depth - u);
}

function p(x, y, z) {
  return { x, y, z };
}

function fillPolygon(points, fill, stroke) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function strokePolygon(points, stroke, width) {
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.closePath();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.stroke();
}

function drawImageOnQuad(image, points, mirrorHorizontal = false) {
  const bottomLeft = points[0];
  const bottomRight = points[1];
  const topRight = points[2];
  const topLeft = points[3];
  const width = image.naturalWidth || image.width;
  const height = image.naturalHeight || image.height;
  if (!width || !height) return;
  const origin = mirrorHorizontal ? topRight : topLeft;
  const xTarget = mirrorHorizontal ? topLeft : topRight;
  const yTarget = mirrorHorizontal ? bottomRight : bottomLeft;
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(points[0].x, points[0].y);
  for (const point of points.slice(1)) ctx.lineTo(point.x, point.y);
  ctx.closePath();
  ctx.clip();
  ctx.transform(
    (xTarget.x - origin.x) / width,
    (xTarget.y - origin.y) / width,
    (yTarget.x - origin.x) / height,
    (yTarget.y - origin.y) / height,
    origin.x,
    origin.y,
  );
  ctx.drawImage(image, 0, 0, width, height);
  ctx.restore();
}

function line(a, b, color, width) {
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function dot(point, radius, color) {
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
}

function endpoint(point, color) {
  ctx.fillStyle = color;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.rect(point.x - 5, point.y - 5, 10, 10);
  ctx.fill();
  ctx.stroke();
}

function label(point, text, color, dx, dy) {
  ctx.fillStyle = color;
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(text, point.x + dx, point.y + dy);
}

function stripHitAtCanvasPoint(x, y) {
  let best = null;
  for (const target of stripHitTargets) {
    const startDistance = Math.hypot(x - target.start.x, y - target.start.y);
    if (startDistance <= 14 && (!best || startDistance < best.distance)) {
      best = { id: target.id, mode: "start", distance: startDistance };
    }
    const endDistance = Math.hypot(x - target.end.x, y - target.end.y);
    if (endDistance <= 14 && (!best || endDistance < best.distance)) {
      best = { id: target.id, mode: "end", distance: endDistance };
    }
    for (let i = 1; i < target.points.length; i += 1) {
      const distance = distanceToSegment({ x, y }, target.points[i - 1], target.points[i]);
      if (distance <= 12 && (!best || distance < best.distance)) {
        best = { id: target.id, mode: "move", distance };
      }
    }
    if (target.points.length === 1) {
      const distance = Math.hypot(x - target.points[0].x, y - target.points[0].y);
      if (distance <= 12 && (!best || distance < best.distance)) {
        best = { id: target.id, mode: "move", distance };
      }
    }
  }
  return best;
}

function distanceToSegment(point, a, b) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return Math.hypot(point.x - a.x, point.y - a.y);
  const t = Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSquared));
  const x = a.x + t * dx;
  const y = a.y + t * dy;
  return Math.hypot(point.x - x, point.y - y);
}

function beginStripDrag(hit, clientX, clientY) {
  selectStrip(hit.id);
  const strip = selectedStrip();
  if (!strip) return;
  editDrag = {
    mode: hit.mode,
    startClientX: clientX,
    startClientY: clientY,
    original: JSON.parse(JSON.stringify(strip)),
  };
  canvas.classList.add("editing-strip");
  statusEl.textContent = hit.mode === "move" ? "Moving strip" : "Resizing strip";
}

function updateStripDrag(clientX, clientY) {
  if (!editDrag) return;
  const strip = selectedStrip();
  if (!strip) return;
  const dx = clientX - editDrag.startClientX;
  const dy = clientY - editDrag.startClientY;
  const anchorU = (editDrag.original.start_u + editDrag.original.end_u) / 2;
  const anchorV = (editDrag.original.start_v + editDrag.original.end_v) / 2;
  const delta = screenDeltaToWallDelta(editDrag.original.wall, anchorU, anchorV, dx, dy);

  strip.wall = editDrag.original.wall;
  if (editDrag.mode === "start") {
    strip.start_u = editDrag.original.start_u + delta.du;
    strip.start_v = editDrag.original.start_v + delta.dv;
    strip.end_u = editDrag.original.end_u;
    strip.end_v = editDrag.original.end_v;
  } else if (editDrag.mode === "end") {
    strip.start_u = editDrag.original.start_u;
    strip.start_v = editDrag.original.start_v;
    strip.end_u = editDrag.original.end_u + delta.du;
    strip.end_v = editDrag.original.end_v + delta.dv;
  } else {
    strip.start_u = editDrag.original.start_u + delta.du;
    strip.start_v = editDrag.original.start_v + delta.dv;
    strip.end_u = editDrag.original.end_u + delta.du;
    strip.end_v = editDrag.original.end_v + delta.dv;
  }
  clampStripToRoom(strip);
  refreshEditorState({ syncDevicesJson: false });
}

function finishStripDrag() {
  if (!editDrag) return;
  editDrag = null;
  canvas.classList.remove("editing-strip");
  markDirty();
}

function screenDeltaToWallDelta(wall, anchorU, anchorV, dx, dy) {
  const project = currentProjector();
  const length = wallLength(wall);
  const epsU = Math.max(0.01, length * 0.01);
  const epsV = Math.max(0.01, spatial.room.height * 0.01);
  const origin = project(wallPoint(wall, anchorU, anchorV));
  const uPoint = project(wallPoint(wall, anchorU + epsU, anchorV));
  const vPoint = project(wallPoint(wall, anchorU, anchorV + epsV));
  const ux = (uPoint.x - origin.x) / epsU;
  const uy = (uPoint.y - origin.y) / epsU;
  const vx = (vPoint.x - origin.x) / epsV;
  const vy = (vPoint.y - origin.y) / epsV;
  const det = ux * vy - uy * vx;
  if (Math.abs(det) < 0.0001) {
    return { du: 0, dv: 0 };
  }
  return {
    du: (dx * vy - dy * vx) / det,
    dv: (ux * dy - uy * dx) / det,
  };
}

function rgb(color) {
  return `rgb(${Math.round(color[0])}, ${Math.round(color[1])}, ${Math.round(color[2])})`;
}

function modeColor(_mode) {
  return [210, 218, 228];
}

async function saveSpatialConfig({ silent = false } = {}) {
  if (saveInFlight) {
    saveAgain = true;
    return;
  }
  packDeviceRanges();
  refreshEditorState({ redraw: false });
  readRenderPerformanceForm();
  saveInFlight = true;
  if (!silent) statusEl.textContent = "Saving";
  const response = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      spatial,
      enabled_effects: { ...(config.enabled_effects || {}), ...pendingEffectOverrides },
      effect_sensitivity: { ...(config.effect_sensitivity || {}), ...pendingSensitivityOverrides },
      ambient_side_spill_base_intensity: pendingAmbientSpillSettings.ambient_side_spill_base_intensity ?? config.ambient_side_spill_base_intensity,
      ambient_side_spill_boost_intensity: pendingAmbientSpillSettings.ambient_side_spill_boost_intensity ?? config.ambient_side_spill_boost_intensity,
      tv_image_blur: pendingAmbientSpillSettings.tv_image_blur ?? config.tv_image_blur,
      target_fps: config.target_fps,
      wled_fps: config.wled_fps,
      render_mode: config.render_mode,
      edge_band_width: config.edge_band_width,
      edge_band_height: config.edge_band_height,
      edge_band_fraction: config.edge_band_fraction,
      hybrid_full_width: config.hybrid_full_width,
      hybrid_full_height: config.hybrid_full_height,
    }),
  });
  const payload = await response.json();
  saveInFlight = false;
  if (!response.ok || !payload.ok) {
    statusEl.textContent = (payload.errors || ["Save failed"]).join("; ");
    return;
  }
  if (!effectSaveInFlight) {
    if (effectSaveTimer) {
      clearTimeout(effectSaveTimer);
      effectSaveTimer = null;
    }
    pendingEffectOverrides = {};
    pendingSensitivityOverrides = {};
    pendingAmbientSpillSettings = {};
  }
  config = payload.config;
  spatial = config.spatial;
  bindForm();
  renderEffectToggles();
  renderEffectSensitivity();
  syncAmbientSpillControls();
  await loadPreviewColors();
  syncLightPreview3d();
  requestDraw();
  statusEl.textContent = silent ? "Auto-saved" : "Saved";
  if (saveAgain) {
    saveAgain = false;
    scheduleAutoSave(100);
  }
}

async function saveConfig() {
  if (saveTimer) {
    clearTimeout(saveTimer);
    saveTimer = null;
  }
  try {
    readRoomTvForm();
    refreshEditorState({ redraw: false });
  } catch (error) {
    statusEl.textContent = "JSON error";
    return;
  }
  await saveSpatialConfig({ silent: false });
}

async function startPreview() {
  if (saveTimer) {
    clearTimeout(saveTimer);
    saveTimer = null;
  }
  await saveSpatialConfig({ silent: true });
  fields.previewStatus.textContent = "Starting preview";
  const response = await fetch("/api/preview/start", { method: "POST" });
  const payload = await response.json();
  previewRunning = Boolean(payload.running);
  fields.previewStatus.textContent = previewRunning ? `Preview running (${payload.mode || "config"})` : `Preview failed: ${payload.error || "unknown error"}`;
  if (previewRunning) schedulePreviewPoll(PREVIEW_POLL_INTERVAL_MS);
}

async function startSimulation() {
  if (saveTimer) {
    clearTimeout(saveTimer);
    saveTimer = null;
  }
  await saveSpatialConfig({ silent: true });
  fields.previewStatus.textContent = "Starting simulation preview";
  const response = await fetch("/api/preview/start-simulation", { method: "POST" });
  const payload = await response.json();
  previewRunning = Boolean(payload.running);
  fields.previewStatus.textContent = previewRunning ? "Simulation preview running" : `Simulation failed: ${payload.error || "unknown error"}`;
  if (previewRunning) schedulePreviewPoll(PREVIEW_POLL_INTERVAL_MS);
}

async function stopPreview(path = "/api/preview/stop") {
  await fetch(path, { method: "POST" });
  previewRunning = false;
  if (previewTimer) {
    clearTimeout(previewTimer);
    previewTimer = null;
  }
  fields.previewStatus.textContent = "Preview stopped";
  loopingEffect = "";
  renderEffectPatterns();
  renderEffectList(fields.triggeredEffects, []);
  renderEffectList(fields.activeEffects, []);
  previewImage = null;
  previewFrameRevision = 0;
  previewFrameRequestRevision = 0;
  lastLedRevision = 0;
  await loadPreviewColors();
  syncLightPreview3d();
  requestDraw();
}

async function stopSimulation() {
  await stopPreview("/api/preview/stop-simulation");
}

function schedulePreviewPoll(delay = 160) {
  if (previewTimer) clearTimeout(previewTimer);
  previewTimer = setTimeout(pollPreviewStatus, Math.max(PREVIEW_POLL_INTERVAL_MS, delay));
}

function unpackLedColors(flat, count) {
  const colors = [];
  const expected = Math.min(flat.length, Math.max(0, count) * 3);
  for (let i = 0; i + 2 < expected; i += 3) {
    colors.push([flat[i], flat[i + 1], flat[i + 2]]);
  }
  return colors;
}

function fetchPreviewFrame(revision) {
  if (!revision || revision <= previewFrameRevision || revision <= previewFrameRequestRevision || previewFrameLoading) return;
  previewFrameLoading = true;
  previewFrameRequestRevision = revision;
  const image = new Image();
  image.onload = () => {
    if (!previewRunning) {
      previewFrameLoading = false;
      return;
    }
    previewImage = image;
    previewFrameRevision = revision;
    previewFrameLoading = false;
    syncLightPreview3d();
    requestDraw();
  };
  image.onerror = () => {
    previewFrameLoading = false;
  };
  image.src = `/api/preview/frame.jpg?rev=${revision}`;
}

async function pollPreviewStatus() {
  if (!previewRunning) return;
  try {
    const response = await fetch("/api/preview/status");
    const payload = await response.json();
    previewRunning = Boolean(payload.running);
    loopingEffect = payload.looping_effect || "";
    renderEffectPatterns();
    updateLiveEffects(payload);
    if (
      Array.isArray(payload.leds_flat)
      && payload.leds_flat.length
      && Number(payload.led_revision || 0) !== lastLedRevision
    ) {
      previewColors = unpackLedColors(payload.leds_flat, Number(payload.led_count || 0));
      lastLedRevision = Number(payload.led_revision || 0);
      syncLightPreview3d();
      requestDraw();
    } else if (Array.isArray(payload.leds) && payload.leds.length) {
      previewColors = payload.leds;
      syncLightPreview3d();
      requestDraw();
    }
    fetchPreviewFrame(Number(payload.frame_revision || 0));
    const waitingForFrames = Number(payload.frame_revision || 0) === 0;
    const mode = payload.preview_mode === "simulation" ? "Simulation" : "Live";
    const source = waitingForFrames
      ? "Waiting for frames"
      : (payload.hyperhdr_connected ? "HyperHDR connected" : "HyperHDR reconnecting");
    fields.previewStatus.textContent = previewRunning
      ? `${mode} preview running | ${source} | FPS ${Number(payload.fps || 0).toFixed(1)} | View render ${Number(previewRenderStats.avgMs || 0).toFixed(2)}ms avg/${Number(previewRenderStats.samples || 0)} | Lit ${Number(payload.lit_led_count || 0)}/${Number(payload.led_count || 0)} | Max LED ${Number(payload.led_max || 0)} | Waves ${payload.active_waves || 0}`
      : "Preview stopped";
  } catch (error) {
    fields.previewStatus.textContent = `Preview error: ${error}`;
  }
  if (previewRunning) schedulePreviewPoll(PREVIEW_POLL_INTERVAL_MS);
}

function addStrip() {
  readRoomTvForm();
  const room = spatial.room;
  const wall = spatial.tv.wall;
  const length = wall === "front" || wall === "rear" ? room.width : room.depth;
  const id = uniqueStripId(`strip-${spatial.strips.length + 1}`);
  spatial.strips.push({
    id,
    name: `Strip ${spatial.strips.length + 1}`,
    wall,
    start_u: 0,
    start_v: Math.min(room.height, spatial.tv.center_v + spatial.tv.height / 2 + 0.2),
    end_u: length,
    end_v: Math.min(room.height, spatial.tv.center_v + spatial.tv.height / 2 + 0.2),
    led_count: 60,
    direction: "forward",
    device_id: spatial.devices[0]?.id || "main",
    device_start: 0,
    sync_mode: "spatial",
    blend: 0.5,
    tv_fill: 1.0,
    tv_fill_spatial: false,
    tv_role: "none",
    extends_strip_id: "",
    extension_mode: "soft_spill",
    extension_strength: 0.45,
    extension_softness: 0.65,
  });
  selectedStripId = id;
  packDeviceRanges();
  refreshEditorState();
  markDirty();
}

for (const element of [fields.roomWidth, fields.roomDepth, fields.roomHeight, fields.tvWall, fields.tvU, fields.tvV, fields.tvWidth, fields.tvHeight]) {
  element.addEventListener("input", () => renderSpatial(true));
}
fields.devicesJson.addEventListener("change", () => {
  try {
    parseDevicesJson();
    refreshEditorState({ syncDevicesJson: true, syncStripsJson: true });
    markDirty();
  } catch (error) {
    statusEl.textContent = error.message || "Devices JSON error";
  }
});
fields.stripsJson.addEventListener("change", () => {
  try {
    parseStripsJson();
    refreshEditorState({ syncDevicesJson: false, syncStripsJson: true });
    markDirty();
  } catch (error) {
    statusEl.textContent = error.message || "Strips JSON error";
  }
});
fields.stripSelect.addEventListener("change", () => selectStrip(fields.stripSelect.value));
for (const input of [
  fields.stripName,
  fields.stripWall,
  fields.stripStartU,
  fields.stripStartV,
  fields.stripEndU,
  fields.stripEndV,
  fields.stripLedCount,
  fields.stripDirection,
  fields.stripDevice,
  fields.stripDeviceStart,
  fields.stripSyncMode,
  fields.stripTvRole,
  fields.stripExtends,
  fields.stripExtensionMode,
  fields.stripExtensionStrength,
  fields.stripExtensionSoftness,
  fields.stripBlend,
]) {
  input.addEventListener("input", updateSelectedStripFromEditor);
}
fields.stripNudgeLeft.addEventListener("click", () => nudgeSelected(-0.05, 0));
fields.stripNudgeRight.addEventListener("click", () => nudgeSelected(0.05, 0));
fields.stripNudgeUp.addEventListener("click", () => nudgeSelected(0, 0.05));
fields.stripNudgeDown.addEventListener("click", () => nudgeSelected(0, -0.05));
fields.deselectStrip.addEventListener("click", deselectStrip);
fields.duplicateStrip.addEventListener("click", duplicateSelectedStrip);
fields.deleteStrip.addEventListener("click", deleteSelectedStrip);
fields.addStrip.addEventListener("click", addStrip);
fields.save.addEventListener("click", saveConfig);
fields.startPreview.addEventListener("click", startPreview);
fields.stopPreview.addEventListener("click", () => stopPreview());
fields.startSimulation.addEventListener("click", startSimulation);
fields.stopSimulation.addEventListener("click", stopSimulation);
fields.stopPattern.addEventListener("click", () => triggerPreviewPattern("idle"));
bindAmbientSpillControls();
for (const input of [
  fields.renderMode,
  fields.targetFps,
  fields.wledFps,
  fields.edgeBandWidth,
  fields.edgeBandHeight,
  fields.edgeBandFraction,
  fields.hybridFullWidth,
  fields.hybridFullHeight,
]) {
  input.addEventListener("change", () => {
    readRenderPerformanceForm();
    markDirty();
  });
}
fields.view2d.addEventListener("click", () => setViewMode("2d"));
fields.view3d.addEventListener("click", () => setViewMode("3d"));
window.addEventListener("resize", () => {
  requestDraw();
  if (activeView === "3d") lightPreview3d.render();
});

canvas.addEventListener("pointerdown", (event) => {
  const rect = canvas.getBoundingClientRect();
  const hit = stripHitAtCanvasPoint(event.clientX - rect.left, event.clientY - rect.top);
  if (hit) {
    beginStripDrag(hit, event.clientX, event.clientY);
    dragging = false;
    lastPointer = null;
    canvas.setPointerCapture(event.pointerId);
    return;
  }
  dragging = true;
  lastPointer = { x: event.clientX, y: event.clientY };
  emptyDragStart = { x: event.clientX, y: event.clientY, moved: false };
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (editDrag) {
    updateStripDrag(event.clientX, event.clientY);
    return;
  }
  if (!dragging || !lastPointer) return;
  if (emptyDragStart) {
    const totalDx = event.clientX - emptyDragStart.x;
    const totalDy = event.clientY - emptyDragStart.y;
    if (!emptyDragStart.moved && Math.hypot(totalDx, totalDy) < 4) return;
    emptyDragStart.moved = true;
  }
  const dx = event.clientX - lastPointer.x;
  const dy = event.clientY - lastPointer.y;
  yaw += dx * 0.008;
  pitch = Math.max(-1.15, Math.min(0.05, pitch + dy * 0.006));
  lastPointer = { x: event.clientX, y: event.clientY };
  requestDraw();
});

canvas.addEventListener("pointerup", () => {
  finishStripDrag();
  if (dragging && emptyDragStart && !emptyDragStart.moved) {
    deselectStrip();
  }
  dragging = false;
  lastPointer = null;
  emptyDragStart = null;
});

canvas.addEventListener("pointercancel", () => {
  finishStripDrag();
  dragging = false;
  lastPointer = null;
  emptyDragStart = null;
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  zoom = Math.max(45, Math.min(320, zoom - event.deltaY * 0.08));
  requestDraw();
}, { passive: false });

loadConfig().catch((error) => {
  statusEl.textContent = String(error);
  resizeCanvas();
  ctx.fillStyle = "#161a20";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#ff9a9a";
  ctx.font = "16px system-ui, sans-serif";
  ctx.fillText("Unable to load room config. Check the server console.", 24, 40);
});
