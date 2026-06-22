import * as THREE from "./vendor/three.module.js?v=three165-live10";
import { OrbitControls } from "./vendor/OrbitControls.js?v=three165-live10";

const LED_POINT_SIZE = 6.5;
const WALL_INSET = 0.018;
const GLOW_SIZE = 0.42;
const LIGHTMAP_SIZE = 512;
const LIGHTMAP_RADIUS_METERS = 0.9;
const LIGHTMAP_ALPHA = 0.2;
const TV_LIGHTMAP_CLEAR_MARGIN = 0.08;
const MAX_DEVICE_PIXEL_RATIO = 2;
const MAX_RENDER_FPS = 90;
const RENDER_FRAME_MS = 1000 / MAX_RENDER_FPS;
const RENDER_STATS_WINDOW = 60;

export function createLightPreview3d({ canvas, fallback, onRenderStats } = {}) {
  let spatial = null;
  let previewColors = [];
  let previewImage = null;
  let active = false;
  let renderer = null;
  let scene = null;
  let camera = null;
  let controls = null;
  let roomGroup = null;
  let ledMesh = null;
  let glowMesh = null;
  let tvMesh = null;
  let tvTexture = null;
  let tvCanvas = null;
  let tvCanvasContext = null;
  let ledEntries = [];
  let wallSurfaces = new Map();
  let surfaceLightmaps = new Map();
  let lastLightmapColors = null;
  let raf = 0;
  let renderTimer = 0;
  let loopRaf = 0;
  let sceneSignature = "";
  let lastImage = null;
  let lastMirror = null;
  let showGlow = true;
  let showMarkers = true;
  let showTv = true;
  let lastLoopRenderAt = 0;
  let lastRenderAt = 0;
  let renderSamples = [];

  if (fallback) fallback.hidden = true;

  return {
    setActive(value) {
      active = Boolean(value);
      if (!active) {
        if (fallback) fallback.hidden = true;
        stopLoop();
        return;
      }
      if (ensureRenderer()) {
        startLoop();
        requestRender();
      }
    },
    setState(next) {
      if (Object.prototype.hasOwnProperty.call(next, "spatial") && next.spatial) spatial = next.spatial;
      if (Object.prototype.hasOwnProperty.call(next, "previewColors")) previewColors = next.previewColors || [];
      if (Object.prototype.hasOwnProperty.call(next, "previewImage")) previewImage = next.previewImage || null;
      if (active && ensureRenderer()) requestRender();
    },
    setOptions(options) {
      showGlow = options.showGlow ?? showGlow;
      showMarkers = options.showMarkers ?? showMarkers;
      showTv = options.showTv ?? showTv;
      if (ledMesh) ledMesh.visible = showMarkers;
      if (glowMesh) glowMesh.visible = showGlow;
      for (const lightmap of surfaceLightmaps.values()) lightmap.mesh.visible = showGlow;
      if (tvMesh) tvMesh.visible = showTv;
      requestRender();
    },
    render,
    stats() {
      return renderStats();
    },
  };

  function ensureRenderer() {
    if (renderer) return true;
    try {
      renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        alpha: false,
        powerPreference: "high-performance",
      });
      renderer.setClearColor(0x10141a, 1);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_DEVICE_PIXEL_RATIO));
      renderer.outputColorSpace = THREE.SRGBColorSpace;

      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x10141a);

      camera = new THREE.PerspectiveCamera(48, 1, 0.01, 1000);
      controls = new OrbitControls(camera, canvas);
      controls.enableDamping = false;
      controls.enablePan = true;
      controls.screenSpacePanning = true;
      controls.addEventListener("change", requestRender);

      scene.add(new THREE.AmbientLight(0xffffff, 0.52));
      const key = new THREE.DirectionalLight(0xdceeff, 1.15);
      key.position.set(-2.5, 5.5, -3.5);
      scene.add(key);

      roomGroup = new THREE.Group();
      scene.add(roomGroup);

      tvCanvas = document.createElement("canvas");
      tvCanvas.width = 2;
      tvCanvas.height = 2;
      tvCanvasContext = tvCanvas.getContext("2d");
      tvTexture = new THREE.CanvasTexture(tvCanvas);
      tvTexture.colorSpace = THREE.SRGBColorSpace;
      drawBlankTvTexture();
      return true;
    } catch (error) {
      console.warn("3D preview WebGL initialization failed", error);
      renderer = null;
      if (fallback) {
        fallback.textContent = "3D preview is unavailable because WebGL rendering is disabled or unsupported.";
        fallback.hidden = false;
      }
      return false;
    }
  }

  function requestRender() {
    if (!active || raf || renderTimer || !renderer) return;
    const elapsed = performance.now() - lastRenderAt;
    const delay = Math.max(0, RENDER_FRAME_MS - elapsed);
    renderTimer = window.setTimeout(() => {
      renderTimer = 0;
      raf = requestAnimationFrame(() => {
        raf = 0;
        render();
      });
    }, delay);
  }

  function startLoop() {
    if (loopRaf) return;
    const tick = (now) => {
      loopRaf = 0;
      if (!active || !renderer) return;
      if (!lastLoopRenderAt || now - lastLoopRenderAt >= RENDER_FRAME_MS) {
        lastLoopRenderAt = now;
        render();
      }
      loopRaf = requestAnimationFrame(tick);
    };
    loopRaf = requestAnimationFrame(tick);
  }

  function stopLoop() {
    if (loopRaf) {
      cancelAnimationFrame(loopRaf);
      loopRaf = 0;
    }
    if (renderTimer) {
      clearTimeout(renderTimer);
      renderTimer = 0;
    }
  }

  function render() {
    if (!active || !spatial || !ensureRenderer()) return;
    const started = performance.now();
    const nextSignature = makeSceneSignature();
    if (nextSignature !== sceneSignature) {
      sceneSignature = nextSignature;
      rebuildScene();
    }
    resizeRenderer();
    updateSeeThroughWall();
    updateTvTexture();
    updateLedColors();
    updateSurfaceLightmaps();
    renderer.render(scene, camera);
    lastRenderAt = performance.now();
    recordRenderSample(performance.now() - started);
  }

  function recordRenderSample(ms) {
    renderSamples.push(ms);
    if (renderSamples.length > RENDER_STATS_WINDOW) renderSamples.shift();
    if (typeof onRenderStats === "function") onRenderStats(renderStats());
  }

  function renderStats() {
    if (!renderSamples.length) return { avgMs: 0, samples: 0, maxFps: MAX_RENDER_FPS };
    const total = renderSamples.reduce((sum, value) => sum + value, 0);
    return {
      avgMs: total / renderSamples.length,
      samples: renderSamples.length,
      maxFps: MAX_RENDER_FPS,
    };
  }

  function rebuildScene() {
    clearGroup(roomGroup);
    wallSurfaces = new Map();
    surfaceLightmaps = new Map();
    lastLightmapColors = null;
    if (ledMesh) {
      disposeObject(ledMesh);
      scene.remove(ledMesh);
      ledMesh = null;
    }
    if (glowMesh) {
      disposeObject(glowMesh);
      scene.remove(glowMesh);
      glowMesh = null;
    }
    if (tvMesh) {
      disposeObject(tvMesh);
      scene.remove(tvMesh);
      tvMesh = null;
    }

    const room = spatial.room;
    buildRoom(room);
    buildTv();
    buildLedMeshes();
    frameCamera(room);
  }

  function buildRoom(room) {
    const wallMaterial = new THREE.MeshStandardMaterial({
      color: 0x202833,
      roughness: 0.85,
      metalness: 0,
      transparent: true,
      opacity: 0.72,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const floorMaterial = new THREE.MeshStandardMaterial({
      color: 0x252b34,
      roughness: 0.92,
      metalness: 0,
      side: THREE.DoubleSide,
    });
    const ceilingMaterial = wallMaterial.clone();
    ceilingMaterial.opacity = 0.32;

    roomGroup.add(makePlane("floor", room.width, room.depth, v(room.width / 2, 0, room.depth / 2), v(1, 0, 0), v(0, 0, 1), v(0, 1, 0), floorMaterial, 0));
    roomGroup.add(makePlane("ceiling", room.width, room.depth, v(room.width / 2, room.height, room.depth / 2), v(1, 0, 0), v(0, 0, -1), v(0, -1, 0), ceilingMaterial, 0));
    roomGroup.add(makeWallSurface("front", room.width, room.height, v(room.width / 2, room.height / 2, 0), v(1, 0, 0), v(0, 1, 0), v(0, 0, 1), wallMaterial.clone()));
    roomGroup.add(makeWallSurface("rear", room.width, room.height, v(room.width / 2, room.height / 2, room.depth), v(-1, 0, 0), v(0, 1, 0), v(0, 0, -1), wallMaterial.clone()));
    roomGroup.add(makeWallSurface("left", room.depth, room.height, v(room.width, room.height / 2, room.depth / 2), v(0, 0, 1), v(0, 1, 0), v(-1, 0, 0), wallMaterial.clone()));
    roomGroup.add(makeWallSurface("right", room.depth, room.height, v(0, room.height / 2, room.depth / 2), v(0, 0, -1), v(0, 1, 0), v(1, 0, 0), wallMaterial.clone()));

    roomGroup.add(makeSurfaceLightmap("front", room.width, room.height, v(room.width / 2, room.height / 2, WALL_INSET * 0.55), v(1, 0, 0), v(0, 1, 0), v(0, 0, 1)));
    roomGroup.add(makeSurfaceLightmap("rear", room.width, room.height, v(room.width / 2, room.height / 2, room.depth - WALL_INSET * 0.55), v(-1, 0, 0), v(0, 1, 0), v(0, 0, -1)));
    roomGroup.add(makeSurfaceLightmap("left", room.depth, room.height, v(room.width - WALL_INSET * 0.55, room.height / 2, room.depth / 2), v(0, 0, 1), v(0, 1, 0), v(-1, 0, 0)));
    roomGroup.add(makeSurfaceLightmap("right", room.depth, room.height, v(WALL_INSET * 0.55, room.height / 2, room.depth / 2), v(0, 0, -1), v(0, 1, 0), v(1, 0, 0)));

    const box = new THREE.BoxGeometry(room.width, room.height, room.depth);
    const edges = new THREE.EdgesGeometry(box);
    const lines = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x7f94a8, transparent: true, opacity: 0.42 }));
    lines.position.set(room.width / 2, room.height / 2, room.depth / 2);
    roomGroup.add(lines);
  }

  function buildTv() {
    const tv = spatial.tv;
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      map: tvTexture,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    tvMesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
    placeWallPlane(tvMesh, tv.wall, tv.center_u, tv.center_v, tv.width, tv.height, WALL_INSET * 2.4);
    tvMesh.visible = showTv;
    tvMesh.renderOrder = 10;
    scene.add(tvMesh);

    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(1.025, 1.025)),
      new THREE.LineBasicMaterial({ color: 0xe9f1f8, transparent: true, opacity: 0.9 }),
    );
    tvMesh.add(frame);
  }

  function buildLedMeshes() {
    ledEntries = collectLedEntries();
    if (!ledEntries.length) return;

    const ledPositions = new Float32Array(ledEntries.length * 3);
    const ledColors = new Float32Array(ledEntries.length * 3);
    ledEntries.forEach((entry, index) => {
      ledPositions[index * 3] = entry.point.x;
      ledPositions[index * 3 + 1] = entry.point.y;
      ledPositions[index * 3 + 2] = entry.point.z;
      ledColors[index * 3] = 0.82;
      ledColors[index * 3 + 1] = 0.86;
      ledColors[index * 3 + 2] = 0.9;
    });
    const ledGeometry = new THREE.BufferGeometry();
    ledGeometry.setAttribute("position", new THREE.BufferAttribute(ledPositions, 3));
    ledGeometry.setAttribute("color", new THREE.BufferAttribute(ledColors, 3).setUsage(THREE.DynamicDrawUsage));
    const ledMaterial = new THREE.PointsMaterial({
      size: LED_POINT_SIZE,
      sizeAttenuation: false,
      vertexColors: true,
      depthTest: false,
      depthWrite: false,
      transparent: true,
      opacity: 1,
      toneMapped: false,
    });
    ledMesh = new THREE.Points(ledGeometry, ledMaterial);
    ledMesh.visible = showMarkers;
    ledMesh.renderOrder = 5;
    scene.add(ledMesh);

    const glowGeometry = new THREE.PlaneGeometry(1, 1);
    const glowMaterial = new THREE.MeshBasicMaterial({
      map: createGlowTexture(),
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.16,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
      vertexColors: true,
      toneMapped: false,
    });
    glowMesh = new THREE.InstancedMesh(glowGeometry, glowMaterial, ledEntries.length);
    glowMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    glowMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(ledEntries.length * 3), 3);
    glowMesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    glowMesh.visible = showGlow;
    glowMesh.renderOrder = 2;
    scene.add(glowMesh);

    const ledDummy = new THREE.Object3D();
    ledEntries.forEach((entry, index) => {
      ledDummy.position.copy(entry.point);
      ledDummy.scale.setScalar(1);
      ledDummy.updateMatrix();

      const matrix = glowMatrix(entry.strip.wall, entry.point, GLOW_SIZE);
      glowMesh.setMatrixAt(index, matrix);
    });
    glowMesh.instanceMatrix.needsUpdate = true;
  }

  function updateLedColors() {
    if (!ledMesh || !glowMesh) return;
    const color = new THREE.Color();
    const glowColor = new THREE.Color();
    const ledColorAttribute = ledMesh.geometry.getAttribute("color");
    ledEntries.forEach((entry, index) => {
      const rgb = previewColors[entry.colorIndex] || [210, 218, 228];
      color.setRGB((rgb[0] || 0) / 255, (rgb[1] || 0) / 255, (rgb[2] || 0) / 255, THREE.SRGBColorSpace);
      ledColorAttribute.setXYZ(index, color.r, color.g, color.b);
      const strength = Math.max(rgb[0] || 0, rgb[1] || 0, rgb[2] || 0) / 255;
      glowColor.copy(color).multiplyScalar(0.18 + strength * 0.82);
      glowMesh.setColorAt(index, glowColor);
    });
    ledColorAttribute.needsUpdate = true;
    if (glowMesh.instanceColor) glowMesh.instanceColor.needsUpdate = true;
  }

  function updateSurfaceLightmaps() {
    if (!showGlow || !surfaceLightmaps.size || previewColors === lastLightmapColors) return;
    lastLightmapColors = previewColors;
    for (const lightmap of surfaceLightmaps.values()) {
      lightmap.context.setTransform(1, 0, 0, 1, 0, 0);
      lightmap.context.clearRect(0, 0, LIGHTMAP_SIZE, LIGHTMAP_SIZE);
      lightmap.context.globalCompositeOperation = "lighter";
    }

    ledEntries.forEach((entry, index) => {
      const color = previewColors[entry.colorIndex] || [210, 218, 228];
      const strength = Math.max(color[0] || 0, color[1] || 0, color[2] || 0) / 255;
      if (strength <= 0.015) return;
      drawLightmapSplat(surfaceLightmaps.get(entry.strip.wall), entry.u, entry.vertical, color, strength);
    });

    for (const lightmap of surfaceLightmaps.values()) {
      lightmap.context.globalCompositeOperation = "source-over";
      clearTvFromLightmap(lightmap);
      lightmap.texture.needsUpdate = true;
    }
  }

  function drawLightmapSplat(lightmap, u, vertical, color, strength) {
    if (!lightmap) return;
    const x = clamp(u / lightmap.width, 0, 1) * LIGHTMAP_SIZE;
    const y = (1 - clamp(vertical / lightmap.height, 0, 1)) * LIGHTMAP_SIZE;
    const metersPerPixel = Math.max(lightmap.width, lightmap.height) / LIGHTMAP_SIZE;
    const radius = Math.max(14, (LIGHTMAP_RADIUS_METERS * (0.55 + strength * 0.55)) / metersPerPixel);
    const alpha = Math.min(0.32, LIGHTMAP_ALPHA * (0.08 + strength * 0.5));
    const gradient = lightmap.context.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, rgba(color, alpha));
    gradient.addColorStop(0.28, rgba(color, alpha * 0.28));
    gradient.addColorStop(0.68, rgba(color, alpha * 0.07));
    gradient.addColorStop(1, rgba(color, 0));
    lightmap.context.fillStyle = gradient;
    lightmap.context.fillRect(x - radius, y - radius, radius * 2, radius * 2);
  }

  function clearTvFromLightmap(lightmap) {
    const tv = spatial?.tv;
    if (!tv || lightmap.name !== tv.wall) return;
    const left = clamp((tv.center_u - tv.width / 2 - TV_LIGHTMAP_CLEAR_MARGIN) / lightmap.width, 0, 1) * LIGHTMAP_SIZE;
    const right = clamp((tv.center_u + tv.width / 2 + TV_LIGHTMAP_CLEAR_MARGIN) / lightmap.width, 0, 1) * LIGHTMAP_SIZE;
    const top = (1 - clamp((tv.center_v + tv.height / 2 + TV_LIGHTMAP_CLEAR_MARGIN) / lightmap.height, 0, 1)) * LIGHTMAP_SIZE;
    const bottom = (1 - clamp((tv.center_v - tv.height / 2 - TV_LIGHTMAP_CLEAR_MARGIN) / lightmap.height, 0, 1)) * LIGHTMAP_SIZE;
    lightmap.context.clearRect(left, top, Math.max(1, right - left), Math.max(1, bottom - top));
  }

  function updateTvTexture() {
    const mirrorInCanvas = false;
    const mirror = "none";
    if (previewImage === lastImage && mirror === lastMirror) return;
    lastImage = previewImage;
    lastMirror = mirror;
    if (!previewImage?.complete || !(previewImage.naturalWidth || previewImage.width)) {
      drawBlankTvTexture();
      return;
    }
    const width = previewImage.naturalWidth || previewImage.width;
    const height = previewImage.naturalHeight || previewImage.height;
    if (tvCanvas.width !== width || tvCanvas.height !== height) {
      tvCanvas.width = width;
      tvCanvas.height = height;
      replaceTvTexture();
    }
    tvCanvasContext.save();
    tvCanvasContext.setTransform(1, 0, 0, 1, 0, 0);
    tvCanvasContext.clearRect(0, 0, width, height);
    if (mirrorInCanvas) {
      tvCanvasContext.translate(width, 0);
      tvCanvasContext.scale(-1, 1);
    }
    tvCanvasContext.drawImage(previewImage, 0, 0, width, height);
    tvCanvasContext.restore();
    tvTexture.needsUpdate = true;
    if (tvMesh?.material) tvMesh.material.needsUpdate = true;
  }

  function drawBlankTvTexture() {
    tvCanvasContext.fillStyle = "#030405";
    tvCanvasContext.fillRect(0, 0, tvCanvas.width, tvCanvas.height);
    tvTexture.needsUpdate = true;
  }

  function replaceTvTexture() {
    if (tvTexture) tvTexture.dispose();
    tvTexture = new THREE.CanvasTexture(tvCanvas);
    tvTexture.colorSpace = THREE.SRGBColorSpace;
    if (tvMesh?.material) {
      tvMesh.material.map = tvTexture;
      tvMesh.material.needsUpdate = true;
    }
  }

  function collectLedEntries() {
    const entries = [];
    let cursor = 0;
    for (const strip of spatial.strips || []) {
      const count = Math.max(1, Number(strip.led_count) || 1);
      for (let i = 0; i < count; i += 1) {
        const colorOffset = strip.direction === "reverse" ? count - 1 - i : i;
        entries.push({ strip, colorIndex: cursor + colorOffset, ...ledWallSample(strip, i, count) });
      }
      cursor += count;
    }
    return entries;
  }

  function ledWallSample(strip, offset, count) {
    const rawT = count === 1 ? 0 : offset / (count - 1);
    const t = rawT;
    const u = Number(strip.start_u || 0) + (Number(strip.end_u || 0) - Number(strip.start_u || 0)) * t;
    const vertical = Number(strip.start_v || 0) + (Number(strip.end_v || 0) - Number(strip.start_v || 0)) * t;
    return { u, vertical, point: wallPoint(strip.wall, u, vertical, WALL_INSET * 1.7) };
  }

  function placeWallPlane(mesh, wall, centerU, centerV, width, height, inset) {
    const basis = wallBasis(wall);
    const center = wallPoint(wall, centerU, centerV, inset);
    const matrix = new THREE.Matrix4().makeBasis(
      basis.u.clone().multiplyScalar(width),
      basis.v.clone().multiplyScalar(height),
      basis.n,
    );
    matrix.setPosition(center);
    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(matrix);
  }

  function glowMatrix(wall, point, size) {
    const basis = wallBasis(wall);
    const matrix = new THREE.Matrix4().makeBasis(
      basis.u.clone().multiplyScalar(size),
      basis.v.clone().multiplyScalar(size),
      basis.n,
    );
    matrix.setPosition(point.clone().add(basis.n.clone().multiplyScalar(0.003)));
    return matrix;
  }

  function wallPoint(wall, u, vertical, inset = WALL_INSET) {
    const room = spatial.room;
    if (wall === "front") return v(u, vertical, inset);
    if (wall === "rear") return v(room.width - u, vertical, room.depth - inset);
    if (wall === "left") return v(room.width - inset, vertical, u);
    return v(inset, vertical, room.depth - u);
  }

  function wallBasis(wall) {
    if (wall === "front") return { u: v(1, 0, 0), v: v(0, 1, 0), n: v(0, 0, 1) };
    if (wall === "rear") return { u: v(-1, 0, 0), v: v(0, 1, 0), n: v(0, 0, -1) };
    if (wall === "left") return { u: v(0, 0, 1), v: v(0, 1, 0), n: v(-1, 0, 0) };
    return { u: v(0, 0, -1), v: v(0, 1, 0), n: v(1, 0, 0) };
  }

  function makePlane(name, width, height, center, uAxis, vAxis, normal, material, renderOrder = 0) {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
    mesh.name = name;
    mesh.renderOrder = renderOrder;
    const matrix = new THREE.Matrix4().makeBasis(
      uAxis.clone().multiplyScalar(width),
      vAxis.clone().multiplyScalar(height),
      normal,
    );
    matrix.setPosition(center);
    mesh.matrixAutoUpdate = false;
    mesh.matrix.copy(matrix);
    return mesh;
  }

  function makeWallSurface(name, width, height, center, uAxis, vAxis, normal, material) {
    const mesh = makePlane(name, width, height, center, uAxis, vAxis, normal, material, 0);
    wallSurfaces.set(name, mesh);
    return mesh;
  }

  function makeSurfaceLightmap(name, width, height, center, uAxis, vAxis, normal) {
    const lightmapCanvas = document.createElement("canvas");
    lightmapCanvas.width = LIGHTMAP_SIZE;
    lightmapCanvas.height = LIGHTMAP_SIZE;
    const context = lightmapCanvas.getContext("2d");
    const texture = new THREE.CanvasTexture(lightmapCanvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const material = new THREE.MeshBasicMaterial({
      map: texture,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      depthTest: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    const mesh = makePlane(`${name}-lightmap`, width, height, center, uAxis, vAxis, normal, material, 1);
    mesh.visible = showGlow;
    surfaceLightmaps.set(name, { name, mesh, context, texture, width, height });
    return mesh;
  }

  function updateSeeThroughWall() {
    const hidden = wallCameraIsOutsideOf();
    for (const [wall, mesh] of wallSurfaces.entries()) {
      mesh.visible = wall !== hidden;
    }
    for (const [wall, lightmap] of surfaceLightmaps.entries()) {
      lightmap.mesh.visible = showGlow && wall !== hidden;
    }
  }

  function wallCameraIsOutsideOf() {
    const room = spatial.room;
    const distances = [
      ["front", Math.max(0, -camera.position.z)],
      ["rear", Math.max(0, camera.position.z - room.depth)],
      ["right", Math.max(0, -camera.position.x)],
      ["left", Math.max(0, camera.position.x - room.width)],
    ];
    distances.sort((a, b) => b[1] - a[1]);
    return distances[0][1] > 0.001 ? distances[0][0] : "";
  }

  function frameCamera(room) {
    const center = v(room.width / 2, room.height / 2, room.depth / 2);
    const span = Math.max(room.width, room.height, room.depth, 1);
    camera.position.set(room.width / 2, room.height * 0.74, room.depth + span * 1.55);
    camera.near = Math.max(0.01, span / 500);
    camera.far = span * 20;
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.minDistance = span * 0.28;
    controls.maxDistance = span * 5;
    controls.update();
  }

  function resizeRenderer() {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    const drawing = renderer.getDrawingBufferSize(new THREE.Vector2());
    const dpr = Math.min(window.devicePixelRatio || 1, MAX_DEVICE_PIXEL_RATIO);
    if (drawing.x !== Math.floor(width * dpr) || drawing.y !== Math.floor(height * dpr)) {
      renderer.setPixelRatio(dpr);
      renderer.setSize(width, height, false);
    }
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function makeSceneSignature() {
    if (!spatial) return "";
    return JSON.stringify({
      room: spatial.room,
      tv: spatial.tv,
      strips: (spatial.strips || []).map((strip) => ({
        id: strip.id,
        wall: strip.wall,
        start_u: strip.start_u,
        start_v: strip.start_v,
        end_u: strip.end_u,
        end_v: strip.end_v,
        led_count: strip.led_count,
        direction: strip.direction,
      })),
    });
  }

  function createGlowTexture() {
    const size = 96;
    const glowCanvas = document.createElement("canvas");
    glowCanvas.width = size;
    glowCanvas.height = size;
    const context = glowCanvas.getContext("2d");
    const gradient = context.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    gradient.addColorStop(0, "rgba(255,255,255,0.82)");
    gradient.addColorStop(0.28, "rgba(255,255,255,0.34)");
    gradient.addColorStop(1, "rgba(255,255,255,0)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, size, size);
    const texture = new THREE.CanvasTexture(glowCanvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }

  function rgba(color, alpha) {
    return `rgba(${Math.round(color[0] || 0)}, ${Math.round(color[1] || 0)}, ${Math.round(color[2] || 0)}, ${alpha})`;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, Number(value) || 0));
  }

  function clearGroup(group) {
    while (group.children.length) {
      const child = group.children[0];
      group.remove(child);
      disposeObject(child);
    }
  }

  function disposeObject(object, options = {}) {
    object.traverse?.((child) => {
      if (child.geometry) child.geometry.dispose();
      if (!options.keepMaterial && child.material) {
        if (Array.isArray(child.material)) child.material.forEach((material) => material.dispose());
        else child.material.dispose();
      }
    });
  }

  function v(x, y, z) {
    return new THREE.Vector3(Number(x) || 0, Number(y) || 0, Number(z) || 0);
  }
}
