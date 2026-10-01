import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const CAP_CELLS = 900000;
const CAP_POINTS = 200000;
const SENSOR_H = 1.73;
const BG = 0x070a0f;
const STAGES = [
  ['load', '#64748b', 'load'], ['inference', '#a78bfa', 'neural net'], ['accumulate', '#38bdf8', 'fusion'],
  ['grid', '#2ecc71', '2.5D grid'], ['objects', '#f5b942', 'objects'],
];
const LEVEL_COLORS = [[167, 139, 250], [56, 189, 248], [45, 212, 191], [250, 204, 21], [251, 146, 60], [244, 114, 182]];
const $ = (id) => document.getElementById(id);

let info = null;
const ui = { colorMode: 0, pointMode: 0, view: 'chase', playing: true, lastPanel: 0, scrubbing: false };
const latencies = [];
const recvTimes = [];

// ------------------------------------------------------------------ renderer / scene
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(BG);
$('viewport').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(BG, 120, 300);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 3000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495;
scene.add(new THREE.HemisphereLight(0xbfd4ff, 0x10141c, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-20, -30, 50);
scene.add(sun);

const vec3Array = (rows) => rows.map((c) => new THREE.Vector3(c[0] / 255, c[1] / 255, c[2] / 255));
const padTo = (arr, n) => { const out = arr.slice(); while (out.length < n) out.push(new THREE.Vector3(0.3, 0.3, 0.3)); return out; };

// ------------------------------------------------------------------ 2.5D map cells (instanced, shader-coloured)
const unitBox = new THREE.BoxGeometry(1, 1, 1);
const cellGeo = new THREE.InstancedBufferGeometry();
cellGeo.index = unitBox.index;
cellGeo.setAttribute('position', unitBox.getAttribute('position'));
cellGeo.setAttribute('normal', unitBox.getAttribute('normal'));
const geoBuf = new THREE.InstancedInterleavedBuffer(new Float32Array(CAP_CELLS * 5), 5).setUsage(THREE.DynamicDrawUsage);
cellGeo.setAttribute('iXY', new THREE.InterleavedBufferAttribute(geoBuf, 2, 0));
cellGeo.setAttribute('iZ', new THREE.InterleavedBufferAttribute(geoBuf, 2, 2));
cellGeo.setAttribute('iSize', new THREE.InterleavedBufferAttribute(geoBuf, 1, 4));
const attrBuf = new THREE.InstancedBufferAttribute(new Uint8Array(CAP_CELLS * 4), 4).setUsage(THREE.DynamicDrawUsage);
cellGeo.setAttribute('iAttr', attrBuf);
cellGeo.instanceCount = 0;

const TURBO = `
vec3 turbo(float x) {
  x = clamp(x, 0.0, 1.0);
  const vec4 kr = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kg = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kb = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kr2 = vec2(-152.94239396, 59.28637943);
  const vec2 kg2 = vec2(4.27729857, 2.82956604);
  const vec2 kb2 = vec2(-89.90310912, 27.34824973);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(dot(v4, kr) + dot(v2, kr2), dot(v4, kg) + dot(v2, kg2), dot(v4, kb) + dot(v2, kb2));
}`;

const cellMat = new THREE.ShaderMaterial({
  uniforms: {
    uMode: { value: 0 },
    uCat: { value: padTo([], 8) }, uTrav: { value: padTo([], 8) }, uLevel: { value: padTo(vec3Array(LEVEL_COLORS), 8) },
    uGroundZ: { value: -SENSOR_H },
    uFogColor: { value: new THREE.Color(BG) }, uFogNear: { value: 120 }, uFogFar: { value: 300 },
  },
  vertexShader: `
    attribute vec2 iXY; attribute vec2 iZ; attribute float iSize; attribute vec4 iAttr;
    uniform int uMode; uniform vec3 uCat[8]; uniform vec3 uTrav[8]; uniform vec3 uLevel[8]; uniform float uGroundZ;
    varying vec3 vColor; varying vec3 vN; varying vec3 vLocal; varying float vDepth;
    ${TURBO}
    void main() {
      float h = max(iZ.y - iZ.x, 0.002);
      vec3 p = vec3(iXY + position.xy * iSize * 0.96, iZ.x + (position.z + 0.5) * h);
      int cat = int(iAttr.x + 0.5), trav = int(iAttr.y + 0.5), lvl = int(iAttr.z + 0.5);
      vec3 c;
      if (uMode == 0) c = uCat[cat];
      else if (uMode == 1) c = uTrav[trav];
      else if (uMode == 2) c = turbo((iZ.y - uGroundZ + 0.5) / 4.5);
      else c = uLevel[lvl];
      if (uMode == 0 && iAttr.w > 0.5) c = mix(c, vec3(1.0, 0.15, 0.35), 0.6);
      vColor = c; vN = normal; vLocal = position;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      vDepth = -mv.z;
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    uniform vec3 uFogColor; uniform float uFogNear; uniform float uFogFar;
    varying vec3 vColor; varying vec3 vN; varying vec3 vLocal; varying float vDepth;
    void main() {
      vec3 L = normalize(vec3(-0.35, -0.5, 1.0));
      float shade = 0.42 + 0.58 * max(dot(normalize(vN), L), 0.0);
      vec3 c = vColor * shade;
      if (vN.z > 0.5) {               // tile outline on the top face makes the cell size visible
        float e = max(abs(vLocal.x), abs(vLocal.y));
        float w = fwidth(e);
        float line = smoothstep(0.5 - 1.6 * w, 0.5, e);
        c = mix(c, c * 0.35, line * clamp(1.0 - w * 10.0, 0.0, 1.0));
      }
      gl_FragColor = vec4(mix(c, uFogColor, smoothstep(uFogNear, uFogFar, vDepth)), 1.0);
    }`,
});
const cells = new THREE.Mesh(cellGeo, cellMat);
cells.frustumCulled = false;
scene.add(cells);

// ------------------------------------------------------------------ raw points
const ptGeo = new THREE.BufferGeometry();
const ptPos = new THREE.BufferAttribute(new Float32Array(CAP_POINTS * 3), 3).setUsage(THREE.DynamicDrawUsage);
const ptCat = new THREE.BufferAttribute(new Uint8Array(CAP_POINTS), 1).setUsage(THREE.DynamicDrawUsage);
const ptGt = new THREE.BufferAttribute(new Uint8Array(CAP_POINTS), 1).setUsage(THREE.DynamicDrawUsage);
ptGeo.setAttribute('position', ptPos);
ptGeo.setAttribute('pCat', ptCat);
ptGeo.setAttribute('pGt', ptGt);
ptGeo.setDrawRange(0, 0);
const ptMat = new THREE.ShaderMaterial({
  uniforms: { uMode: { value: 0 }, uCat: { value: padTo([], 8) }, uScale: { value: renderer.getPixelRatio() } },
  vertexShader: `
    attribute float pCat; attribute float pGt;
    uniform int uMode; uniform vec3 uCat[8]; uniform float uScale;
    varying vec3 vColor;
    void main() {
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = clamp(70.0 / -mv.z, 1.2, 3.5) * uScale;
      int c = int(pCat + 0.5), g = int(pGt + 0.5);
      if (uMode == 0) vColor = uCat[c] * 1.25;
      else if (uMode == 1) vColor = uCat[g] * 1.25;
      else vColor = g == 0 ? vec3(0.22) : (c == g ? vec3(0.30, 0.36, 0.44) : vec3(1.0, 0.1, 0.85));
    }`,
  fragmentShader: `
    varying vec3 vColor;
    void main() {
      vec2 d = gl_PointCoord - 0.5;
      if (dot(d, d) > 0.25) discard;
      gl_FragColor = vec4(vColor, 1.0);
    }`,
});
const points = new THREE.Points(ptGeo, ptMat);
points.frustumCulled = false;
scene.add(points);

// ------------------------------------------------------------------ ego vehicle, sweep, trajectory
const ego = new THREE.Group();
const bodyMat = new THREE.MeshStandardMaterial({ color: 0xe6edf7, metalness: 0.35, roughness: 0.45 });
const glassMat = new THREE.MeshStandardMaterial({ color: 0x0f1a2a, metalness: 0.6, roughness: 0.2 });
const body = new THREE.Mesh(new THREE.BoxGeometry(4.3, 1.8, 0.75), bodyMat);
body.position.set(0.0, 0, -SENSOR_H + 0.3 + 0.375);
const cabin = new THREE.Mesh(new THREE.BoxGeometry(2.3, 1.62, 0.62), glassMat);
cabin.position.set(-0.35, 0, -SENSOR_H + 0.3 + 0.75 + 0.31);
const puck = new THREE.Mesh(new THREE.CylinderGeometry(0.16, 0.16, 0.22, 24), new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0b4a6b }));
puck.rotation.x = Math.PI / 2;
puck.position.set(0, 0, -0.1);
ego.add(body, cabin, puck);
scene.add(ego);

const sweep = new THREE.Mesh(
  new THREE.CircleGeometry(60, 48, 0, 0.5),
  new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.07, depthWrite: false, blending: THREE.AdditiveBlending }));
sweep.position.z = -SENSOR_H + 0.05;
scene.add(sweep);

const trajGeo = new THREE.BufferGeometry();
trajGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(300 * 3), 3));
const traj = new THREE.Line(trajGeo, new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.85 }));
traj.frustumCulled = false;
scene.add(traj);

// ------------------------------------------------------------------ resolution rings + HTML labels
const rings = new THREE.Group();
scene.add(rings);
const labelLayer = $('labels');
const ringLabels = [];
function buildRings() {
  rings.clear();
  ringLabels.forEach((l) => l.el.remove());
  ringLabels.length = 0;
  let inner = 0;
  info.levels.forEach((lv, k) => {
    const h = lv.half_extent;
    const pts = [[-h, -h], [h, -h], [h, h], [-h, h], [-h, -h]].map(([x, y]) => new THREE.Vector3(x, y, -SENSOR_H + 0.08));
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const c = LEVEL_COLORS[k];
    const line = new THREE.Line(g, new THREE.LineDashedMaterial({ color: new THREE.Color(c[0] / 255, c[1] / 255, c[2] / 255), dashSize: 1.2, gapSize: 0.8, transparent: true, opacity: 0.8 }));
    line.computeLineDistances();
    rings.add(line);
    const el = document.createElement('div');
    el.className = 'tag ring';
    el.style.color = `rgb(${c})`;
    el.textContent = `${Math.round(lv.cell * 100)} cm cells · ${inner}–${h} m`;
    labelLayer.appendChild(el);
    ringLabels.push({ el, pos: new THREE.Vector3(h * 0.72, -h, -SENSOR_H + 0.1) });
    inner = h;
  });
}

// object boxes (pooled)
const boxEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
const boxPool = [];
const boxLabels = [];
function setBoxes(boxes) {
  while (boxPool.length < boxes.length) {
    const m = new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({ color: 0xffffff }));
    scene.add(m);
    boxPool.push(m);
    const el = document.createElement('div');
    el.className = 'tag';
    labelLayer.appendChild(el);
    boxLabels.push({ el, pos: new THREE.Vector3() });
  }
  const show = $('lay-boxes').checked;
  boxPool.forEach((m, i) => {
    const b = boxes[i];
    const lab = boxLabels[i];
    if (!b || !show) { m.visible = false; lab.el.style.display = 'none'; lab.active = false; return; }
    m.visible = true;
    m.position.set(...b.center);
    m.scale.set(...b.size);
    m.rotation.set(0, 0, b.yaw);
    const c = info.category_colors[b.category];
    m.material.color.setRGB(c[0] / 255, c[1] / 255, c[2] / 255);
    if (b.moving) m.material.color.setRGB(1, 0.3, 0.4);
    lab.active = true;
    lab.pos.set(b.center[0], b.center[1], b.center[2] + b.size[2] / 2 + 0.3);
    const name = info.categories[b.category];
    lab.el.className = 'tag' + (b.moving ? ' mv' : '');
    lab.el.textContent = `${name} #${b.id} · ${b.distance.toFixed(0)} m` + (b.moving ? ` · ${(b.speed * 3.6).toFixed(0)} km/h` : '');
  });
}

const tmpV = new THREE.Vector3();
function placeLabels() {
  const w = innerWidth, h = innerHeight;
  const place = (lab, visible) => {
    if (!visible) { lab.el.style.display = 'none'; return; }
    tmpV.copy(lab.pos).project(camera);
    if (tmpV.z > 1 || Math.abs(tmpV.x) > 1.05 || Math.abs(tmpV.y) > 1.05) { lab.el.style.display = 'none'; return; }
    lab.el.style.display = 'block';
    lab.el.style.left = `${(tmpV.x * 0.5 + 0.5) * w}px`;
    lab.el.style.top = `${(-tmpV.y * 0.5 + 0.5) * h}px`;
  };
  const showRings = $('lay-rings').checked;
  ringLabels.forEach((l) => place(l, showRings));
  boxLabels.forEach((l) => place(l, l.active && $('lay-boxes').checked && camera.position.distanceTo(l.pos) < 90));
}

// ------------------------------------------------------------------ camera presets
const VIEWS = {
  chase: { pos: [-17, -3, 8.5], target: [14, 0, -SENSOR_H], rotate: false },
  top: { pos: [-0.5, 0, 110], target: [0, 0, -SENSOR_H], rotate: false },
  orbit: { pos: [-45, -38, 32], target: [0, 0, -SENSOR_H], rotate: true },
};
let camTween = null;
function setView(v) {
  ui.view = v;
  const p = VIEWS[v];
  camTween = { from: camera.position.clone(), fromT: controls.target.clone(), to: new THREE.Vector3(...p.pos), toT: new THREE.Vector3(...p.target), t0: performance.now() };
  controls.autoRotate = p.rotate;
  controls.autoRotateSpeed = 0.6;
  document.querySelectorAll('#view-mode button').forEach((b) => b.classList.toggle('on', b.dataset.v === v));
}
camera.position.set(...VIEWS.chase.pos);
controls.target.set(...VIEWS.chase.target);

// ------------------------------------------------------------------ websocket
let ws = null;
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }
function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => setConn('live', 'ok');
  ws.onclose = () => { setConn('reconnecting…', 'err'); setTimeout(connect, 1500); };
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') onInfo(JSON.parse(ev.data));
    else onFrame(decode(ev.data));
  };
}
function setConn(text, cls) {
  const c = $('chip-conn');
  c.className = `chip ${cls}`;
  c.querySelector('span').textContent = text;
}

const DTYPES = { float32: Float32Array, uint8: Uint8Array, int32: Int32Array, float64: Float64Array, uint16: Uint16Array, int16: Int16Array };
function decode(buf) {
  const hlen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen)));
  const base = 4 + hlen;
  const arrays = {};
  for (const b of header.buffers) arrays[b.name] = new DTYPES[b.dtype](buf, base + b.offset, b.length);
  return { header, arrays };
}

function onInfo(msg) {
  info = msg;
  $('chip-seq').textContent = `SEQ ${msg.sequence} · ${msg.frames} frames`;
  $('scrub').max = msg.frames - 1;
  const cat = vec3Array(msg.category_colors);
  cellMat.uniforms.uCat.value = padTo(cat, 8);
  ptMat.uniforms.uCat.value = padTo(cat, 8);
  cellMat.uniforms.uTrav.value = padTo(vec3Array(msg.traversability_colors), 8);
  const modelBtn = document.querySelector('#source-mode [data-s=model]');
  modelBtn.disabled = !msg.has_model;
  $('model-hint').textContent = msg.has_model
    ? `Sparse 3D U-Net, ${Math.round(msg.model_voxel * 100)} cm voxels` + (msg.model_val_miou ? `, val mIoU ${(msg.model_val_miou * 100).toFixed(1)}%` : '')
    : 'No checkpoint loaded — showing ground-truth labels.';
  setSourceButtons(msg.source);
  $('acc').value = msg.accumulate;
  $('acc-val').textContent = msg.accumulate;
  buildRings();
  buildMemory();
  buildLegend();
}

function setSourceButtons(src) {
  document.querySelectorAll('#source-mode button').forEach((b) => b.classList.toggle('on', b.dataset.s === src));
  $('chip-source').textContent = src === 'model' ? 'NEURAL NET' : 'GROUND TRUTH';
}

function onFrame({ header: h, arrays: a }) {
  // cells
  const geo = a.cells_geo, attr = a.cells_attr;
  const n = Math.min(geo.length / 5, CAP_CELLS);
  geoBuf.array.set(geo.subarray(0, n * 5));
  geoBuf.clearUpdateRanges(); geoBuf.addUpdateRange(0, n * 5); geoBuf.needsUpdate = true;
  attrBuf.array.set(attr.subarray(0, n * 4));
  attrBuf.clearUpdateRanges(); attrBuf.addUpdateRange(0, n * 4); attrBuf.needsUpdate = true;
  cellGeo.instanceCount = n;

  // points
  if (a.points) {
    const m = Math.min(a.points.length / 3, CAP_POINTS);
    ptPos.array.set(a.points.subarray(0, m * 3));
    ptPos.clearUpdateRanges(); ptPos.addUpdateRange(0, m * 3); ptPos.needsUpdate = true;
    ptCat.array.set(a.point_cat.subarray(0, m));
    ptCat.clearUpdateRanges(); ptCat.addUpdateRange(0, m); ptCat.needsUpdate = true;
    if (a.point_gt) {
      ptGt.array.set(a.point_gt.subarray(0, m));
      ptGt.clearUpdateRanges(); ptGt.addUpdateRange(0, m); ptGt.needsUpdate = true;
    }
    ptGeo.setDrawRange(0, m);
  }

  // trajectory (sensor origin -> ground)
  const t = a.trajectory;
  const tp = trajGeo.getAttribute('position');
  const k = Math.min(t.length / 3, 300);
  for (let i = 0; i < k; i++) tp.setXYZ(i, t[i * 3], t[i * 3 + 1], t[i * 3 + 2] - SENSOR_H + 0.12);
  tp.needsUpdate = true;
  trajGeo.setDrawRange(0, k);

  setBoxes(h.boxes);
  latencies.push(h.pipeline_ms);
  if (latencies.length > 150) latencies.shift();
  const now = performance.now();
  recvTimes.push(now);
  while (recvTimes.length && now - recvTimes[0] > 2000) recvTimes.shift();

  if (!ui.scrubbing) $('scrub').value = h.index;
  $('frame-label').textContent = `${h.index} / ${h.frames - 1}`;
  if (h.source !== undefined) setSourceButtons(h.source);
  ui.playing = h.playing;
  $('btn-play').textContent = h.playing ? '⏸' : '▶';
  if (now - ui.lastPanel > 200) { ui.lastPanel = now; updatePanels(h); }
}

// ------------------------------------------------------------------ panels
const fmtBytes = (b) => b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} kB`;

function buildMemory() {
  const m = info.memory;
  const rows = [
    ['FOVEA variable-res 2.5D', m.ours_bytes, '#2ecc71', `${(m.ours_cells_allocated / 1e3).toFixed(0)}k cells`],
    ['Uniform 5 cm 2.5D grid', m.uniform_2p5d_bytes, '#f5b942', `${(m.uniform_2p5d_cells / 1e6).toFixed(0)}M cells`],
    ['Dense 5 cm 3D voxels', m.dense_3d_bytes_1B_per_voxel, '#ff4757', `${(m.dense_3d_voxels / 1e9).toFixed(2)}B voxels`],
  ];
  const lmax = Math.log10(rows[2][1]);
  $('mem').innerHTML = rows.map(([name, b, c, sub]) => `
    <div class="bar"><div class="row"><span>${name} <span class="n">· ${sub}</span></span><span class="v">${fmtBytes(b)}</span></div>
    <div class="track"><div class="fill" style="width:${(Math.log10(b) / lmax * 100).toFixed(1)}%;background:${c}"></div></div></div>`).join('')
    + `<div class="mem-note"><b>${(m.uniform_2p5d_bytes / m.ours_bytes).toFixed(0)}×</b> smaller than a uniform 5 cm 2.5D grid, `
    + `<b>${(m.dense_3d_bytes_1B_per_voxel / m.ours_bytes).toFixed(0)}×</b> smaller than dense 3D voxels (1 byte/voxel, 8 m tall), `
    + `same 200 m × 200 m coverage. Bars are log-scaled.</div>`;
}

function buildLegend() {
  let html = '<h3>Legend</h3>';
  if (ui.colorMode === 0) {
    info.categories.forEach((n, i) => { if (i) html += item(info.category_colors[i], n); });
    html += item([255, 38, 89], 'moving (ground truth)');
  } else if (ui.colorMode === 1) {
    info.traversability.forEach((n, i) => { if (i) html += item(info.traversability_colors[i], n); });
  } else if (ui.colorMode === 2) {
    html += '<div style="height:10px;border-radius:5px;background:linear-gradient(90deg,#30123b,#4686fb,#1be5b5,#a4fc3c,#fb8022,#7a0403)"></div>'
      + '<div class="row" style="display:flex;justify-content:space-between;color:var(--muted);font-size:11px;margin-top:4px"><span>−0.5 m</span><span>height above road</span><span>4 m</span></div>';
  } else {
    info.levels.forEach((lv, k) => { html += item(LEVEL_COLORS[k], `${Math.round(lv.cell * 100)} cm cells (≤ ${lv.half_extent} m)`); });
  }
  $('legend').innerHTML = html;
  function item(c, n) { return `<div class="item"><span class="sw" style="background:rgb(${c})"></span>${n}</div>`; }
}

function updatePanels(h) {
  const p50 = h.latency_p50, p95 = h.latency_p95;
  $('kpi-fps').textContent = (1000 / p50).toFixed(0);
  $('kpi-p50').textContent = p50.toFixed(1);
  $('kpi-p95').textContent = p95.toFixed(1);
  $('chip-fps').innerHTML = `<b>${(1000 / p50).toFixed(0)}</b> FPS`;
  $('chip-lat').innerHTML = `<b>${p50.toFixed(1)}</b> ms`;

  const total = STAGES.reduce((s, [k]) => s + (h.times[k] || 0), 0) || 1;
  $('stage-bar').innerHTML = STAGES.map(([k, c]) => `<div style="width:${((h.times[k] || 0) / total * 100).toFixed(1)}%;background:${c}"></div>`).join('');
  $('stage-legend').innerHTML = STAGES.map(([k, c, n]) => `<span><i style="background:${c}"></i>${n} <b>${(h.times[k] || 0).toFixed(1)}</b></span>`).join('');
  drawSpark();

  // rings table
  let inner = 0;
  $('rings').innerHTML = '<tr><th>cell</th><th>range</th><th>observed</th><th>alloc</th></tr>' + info.levels.map((lv, k) => {
    const w = Math.round(2 * lv.half_extent / lv.cell);
    const holeW = Math.round(2 * inner / lv.cell);
    const row = `<tr><td><span class="sw" style="background:rgb(${LEVEL_COLORS[k]})"></span>${Math.round(lv.cell * 100)} cm</td>
      <td>${inner}–${lv.half_extent} m</td><td>${(h.cells_per_level[k] / 1e3).toFixed(1)}k</td><td>${((w * w - holeW * holeW) / 1e3).toFixed(0)}k</td></tr>`;
    inner = lv.half_extent;
    return row;
  }).join('');

  // accuracy
  const met = h.metrics;
  $('acc-section').style.display = met ? '' : 'none';
  if (met) {
    $('kpi-miou').textContent = `${(met.miou_19 * 100).toFixed(1)}%`;
    const cv = Object.values(met.category_iou).filter((v) => v !== null);
    $('kpi-cat').textContent = `${(cv.reduce((s, v) => s + v, 0) / cv.length * 100).toFixed(1)}%`;
    $('acc-points').textContent = `${(met.points_evaluated / 1e6).toFixed(1)}M pts`;
    $('dist-bars').innerHTML = Object.entries(met.by_distance).filter(([, v]) => v.points > 0)
      .map(([k, v]) => bar(k, v.miou, '#38bdf8', `${(v.points / 1e3).toFixed(0)}k pts`)).join('');
    $('cat-bars').innerHTML = Object.entries(met.category_iou).filter(([, v]) => v !== null)
      .map(([k, v]) => bar(k, v, `rgb(${info.category_colors[info.categories.indexOf(k)]})`)).join('');
  }

  // objects
  const boxes = [...h.boxes].sort((a, b) => a.distance - b.distance);
  $('obj-count').textContent = `${boxes.length} · ${boxes.filter((b) => b.moving).length} moving`;
  $('objects').innerHTML = boxes.length ? boxes.slice(0, 14).map((b) => `
    <div class="obj"><span class="sw" style="background:rgb(${info.category_colors[b.category]})"></span>
    <span>${info.categories[b.category]} #${b.id}<span class="badge ${b.moving ? 'mv' : 'st'}">${b.moving ? 'MOVING' : 'STATIC'}</span></span>
    <span class="d">${b.distance.toFixed(1)} m</span><span class="s">${(b.speed * 3.6).toFixed(0)} km/h</span></div>`).join('')
    : '<div class="empty">No vehicles or pedestrians in view.</div>';
}

function bar(name, v, color, note = '') {
  return `<div class="bar"><div class="row"><span>${name} <span class="n">${note}</span></span><span class="v">${(v * 100).toFixed(1)}%</span></div>
    <div class="track"><div class="fill" style="width:${(v * 100).toFixed(1)}%;background:${color}"></div></div></div>`;
}

function drawSpark() {
  const c = $('spark'), g = c.getContext('2d');
  const W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  const ymax = Math.max(120, ...latencies) * 1.1;
  const y = (v) => H - (v / ymax) * H;
  g.setLineDash([6, 6]); g.strokeStyle = 'rgba(255,71,87,0.6)'; g.lineWidth = 2;
  g.beginPath(); g.moveTo(0, y(100)); g.lineTo(W, y(100)); g.stroke();
  g.setLineDash([]);
  if (latencies.length < 2) return;
  const step = W / (150 - 1);
  const grad = g.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, 'rgba(46,204,113,0.35)'); grad.addColorStop(1, 'rgba(46,204,113,0)');
  g.beginPath();
  latencies.forEach((v, i) => (i ? g.lineTo(i * step, y(v)) : g.moveTo(0, y(v))));
  g.strokeStyle = '#2ecc71'; g.lineWidth = 2.5; g.stroke();
  g.lineTo((latencies.length - 1) * step, H); g.lineTo(0, H); g.closePath(); g.fillStyle = grad; g.fill();
}

// ------------------------------------------------------------------ controls wiring
function segment(id, attr, fn) {
  document.querySelectorAll(`#${id} button`).forEach((b) => b.addEventListener('click', () => {
    if (b.disabled) return;
    document.querySelectorAll(`#${id} button`).forEach((x) => x.classList.toggle('on', x === b));
    fn(b.dataset[attr]);
  }));
}
segment('view-mode', 'v', setView);
segment('color-mode', 'm', (m) => { ui.colorMode = +m; cellMat.uniforms.uMode.value = +m; if (info) buildLegend(); });
segment('point-mode', 'p', (p) => { ui.pointMode = +p; ptMat.uniforms.uMode.value = +p; });
segment('source-mode', 's', (s) => send({ cmd: 'source', source: s }));
$('lay-cells').onchange = (e) => { cells.visible = e.target.checked; };
$('lay-points').onchange = (e) => { points.visible = e.target.checked; send({ cmd: 'points', on: e.target.checked }); };
$('lay-rings').onchange = (e) => { rings.visible = e.target.checked; };
$('lay-traj').onchange = (e) => { traj.visible = e.target.checked; };
$('lay-boxes').onchange = () => {};
$('acc').oninput = (e) => { $('acc-val').textContent = e.target.value; };
$('acc').onchange = (e) => send({ cmd: 'accumulate', frames: +e.target.value });
$('btn-play').onclick = () => send({ cmd: ui.playing ? 'pause' : 'play' });
$('btn-prev').onclick = () => { send({ cmd: 'pause' }); send({ cmd: 'step', delta: -1 }); };
$('btn-next').onclick = () => { send({ cmd: 'pause' }); send({ cmd: 'step', delta: 1 }); };
$('speed').onchange = (e) => send({ cmd: 'speed', fps: +e.target.value });
$('scrub').addEventListener('pointerdown', () => { ui.scrubbing = true; });
$('scrub').addEventListener('pointerup', () => { ui.scrubbing = false; });
$('scrub').onchange = (e) => send({ cmd: 'seek', frame: +e.target.value });
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
  if (e.code === 'Space') { e.preventDefault(); $('btn-play').click(); }
  if (e.code === 'ArrowRight') $('btn-next').click();
  if (e.code === 'ArrowLeft') $('btn-prev').click();
  if (['Digit1', 'Digit2', 'Digit3', 'Digit4'].includes(e.code)) document.querySelectorAll('#color-mode button')[+e.code.slice(-1) - 1].click();
});
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ------------------------------------------------------------------ render loop
function animate(t) {
  requestAnimationFrame(animate);
  if (camTween) {
    const u = Math.min((performance.now() - camTween.t0) / 700, 1);
    const e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
    camera.position.lerpVectors(camTween.from, camTween.to, e);
    controls.target.lerpVectors(camTween.fromT, camTween.toT, e);
    if (u >= 1) camTween = null;
  }
  sweep.rotation.z = -t * 0.0042;
  controls.update();
  renderer.render(scene, camera);
  placeLabels();
}
setView('chase');
connect();
requestAnimationFrame(animate);
