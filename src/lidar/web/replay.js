// Static replay: plays a drive recorded by scripts/record_replay.py and packed by
// scripts/pack_replay.py, with no server and no GPU. It stands in for the simulator's WebSocket:
// the viewer sends it the same commands and gets the same info / frame / inspect messages back.
// Not available offline: the raw point cloud, ground-truth labels and changing the fusion window.

const Q = 0.025, ZQ = 0.01, SLAB = 4; // lattice, height step, ground-slab height in steps

function decodeFrame(buf, at) {
  const dv = new DataView(buf);
  let o = at;
  const u32 = () => { const v = dv.getUint32(o, true); o += 4; return v; };
  const hl = u32();
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, o, hl)));
  o += hl;
  const n = u32();
  const lens = [u32(), u32(), u32(), u32(), u32(), u32()];
  const sec = lens.map((len) => { const s = new Uint8Array(buf, o, len); o += len; return s; });
  const tn = u32();
  const trajectory = new Float32Array(buf.slice(o, o + tn * 12));
  o += tn * 12;
  return { header, n, sec, trajectory, end: o };
}

// LEB128 varints into plain numbers (keys reach 2^46, still exact in a double)
function* varints(bytes) {
  let v = 0, mul = 1;
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    v += (b & 0x7f) * mul;
    if (b & 0x80) mul *= 128;
    else { yield v; v = 0; mul = 1; }
  }
}
const unzig = (v) => (v % 2 ? -(v + 1) / 2 : v / 2);

export function createReplay(base, sink) {
  const url = (p) => new URL(p, new URL(base, location.href)).href;
  let info = null;
  const chunks = [];      // ArrayBuffer per chunk, once loaded
  const offsets = [];     // per frame: [chunk, byte offset]
  let index = 0, playing = true, fps = 10, timer = 0, dirty = true;
  let last = null;        // last decoded frame, for inspect

  function expand(f) {
    const { n, sec } = f;
    const sizes = info.levels.map((l) => l.cell);
    const geo = new Float32Array(n * 5);
    const attr = new Uint8Array(n * 4);
    const keys = new Float64Array(n);
    let key = 0, z0 = 0, k = 0;
    const kIt = varints(sec[0]), zIt = varints(sec[1]), hIt = varints(sec[2]);
    for (let i = 0; i < n; i++) {
      key += kIt.next().value;
      z0 += unzig(zIt.next().value);
      const h = unzig(hIt.next().value);
      const lvl = Math.floor(key / 2 ** 44);
      const rest = key - lvl * 2 ** 44;
      const iy = Math.floor(rest / 2 ** 22) - 2 ** 21;
      const ix = (rest % 2 ** 22) - 2 ** 21;
      const size = sizes[lvl];
      k = i * 5;
      geo[k] = (ix + 0.5) * size;
      geo[k + 1] = (iy + 0.5) * size;
      geo[k + 2] = z0 * ZQ;
      geo[k + 3] = (z0 + h) * ZQ;
      geo[k + 4] = size;
      attr[i * 4] = sec[3][i];
      attr[i * 4 + 1] = sec[4][i];
      attr[i * 4 + 2] = lvl;
      attr[i * 4 + 3] = (sec[5][i >> 3] >> (i & 7)) & 1;
      keys[i] = key;
    }
    return { geo, attr, keys };
  }

  function show() {
    const where = offsets[index];
    if (!where || !chunks[where[0]]) { sink.setConn('buffering…', 'err'); return false; }
    const f = decodeFrame(chunks[where[0]], where[1]);
    const { geo, attr, keys } = expand(f);
    const h = { ...f.header, index, frames: info.replay.count, playing, buffers: [] };
    last = { h, geo, attr, keys };
    sink.setConn(`replay · frame ${info.replay.first + index}`, 'ok');
    sink.onFrame({ header: h, arrays: { cells_geo: geo, cells_attr: attr, trajectory: f.trajectory } });
    return true;
  }

  function tick() {
    clearTimeout(timer);
    if (dirty || playing) {
      if (show()) {
        dirty = false;
        if (playing) index = (index + 1) % info.replay.count;
      }
    }
    timer = setTimeout(tick, playing ? 1000 / fps : 50);
  }

  function inspect(x, y) {
    if (!last) return;
    // the viewer raycasts in the sensor frame; cells live in the map frame (rigid transform)
    const a = last.h.sensor_from_map;   // column-major 4x4
    const v = [x - a[12], y - a[13], -1.73 - a[14]];
    const mx = a[0] * v[0] + a[1] * v[1] + a[2] * v[2];
    const my = a[4] * v[0] + a[5] * v[1] + a[6] * v[2];
    const levels = info.levels;
    for (let lvl = 0; lvl < levels.length; lvl++) {
      const size = levels[lvl].cell;
      const ix = Math.floor(mx / size), iy = Math.floor(my / size);
      const key = lvl * 2 ** 44 + (iy + 2 ** 21) * 2 ** 22 + (ix + 2 ** 21);
      // keys are sorted: binary search
      let lo = 0, hi = last.keys.length;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (last.keys[mid] < key) lo = mid + 1; else hi = mid; }
      if (last.keys[lo] !== key) continue;
      const rows = [];
      for (let i = lo; i < last.keys.length && last.keys[i] === key; i++) rows.push(i);
      const g = last.geo, at = last.attr;
      const z = (i) => [g[i * 5 + 2], g[i * 5 + 3]];
      const isSlab = (i) => Math.round((g[i * 5 + 3] - g[i * 5 + 2]) / ZQ) === SLAB;
      const slab = rows.find(isSlab), solid = rows.find((i) => !isSlab(i));
      const ground = slab !== undefined ? z(slab)[1] : z(solid)[0];
      const main = solid ?? slab;
      const over = slab !== undefined && solid !== undefined;
      sink.onInspect({
        type: 'inspect', inside: true, level: lvl, cell: size,
        ring: [lvl ? levels[lvl - 1].half_extent : 0, levels[lvl].half_extent],
        ix, iy, center: [(ix + 0.5) * size, (iy + 0.5) * size],
        count: null, category: at[main * 4], traversability: at[main * 4 + 1],
        ground_z: ground, ground_observed: true,
        obstacle_top: solid !== undefined ? z(solid)[1] - ground : null,
        clearance: over ? z(solid)[0] - ground : null,
        step: null, moving_points: at[main * 4 + 3] ? 1 : 0,
      });
      return;
    }
    sink.onInspect({ type: 'inspect', inside: false });
  }

  async function load() {
    info = await (await fetch(url('info.json'))).json();
    info.frames = info.replay.count;
    sink.onInfo(info);
    tick();
    // fetch chunks in order; playback starts with the first and waits if it catches up
    for (let c = 0; c < info.replay.chunks; c++) {
      const buf = await (await fetch(url(`chunk-${String(c).padStart(3, '0')}.bin`))).arrayBuffer();
      const dv = new DataView(buf);
      const count = dv.getUint32(4, true);
      let o = 8;
      for (let k = 0; k < count; k++) {
        offsets[c * info.replay.per_chunk + k] = [c, o];
        o = decodeFrame(buf, o).end;
      }
      chunks[c] = buf;
      sink.onProgress?.((c + 1) / info.replay.chunks);
    }
  }

  load().catch((e) => sink.setConn(`replay failed: ${e.message}`, 'err'));
  return {
    send(m) {
      switch (m.cmd) {
        case 'play': playing = true; break;
        case 'pause': playing = false; if (last) index = last.h.index; dirty = true; break;
        case 'seek': index = Math.max(0, Math.min(info.replay.count - 1, m.frame | 0)); dirty = true; break;
        case 'step': index = (index + (m.delta || 1) + info.replay.count) % info.replay.count; dirty = true; break;
        case 'speed': fps = m.fps; break;
        case 'inspect': inspect(m.x, m.y); break;
        default: break; // source, fusion window, points: fixed in a recording
      }
    },
  };
}
