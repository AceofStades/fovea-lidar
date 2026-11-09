"""Pack a recorded drive (scripts/record_replay.py) into small files for the static web replay.

Per cell the live simulator sends x, y, z0, z1, size as float32 plus four label bytes (24 B). Here:
  * x, y, size -> the cell's index on its ring's lattice (exact: centres sit at (i + 0.5) * size)
  * z0 -> 1 cm steps, delta-coded along the sorted cells; height z1 - z0 -> 1 cm steps
  * category, traversability, moving -> unchanged
Columns are LEB128/zig-zag varints; the server brotli-compresses the files. Every frame is checked:
decoded positions must match the recording exactly, heights to within 5 mm.

    uv run python scripts/pack_replay.py runs/replay ../portfolio/frontend/public/demos/fovea/data

Chunk file (chunk-NNN.bin, little endian): "FOVC" u32 frames, then per frame:
  u32 header_len, header JSON, u32 cells, u32 len x 6 (key, z0, h, cat, trav, moving bits),
  the six sections, u32 trajectory points, f32[3 * points]
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np

Q = 0.025   # position lattice (half the finest cell)
ZQ = 0.01   # height step
PER_CHUNK = 10


def leb(vals: np.ndarray) -> bytes:
    out = bytearray()
    for v in vals.tolist():
        while True:
            b = v & 0x7F
            v >>= 7
            if v:
                out.append(b | 0x80)
            else:
                out.append(b)
                break
    return bytes(out)


def zigzag(x: np.ndarray) -> np.ndarray:
    return (x << 1) ^ (x >> 63)


def encode_frame(geo, attrs, sizes):
    lvl = attrs[:, 2].astype(np.int64)
    unit = np.round(np.array(sizes) / Q).astype(np.int64)[lvl]
    q = np.round(geo[:, :2] / Q).astype(np.int64)
    ix = np.floor_divide(q[:, 0], unit)
    iy = np.floor_divide(q[:, 1], unit)
    # exactness: the centre must be (i + 0.5) * size
    for a, i in ((geo[:, 0], ix), (geo[:, 1], iy)):
        err = np.abs((i + 0.5) * unit * Q - a).max()
        assert err < 1e-3, f"cell centre off the lattice by {err} m"
    assert np.allclose(geo[:, 4], np.array(sizes)[lvl]), "size does not match the level"
    key = (lvl << 44) | ((iy + 2**21) << 22) | (ix + 2**21)
    z0 = np.round(geo[:, 2] / ZQ).astype(np.int64)
    h = np.round((geo[:, 3] - geo[:, 2]) / ZQ).astype(np.int64)
    order = np.lexsort((z0, key))
    key, z0, h, a = key[order], z0[order], h[order], attrs[order]
    assert np.abs(z0 * ZQ - geo[order, 2]).max() <= ZQ / 2 + 1e-4
    sections = [
        leb(np.diff(key, prepend=0)),
        leb(zigzag(np.diff(z0, prepend=0))),
        leb(zigzag(h)),
        a[:, 0].astype(np.uint8).tobytes(),
        a[:, 1].astype(np.uint8).tobytes(),
        np.packbits(a[:, 3] > 0, bitorder="little").tobytes(),
    ]
    return len(key), sections


def main(src: Path, out: Path):
    info = json.loads((src / "info.json").read_text())
    sizes = [lv["cell"] for lv in info["levels"]]
    frames = sorted(p.stem for p in src.glob("[0-9][0-9][0-9][0-9].npz"))
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("chunk-*.bin"):
        old.unlink()
    lat = []
    first = json.loads((src / f"{frames[0]}.json").read_text())["index"]
    total = 0
    for c in range(0, len(frames), PER_CHUNK):
        blob = bytearray(b"FOVC" + struct.pack("<I", len(frames[c:c + PER_CHUNK])))
        for name in frames[c:c + PER_CHUNK]:
            z = np.load(src / f"{name}.npz")
            h = json.loads((src / f"{name}.json").read_text())
            lat.append(h["pipeline_ms"])
            h.update(type="frame", latency_p50=float(np.percentile(lat, 50)), latency_p95=float(np.percentile(lat, 95)),
                     replay_index=h["index"] - first)
            hj = json.dumps(h, separators=(",", ":")).encode()
            n, secs = encode_frame(z["geo"], z["attrs"], sizes)
            traj = z["traj"].astype("<f4")
            blob += struct.pack("<I", len(hj)) + hj + struct.pack("<I", n) + struct.pack("<6I", *map(len, secs))
            for s in secs:
                blob += s
            blob += struct.pack("<I", len(traj)) + traj.tobytes()
        path = out / f"chunk-{c // PER_CHUNK:03d}.bin"
        path.write_bytes(blob)
        total += len(blob)
    info.update(frames=len(frames), replay={"first": first, "count": len(frames), "per_chunk": PER_CHUNK,
                                             "chunks": (len(frames) + PER_CHUNK - 1) // PER_CHUNK, "fps": 10})
    info["source"] = "model"
    (out / "info.json").write_text(json.dumps(info, separators=(",", ":")))
    print(f"{len(frames)} frames -> {(len(frames) + PER_CHUNK - 1) // PER_CHUNK} chunks, {total / 1e6:.1f} MB before brotli")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
