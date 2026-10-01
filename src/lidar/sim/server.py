"""Live simulator: replays a SemanticKITTI drive through the full pipeline and streams every frame
to the browser over a WebSocket.

    python -m lidar.sim.server --seq 08 --checkpoint runs/spunet-5cm/best.pt
    open http://localhost:8000
"""
import argparse
import asyncio
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..grid import OVERHANG, TRAVERSABILITY_COLORS, TRAVERSABILITY_NAMES, GridConfig
from ..labels import CATEGORY_NAMES, CATEGORY_COLORS, TRAIN_TO_CATEGORY
from ..pipeline import Pipeline

WEB = Path(__file__).resolve().parent.parent / "web"
GROUND_SLAB = 0.04


def encode(header, buffers):
    """[u32 header length][header json][pad to 8][buffer 0][pad]... ; header lists dtype/offset/length."""
    specs, blobs, offset = [], [], 0
    for name, arr in buffers:
        arr = np.ascontiguousarray(arr)
        specs.append({"name": name, "dtype": arr.dtype.name, "offset": offset, "length": int(arr.size)})
        blob = arr.tobytes()
        pad = (-len(blob)) % 8
        blobs.append(blob + b"\0" * pad)
        offset += len(blob) + pad
    header["buffers"] = specs
    h = json.dumps(header).encode()
    h += b" " * ((-(len(h) + 4)) % 8)
    return struct.pack("<I", len(h)) + h + b"".join(blobs)


def cell_instances(pipe, frame):
    """Observed cells -> box instances (x, y, z0, z1, size) + (category, traversability, level, moving).
    Obstacles are extruded from the ground to their measured top; overhangs float at their clearance
    with a separate ground tile underneath, so the 2.5D structure is visible."""
    obs = torch.nonzero(frame.count > 0).squeeze(1)
    xy = pipe.cell_xy[obs]
    size = pipe.cell_size[obs]
    g = frame.ground_z[obs]
    top = torch.nan_to_num(frame.obstacle_top[obs], neginf=0.0)
    clear = torch.nan_to_num(frame.clearance[obs], posinf=0.0)
    trav = frame.traversability[obs]
    has_obs = torch.isfinite(frame.clearance[obs])
    z0 = torch.where(has_obs, g + torch.where(trav == OVERHANG, clear, torch.zeros_like(g)), g - GROUND_SLAB)
    z1 = torch.where(has_obs, g + top.clamp_min(0.1), g)
    attrs = torch.stack([frame.category[obs], trav, pipe.cell_level[obs],
                         (frame.moving_count[obs] > 0).to(torch.uint8)], 1)
    geo = torch.stack([xy[:, 0], xy[:, 1], z0, z1, size], 1)
    # ground tiles under passable overhangs
    under = trav == OVERHANG
    if under.any():
        tile_attrs = attrs[under].clone()
        tile_attrs[:, 0] = 1  # drivable-looking ground
        geo = torch.cat([geo, torch.stack([xy[under, 0], xy[under, 1], g[under] - GROUND_SLAB, g[under], size[under]], 1)])
        attrs = torch.cat([attrs, tile_attrs])
    return geo.float().cpu().numpy(), attrs.to(torch.uint8).cpu().numpy()


def inspect_cell(pipe, frame, x, y):
    """Every layer of the cell containing (x, y) in the latest frame."""
    g = pipe.grid
    flat, level = g.cell_index(torch.tensor([[x, y]], device=pipe.device))
    k = int(level[0])
    if k >= g.num_levels:
        return {"type": "inspect", "inside": False}
    f = int(flat[0])
    local = f - g.offsets[k]
    w = g.widths[k]
    val = lambda t: float(t[f])
    finite = lambda v: v if np.isfinite(v) else None
    return {
        "type": "inspect", "inside": True, "level": k, "cell": g.cell_sizes[k],
        "ring": [g.cfg.half_extents[k - 1] if k else 0.0, g.cfg.half_extents[k]],
        "ix": local % w, "iy": local // w, "center": pipe.cell_xy[f].tolist(),
        "count": int(frame.count[f]), "category": int(frame.category[f]),
        "traversability": int(frame.traversability[f]),
        "ground_z": val(frame.ground_z), "ground_observed": bool(frame.ground_observed[f]),
        "obstacle_top": finite(val(frame.obstacle_top)), "clearance": finite(val(frame.clearance)),
        "step": val(frame.step), "moving_points": int(frame.moving_count[f]),
    }


class Session:
    def __init__(self, pipe, fps):
        self.pipe = pipe
        self.index = 0
        self.playing = True
        self.fps = fps
        self.send_points = True
        self.dirty = True
        self.last_frame = None
        self.sensor_from_map = np.eye(4)
        self.lock = asyncio.Lock()


def build_app(pipe, fps=10.0):
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    static_info = {
        "type": "info",
        "sequence": pipe.seq,
        "frames": len(pipe),
        "source": pipe.source,
        "has_model": pipe.segmenter is not None,
        "model_val_miou": getattr(pipe.segmenter, "val_miou", None),
        "model_voxel": getattr(pipe.segmenter, "voxel", None),
        "levels": [{"cell": s, "half_extent": h} for s, h in zip(pipe.grid.cell_sizes, pipe.grid.cfg.half_extents)],
        "memory": pipe.grid.memory_report(),
        "categories": CATEGORY_NAMES, "category_colors": CATEGORY_COLORS.tolist(),
        "traversability": TRAVERSABILITY_NAMES, "traversability_colors": TRAVERSABILITY_COLORS.tolist(),
        "vehicle_height": pipe.grid.cfg.vehicle_height,
        "accumulate": pipe.accumulate,
        "anchor": pipe.anchor,
    }

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        await socket.accept()
        s = Session(pipe, fps)
        pipe.reset_metrics()
        pipe.prev_index = None
        await socket.send_text(json.dumps(static_info))
        loop = asyncio.get_running_loop()

        async def receive():
            while True:
                msg = json.loads(await socket.receive_text())
                cmd = msg.get("cmd")
                if cmd == "play":
                    s.playing = True
                elif cmd == "pause":
                    s.playing = False
                elif cmd == "seek":
                    s.index = int(np.clip(msg["frame"], 0, len(pipe) - 1)); s.dirty = True
                elif cmd == "step":
                    s.index = int(np.clip(s.index + msg.get("delta", 1), 0, len(pipe) - 1)); s.dirty = True
                elif cmd == "speed":
                    s.fps = float(msg["fps"])
                elif cmd == "source" and msg["source"] in ("model", "gt"):
                    if msg["source"] == "model" and pipe.segmenter is None:
                        continue
                    pipe.source = msg["source"]; pipe.prev_index = None; s.dirty = True
                elif cmd == "accumulate":
                    pipe.accumulate = int(np.clip(msg["frames"], 1, 30)); pipe.prev_index = None; s.dirty = True
                elif cmd == "points":
                    s.send_points = bool(msg["on"])
                elif cmd == "reset_metrics":
                    pipe.reset_metrics()
                elif cmd == "inspect" and s.last_frame is not None:
                    # the client raycasts in the sensor frame; the grid lives in the map frame
                    q = np.linalg.inv(s.sensor_from_map) @ np.array([float(msg["x"]), float(msg["y"]), -1.73, 1.0])
                    reply = inspect_cell(pipe, s.last_frame, float(q[0]), float(q[1]))
                    async with s.lock:
                        await socket.send_text(json.dumps(reply))

        async def produce():
            last_sent = time.perf_counter()
            while True:
                if not (s.playing or s.dirty):
                    await asyncio.sleep(0.02)
                    continue
                s.dirty = False
                t0 = time.perf_counter()
                out = await loop.run_in_executor(None, pipe.step, s.index)
                t1 = time.perf_counter()
                geo, attrs = await loop.run_in_executor(None, cell_instances, pipe, out["frame"])
                bufs = [("cells_geo", geo), ("cells_attr", attrs)]
                if s.send_points:
                    pts = out["pts"][:, :3].cpu().numpy()
                    bufs.append(("points", pts))
                    bufs.append(("point_cat", out["category"].to(torch.uint8).cpu().numpy()))
                    if out["gt"] is not None:
                        gt = out["gt"].cpu().numpy()
                        gt_cat = np.where(gt == 255, 0, TRAIN_TO_CATEGORY[np.minimum(gt, 18)]).astype(np.uint8)
                        bufs.append(("point_gt", gt_cat))
                inv = np.linalg.inv(out["pose"])
                lo = max(0, s.index - 80)
                traj = (inv @ pipe.poses[lo:s.index + 1, :, 3].T).T[:, :3].astype(np.float32)
                bufs.append(("trajectory", traj))
                header = {
                    "type": "frame", "index": s.index, "frames": len(pipe), "source": pipe.source,
                    "times": {k: round(v, 3) for k, v in out["times"].items()},
                    "encode_ms": round((time.perf_counter() - t1) * 1000, 3),
                    "pipeline_ms": round(out["pipeline_ms"], 3),
                    "wall_ms": round((t1 - t0) * 1000, 3),
                    "boxes": out["boxes"],
                    "num_points": int(len(out["pts"])),
                    "accumulated_points": out["accumulated_points"],
                    "observed_cells": int(len(geo)),
                    "cells_per_level": np.bincount(attrs[:, 2], minlength=pipe.grid.num_levels).tolist(),
                    "metrics": pipe.metrics_summary(),
                    "latency_p50": float(np.percentile(pipe.latency_hist, 50)) if pipe.latency_hist else out["pipeline_ms"],
                    "latency_p95": float(np.percentile(pipe.latency_hist, 95)) if pipe.latency_hist else out["pipeline_ms"],
                    "sensor_from_map": out["sensor_from_map"].T.reshape(-1).tolist(),  # column-major for three.js
                    "playing": s.playing,
                    "accumulate": pipe.accumulate,
                }
                # pace to the requested replay rate (10 Hz = real sensor rate)
                wait = 1.0 / max(s.fps, 0.1) - (time.perf_counter() - last_sent)
                if s.playing and wait > 0:
                    await asyncio.sleep(wait)
                s.last_frame = out["frame"]
                s.sensor_from_map = out["sensor_from_map"]
                async with s.lock:
                    await socket.send_bytes(encode(header, bufs))
                last_sent = time.perf_counter()
                if s.playing:
                    s.index = (s.index + 1) % len(pipe)

        try:
            await asyncio.gather(receive(), produce())
        except (WebSocketDisconnect, RuntimeError):
            pass

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/semantickitti")
    ap.add_argument("--seq", default="08")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--accumulate", type=int, default=10)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--anchor", choices=["world", "ego"], default="world",
                    help="world: cell lattice fixed in the world (scrolling rings); ego: grid turns with the vehicle")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    pipe = Pipeline(args.root, args.seq, args.checkpoint, GridConfig(), args.accumulate, anchor=args.anchor)
    print(f"sequence {args.seq}: {len(pipe)} frames, source={pipe.source}; open http://{args.host}:{args.port}")
    uvicorn.run(build_app(pipe, args.fps), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
