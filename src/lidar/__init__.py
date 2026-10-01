"""FOVEA: adaptive variable-resolution 2.5D LiDAR mapping."""
import importlib
import sys

COMMANDS = {
    "sim": ("lidar.sim.server", "live simulator (WebSocket server + web UI)"),
    "train": ("lidar.train", "train the sparse U-Net"),
    "bench": ("lidar.bench", "latency / memory / accuracy benchmark"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("usage: lidar {sim,train,bench} [options]\n")
        for name, (_, help_text) in COMMANDS.items():
            print(f"  {name:6s} {help_text}")
        raise SystemExit(2)
    name = sys.argv[1]
    sys.argv = [f"lidar {name}"] + sys.argv[2:]
    importlib.import_module(COMMANDS[name][0]).main()
