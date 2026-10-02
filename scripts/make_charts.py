"""Render the results charts used in the README / presentation from benchmark JSON files.

    python scripts/make_charts.py docs/results/spunet-5cm.json docs/results/spunet-10cm.json --out docs/img

Each JSON is the output of `python -m lidar.bench`. The label of a run is its file stem.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#c3c2b7"
STAGES = [("load", "load"), ("inference", "neural net"), ("accumulate", "fusion"), ("objects", "objects"), ("grid", "2.5D grid")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": BASELINE, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlelocation": "left", "axes.titlepad": 14,
})


def style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left" if grid_axis == "x" else "bottom"].set_color(BASELINE)
    ax.spines["bottom" if grid_axis == "x" else "left"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


LABELS = {
    "spunet-10cm-mix": "10 cm + PolarMix", "spunet-5cm-mix": "5 cm + PolarMix",
    "spunet-5cm-narrow-mix": "5 cm narrow + PolarMix", "spunet-5cm": "5 cm", "spunet-10cm": "10 cm",
}


def load(paths):
    return [(LABELS.get(Path(p).stem, Path(p).stem), json.loads(Path(p).read_text())) for p in paths]


def memory_chart(res, out):
    m = res["memory"]
    rows = [("Dense 5 cm 3D voxels", m["dense_3d_bytes_1B_per_voxel"]),
            ("Uniform 5 cm 2.5D grid", m["uniform_2p5d_bytes"]),
            ("FOVEA variable-resolution 2.5D", m["ours_bytes"])]
    fig, ax = plt.subplots(figsize=(8.4, 2.9))
    ys = range(len(rows))
    colors = [NEUTRAL, NEUTRAL, SERIES[0]]
    ax.barh(list(ys), [r[1] for r in rows], height=0.5, color=colors, edgecolor=SURFACE, linewidth=2)
    ax.set_xscale("log")
    ax.set_yticks(list(ys), [r[0] for r in rows], color=INK)
    for y, (_, b) in zip(ys, rows):
        txt = f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB" if b >= 1e7 else f"{b / 1e6:.1f} MB"
        if b != m["ours_bytes"]:
            txt += f"  ({b / m['ours_bytes']:.0f}× more)"
        ax.text(b * 1.15, y, txt, va="center", color=INK, fontsize=11)
    ax.set_xlim(1e6, 3e11)
    ax.set_xlabel("bytes for a 200 m × 200 m map (log scale)")
    ax.set_title("Map memory at equal coverage")
    style(ax, "x")
    fig.tight_layout()
    fig.savefig(out / "chart_memory.png", dpi=160)
    plt.close(fig)


def latency_chart(runs, out):
    fig, ax = plt.subplots(figsize=(8.4, 2.2 + 0.55 * len(runs)))
    xmax = max(115, max(r["latency_ms"]["total"]["p50"] for _, r in runs) * 1.35)
    for y, (name, r) in enumerate(runs):
        left = 0.0
        for k, (key, label) in enumerate(STAGES):
            v = r["latency_ms"].get(key, {}).get("p50", 0.0)
            ax.barh(y, v, left=left, height=0.5, color=SERIES[k], edgecolor=SURFACE, linewidth=2,
                    label=label if y == 0 else None)
            if v > 0.07 * xmax:
                ax.text(left + v / 2, y, f"{v:.0f}", va="center", ha="center", fontsize=9.5,
                        color="white" if k < 2 else INK)  # light fills (aqua, yellow, magenta) get dark ink
            left += v
        total = r["latency_ms"]["total"]["p50"]
        ax.text(left + 1.5, y, f"{total:.0f} ms · {1000 / total:.0f} FPS", va="center", color=INK, fontsize=10.5)
    ax.axvline(100, color=MUTED, linestyle=(0, (4, 4)), linewidth=1)
    ax.text(99, -0.62, "10 Hz sensor budget", color=MUTED, fontsize=9.5, ha="right", va="bottom")
    ax.set_yticks(range(len(runs)), [n for n, _ in runs], color=INK)
    ax.set_xlabel("median latency per frame (ms), RTX 5060 Ti")
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(runs) - 0.5, -0.75)
    ax.legend(ncol=len(STAGES), loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9.5,
              handlelength=1.0, columnspacing=1.2)
    ax.set_title("End-to-end latency by stage", pad=30)
    style(ax, "x")
    fig.tight_layout()
    fig.savefig(out / "chart_latency.png", dpi=160)
    plt.close(fig)


def accuracy_by_distance_chart(runs, out):
    runs = [(n, r) for n, r in runs if "segmentation" in r]
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=False)
    bins = list(runs[0][1]["segmentation"]["by_distance"].keys())
    x = range(len(bins))
    for k, (name, r) in enumerate(runs[:4]):
        seg = [r["segmentation"]["by_distance"][b]["miou"] * 100 for b in bins]
        axes[0].plot(list(x), seg, color=SERIES[k], linewidth=2, marker="o", markersize=7, label=name,
                     markeredgecolor=SURFACE, markeredgewidth=2)
        axes[0].annotate(f"{seg[-1]:.0f}", (len(bins) - 1, seg[-1]), xytext=(8, 0), textcoords="offset points",
                         va="center", color=INK_2, fontsize=9.5)
        veh = [r["objects"]["vehicle"]["by_distance"][b]["recall"] for b in bins]
        pts = [(i, v * 100) for i, v in enumerate(veh) if v is not None]
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[k], linewidth=2, marker="o", markersize=7,
                     label=name, markeredgecolor=SURFACE, markeredgewidth=2)
    axes[0].set_title("Point segmentation, map categories (mIoU %)")
    axes[1].set_title("Vehicle detection recall (%)")
    for ax in axes:
        ax.set_xticks(list(x), [b.replace("m", " m") for b in bins])
        ax.set_xlabel("distance from sensor")
        ax.set_ylim(0, 100)
        style(ax, "y")
    if len(runs) > 1:
        axes[0].legend(frameon=False, fontsize=9.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "chart_accuracy_distance.png", dpi=160)
    plt.close(fig)


def tradeoff_chart(runs, out):
    runs = [(n, r) for n, r in runs if "segmentation" in r]
    if len(runs) < 2:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, r in runs:
        lat, miou = r["latency_ms"]["total"]["p50"], r["segmentation"]["miou_19"] * 100
        ax.plot(lat, miou, "o", color=SERIES[0], markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(name, (lat, miou), xytext=(9, 4), textcoords="offset points", color=INK, fontsize=10)
    ax.set_xlabel("median end-to-end latency (ms)")
    ax.set_ylabel("mIoU, 19 classes (%)")
    ax.set_title("Accuracy vs latency across trained variants")
    style(ax, "y")
    fig.tight_layout()
    fig.savefig(out / "chart_tradeoff.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="docs/img")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load(args.results)
    memory_chart(runs[0][1], out)
    latency_chart(runs, out)
    accuracy_by_distance_chart(runs, out)
    tradeoff_chart(runs, out)
    print("charts written to", out)


if __name__ == "__main__":
    main()
