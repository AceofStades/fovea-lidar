"""Publication-style figure of the sparse 3D U-Net, drawn with PlotNeuralNet (TikZ).

    python scripts/figures/model_figure.py            # -> docs/figures/model_architecture.{tex,pdf,png}

Needs: PlotNeuralNet (cloned to ~/.cache/fovea-tools/PlotNeuralNet if missing), the tectonic LaTeX
engine on PATH, poppler's pdftoppm, and a CUDA GPU for the real input/output panels (the final
model segments one scan of sequence 08).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
PNN = Path(os.environ.get("PLOTNEURALNET", Path.home() / ".cache" / "fovea-tools" / "PlotNeuralNet"))
PNN_REPO = "https://github.com/HarisIqbal88/PlotNeuralNet.git"

# box height/depth per voxel size and band width per channel count
SIZE = {"10~cm": 40, "20~cm": 32, "40~cm": 25, "80~cm": 17, "1.6~m": 10}
WIDTH = {32: 2.0, 64: 2.8, 96: 3.2, 128: 3.8, 256: 5.0}


def ensure_plotneuralnet():
    if not (PNN / "pycore" / "tikzeng.py").exists():
        PNN.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", PNN_REPO, str(PNN)], check=True)
    # vendor the four TikZ style files (MIT) next to the figure so the .tex builds anywhere
    dst = OUT / "plotneuralnet"
    (dst / "layers").mkdir(parents=True, exist_ok=True)
    for f in (PNN / "layers").glob("*"):
        shutil.copy(f, dst / "layers" / f.name)
    shutil.copy(PNN / "LICENSE", dst / "LICENSE")
    sys.path.insert(0, str(PNN))


def scan_panels(frame=880):
    """Bird's-eye images of one real scan: raw points by height, and the final model's prediction."""
    import torch
    from lidar.data import read_scan, scan_files
    from lidar.infer import Segmenter
    from lidar.labels import CATEGORY_COLORS, train_to_category

    files = scan_files(ROOT / "data" / "semantickitti", ["08"])
    pts = read_scan(files[frame][0])
    seg = Segmenter(ROOT / "models" / "spunet-10cm-mix.pt")
    pred, _ = seg(torch.from_numpy(pts).cuda())
    cat = train_to_category(np.minimum(pred.cpu().numpy(), 255).astype(np.uint8))
    keep = (np.abs(pts[:, 0]) < 22) & (np.abs(pts[:, 1]) < 22)
    p, c = pts[keep], cat[keep]
    order = np.argsort(p[:, 2])  # draw low points first so walls and cars stay visible

    def panel(colors, name):
        fig = plt.figure(figsize=(4, 4), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(p[order, 0], p[order, 1], s=1.6, c=colors[order], linewidths=0)
        ax.set_xlim(-22, 22); ax.set_ylim(-22, 22); ax.set_aspect("equal"); ax.axis("off")
        fig.patch.set_facecolor("#0b111a")
        fig.savefig(OUT / name, facecolor=fig.get_facecolor())
        plt.close(fig)

    z = np.clip((p[:, 2] + 2.0) / 4.0, 0, 1)
    panel(plt.cm.plasma(0.15 + 0.85 * z), "scan_input.png")
    panel(CATEGORY_COLORS[c] / 255.0, "scan_output.png")


def architecture():
    from pycore.tikzeng import (to_begin, to_connection, to_Conv, to_ConvConvRelu, to_ConvRes, to_ConvSoftMax,
                                to_cor, to_end, to_input, to_Pool, to_skip, to_UnPool)

    def head():
        return r"""
\documentclass[border=12pt, multi, tikz]{standalone}
\usepackage{import}
\subimport{plotneuralnet/layers/}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d}
"""

    def down(name, after, res):
        s = SIZE[res]
        return to_Pool(name, offset="(0,0,0)", to=f"({after}-east)", width=1, height=s - 6, depth=s - 6, opacity=0.5)

    def enc(name, after, res, ch, caption):
        s = SIZE[res]
        w = WIDTH[ch]
        return [to_ConvConvRelu(name, s_filer="", n_filer=("", ""), offset="(2.7,0,0)", to=f"({after}-east)",
                                width=(w, w), height=s, depth=s, caption=caption + r"\\" + res),
                to_connection(after, name)]

    def dec(name, after, res, ch, skip_ch, caption):
        s = SIZE[res]
        w = WIDTH[ch]
        return [to_UnPool(f"up_{name}", offset="(2.2,0,0)", to=f"({after}-east)", width=1, height=s, depth=s, opacity=0.5),
                to_ConvRes(f"cat_{name}", s_filer="", n_filer="", offset="(0,0,0)", to=f"(up_{name}-east)",
                           width=WIDTH[skip_ch], height=s, depth=s, opacity=0.35),
                to_ConvConvRelu(name, s_filer="", n_filer=("", ""), offset="(0,0,0)", to=f"(cat_{name}-east)",
                                width=(w, w), height=s, depth=s, caption=caption + r"\\" + res),
                to_connection(after, f"up_{name}")]

    legend = r"""
\begin{scope}[shift={(0.5,-8.6,0)}, every node/.style={font=\Huge, anchor=west}]
\fill[fill=\ConvColor] (0,0) rectangle ++(0.8,0.8); \node at (1.0,0.4) {residual block};
\fill[fill=\PoolColor, opacity=0.6] (11,0) rectangle ++(0.8,0.8); \node at (12.0,0.4) {$\downarrow$2 sparse conv};
\fill[fill=\UnpoolColor, opacity=0.6] (24,0) rectangle ++(0.8,0.8); \node at (25.0,0.4) {$\uparrow$2 inverse conv};
\draw[copyconnection] (37,0.4) -- ++(0.8,0); \node at (38.0,0.4) {skip (concat)};
\end{scope}
\node[font=\Huge\bfseries, anchor=west] at (-3,9.0,0) {Sparse 3D U-Net \textperiodcentered{} 10 cm voxels \textperiodcentered{} 67.7\,\% mIoU};
"""

    arch = [
        head(), to_cor(), to_begin(), "\\tikzset{every edge quotes/.append style={font=\\Huge}}\n",
        to_input("scan_input.png", to="(-3.2,0,0)", width=7, height=7, name="scan"),
        to_Conv("stem", s_filer="", n_filer="", offset="(0,0,0)", to="(0,0,0)", width=WIDTH[32],
                height=SIZE["10~cm"], depth=SIZE["10~cm"], caption=r"Stem\\10~cm"),
        down("down1", "stem", "10~cm"),
        *enc("enc1", "down1", "20~cm", 32, "Enc~1"),
        down("down2", "enc1", "20~cm"),
        *enc("enc2", "down2", "40~cm", 64, "Enc~2"),
        down("down3", "enc2", "40~cm"),
        *enc("enc3", "down3", "80~cm", 128, "Enc~3"),
        down("down4", "enc3", "80~cm"),
        *enc("enc4", "down4", "1.6~m", 256, "Enc~4"),
        *dec("dec1", "enc4", "80~cm", 256, 128, "Dec~1"),
        to_skip(of="enc3", to="cat_dec1", pos=1.25),
        *dec("dec2", "dec1", "40~cm", 128, 64, "Dec~2"),
        to_skip(of="enc2", to="cat_dec2", pos=1.25),
        *dec("dec3", "dec2", "20~cm", 96, 32, "Dec~3"),
        to_skip(of="enc1", to="cat_dec3", pos=1.25),
        *dec("dec4", "dec3", "10~cm", 96, 32, "Dec~4"),
        to_skip(of="stem", to="cat_dec4", pos=1.25),
        to_ConvSoftMax("head", s_filer="", offset="(2.6,0,0)", to="(dec4-east)", width=2.4,
                       height=SIZE["10~cm"], depth=SIZE["10~cm"], caption=r"Head\\19~classes"),
        to_connection("dec4", "head"),
        # output panel: the image node is pinned to a zy plane, so place it through a shifted scope
        r"""
\begin{scope}[shift={($(head-east)+(3.6,0,0)$)}]
\node[canvas is zy plane at x=0] (pred) at (0,0,0) {\includegraphics[width=7cm,height=7cm]{scan_output.png}};
\end{scope}
""",
        legend,
        to_end(),
    ]
    return "".join(arch).replace(r"\usetikzlibrary{3d}", "\\usetikzlibrary{3d}\n\\usetikzlibrary{calc}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_plotneuralnet()
    scan_panels()
    tex = OUT / "model_architecture.tex"
    tex.write_text(architecture())
    subprocess.run(["tectonic", "--keep-logs", "--outdir", str(OUT), str(tex)], check=True, cwd=OUT)
    subprocess.run(["pdftoppm", "-png", "-r", "170", "-singlefile", str(OUT / "model_architecture.pdf"),
                    str(OUT / "model_architecture")], check=True)
    print("wrote", tex.with_suffix(".pdf"), "and", tex.with_suffix(".png"))


if __name__ == "__main__":
    main()
