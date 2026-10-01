"""Push a training run to Kaggle (2x T4) under one of the team's accounts.

    python scripts/kaggle_launch.py --account ansh --name spunet-5cm -- --voxel 0.05 --max-hours 11
    python scripts/kaggle_launch.py --account ansh --name spunet-5cm --status
    python scripts/kaggle_launch.py --account ansh --name spunet-5cm --fetch runs/kaggle-spunet-5cm

The source tree is embedded in the kernel script, so no extra upload is needed; the data comes from
the public SemanticKITTI copy on Kaggle. Pass --resume-from <user>/<kernel> to continue a previous
run's last.pt in a new 12 h session.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

# Only these accounts belong to the team (see the allow-list the user gave); never glob the folder.
ACCOUNTS = ["own", "ansh", "dave", "dip", "pra", "sid", "tiya", "t1", "vishal", "rid", "yug"]
DATASET = "luischavarriazamora/semantic-kitti"
ROOT = Path(__file__).resolve().parents[1]

RUNNER = r'''
import base64, io, os, shutil, subprocess, sys, tarfile
CODE = "{code}"
TRAIN_ARGS = {train_args}
WORK = "/kaggle/working"
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall(WORK + "/code")
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv; python --version", shell=True)
# cumm/spconv treat themselves as editable installs (and try to JIT-compile) when site-packages
# contains a setup.py or .gitignore, which the Kaggle image does; disable that.
os.environ.update(CUMM_DISABLE_JIT="1", SPCONV_DISABLE_JIT="1")
import site
for sp in site.getsitepackages():
    for f in ("setup.py", ".gitignore"):
        if os.path.exists(os.path.join(sp, f)):
            os.rename(os.path.join(sp, f), os.path.join(sp, f + ".moved"))
            print("moved", os.path.join(sp, f), flush=True)
SMOKE = ("import torch, spconv.pytorch as sp; c = torch.randint(0, 50, (2000, 3)).int().unique(dim=0); "
         "c = torch.cat([torch.zeros(len(c), 1, dtype=torch.int32), c], 1).cuda(); "
         "x = sp.SparseConvTensor(torch.randn(len(c), 8).cuda(), c, [50, 50, 50], 1); "
         "y = sp.SubMConv3d(8, 16, 3, indice_key='a').cuda()(x); print('spconv ok', y.features.shape)")
for whl in ("spconv-cu126", "spconv-cu124", "spconv-cu121"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", whl])
    if subprocess.run([sys.executable, "-c", SMOKE]).returncode == 0:
        print("using", whl, flush=True)
        break
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", whl])
else:
    sys.exit("no working spconv build")
out = WORK + "/run"
os.makedirs(out, exist_ok=True)
for dirpath, dirnames, files in os.walk("/kaggle/input"):
    if "sequences" in dirnames or dirpath.count(os.sep) > 9:
        dirnames[:] = []
    if "last.pt" in files and not os.path.exists(out + "/last.pt"):
        for f in ("last.pt", "best.pt", "log.jsonl"):
            if os.path.exists(os.path.join(dirpath, f)):
                shutil.copy(os.path.join(dirpath, f), out)
        print("resuming from", dirpath, flush=True)
sys.path.insert(0, WORK + "/code/src")
from lidar.data import scan_files, TRAIN_SEQS, VAL_SEQS
n_train, n_val = len(scan_files("/kaggle/input", TRAIN_SEQS)), len(scan_files("/kaggle/input", VAL_SEQS))
print("dataset scans: train", n_train, "val", n_val, flush=True)
if not n_train or not n_val:
    subprocess.run("find /kaggle/input -maxdepth 6 -type d | head -40", shell=True)
    sys.exit("dataset not found")
import torch
ngpu = max(torch.cuda.device_count(), 1)
env = dict(os.environ, PYTHONPATH=WORK + "/code/src", OMP_NUM_THREADS="1")
cmd = ["torchrun", "--standalone", f"--nproc_per_node={{ngpu}}", "-m", "lidar.train",
       "--root", "/kaggle/input", "--out", out] + TRAIN_ARGS
print(" ".join(cmd), flush=True)
rc = subprocess.run(cmd, env=env).returncode
shutil.rmtree(WORK + "/code", ignore_errors=True)
sys.exit(rc)
'''


def config_dir(account):
    if account not in ACCOUNTS:
        raise SystemExit(f"account {account!r} is not on the team allow-list: {ACCOUNTS}")
    return Path.home() / (".kaggle" if account == "own" else f".config/kaggle/{account}")


def username(account):
    d = config_dir(account)
    f = d / ("credentials.json" if account == "own" else "kaggle.json")
    return json.loads(f.read_text())["username"]


def kaggle(account, *args, check=True):
    env = dict(os.environ, KAGGLE_CONFIG_DIR=str(config_dir(account)))
    if account == "own":
        env.pop("KAGGLE_CONFIG_DIR")
    return subprocess.run(["kaggle", *args], env=env, check=check, text=True, capture_output=True)


def code_tarball():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(ROOT / "src" / "lidar", arcname="src/lidar",
                filter=lambda t: None if "__pycache__" in t.name else t)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--name", required=True, help="kernel slug, e.g. spunet-5cm")
    ap.add_argument("--resume-from", default=None, help="<user>/<kernel> whose output holds last.pt")
    ap.add_argument("--gpu", default="NvidiaTeslaT4")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--logs", action="store_true", help="print the log of the latest (finished) run")
    ap.add_argument("--fetch", default=None, help="download the kernel output into this folder")
    ap.add_argument("train_args", nargs="*")
    args = ap.parse_args()
    user = username(args.account)
    ref = f"{user}/{args.name}"

    if args.status:
        print(kaggle(args.account, "kernels", "status", ref, check=False).stdout.strip())
        return
    if args.logs:
        raw = kaggle(args.account, "kernels", "logs", ref, check=False).stdout
        try:
            text = "".join(d.get("data", "") for d in json.loads(raw))
        except ValueError:
            text = raw
        noise = ("warn", "fx_tracing", "return func")
        print("\n".join(l for l in text.splitlines() if l.strip() and not any(n in l.lower() for n in noise)))
        return
    if args.fetch:
        os.makedirs(args.fetch, exist_ok=True)
        r = kaggle(args.account, "kernels", "output", ref, "-p", args.fetch, check=False)
        print(r.stdout[-2000:], r.stderr[-2000:])
        return

    build = ROOT / "kaggle_build" / args.name
    build.mkdir(parents=True, exist_ok=True)
    (build / "run.py").write_text(RUNNER.format(code=code_tarball(), train_args=repr(args.train_args)))
    meta = {
        "id": ref, "title": args.name, "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "dataset_sources": [DATASET], "competition_sources": [],
        "kernel_sources": [args.resume_from] if args.resume_from else [], "model_sources": [],
        "machine_shape": args.gpu,
    }
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    r = kaggle(args.account, "kernels", "push", "-p", str(build), "--accelerator", args.gpu, check=False)
    print(r.stdout.strip(), r.stderr.strip())


if __name__ == "__main__":
    main()
