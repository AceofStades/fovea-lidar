"""Download only the SemanticKITTI pieces we need, using HTTP range requests.

The official KITTI velodyne archive is a single 80 GB zip. We read its central
directory remotely and pull individual scans, so we can fetch sequence 08 first
(validation / demo) and skip the unlabeled test sequences 11-21 entirely.
"""
import argparse
import os
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlretrieve

from remotezip import RemoteZip

VELO_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
CALIB_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip"
LABELS_URL = "http://www.semantic-kitti.org/assets/data_odometry_labels.zip"

_local = threading.local()


def remote_zip():
    if not hasattr(_local, "zf"):
        _local.zf = RemoteZip(VELO_URL)
    return _local.zf


def fetch(info, root):
    out = os.path.join(root, info.filename)
    if os.path.exists(out) and os.path.getsize(out) == info.file_size:
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".part"
    with remote_zip().open(info) as src, open(tmp, "wb") as dst:
        dst.write(src.read())
    os.replace(tmp, out)
    return info.file_size


def fetch_small_zip(url, root, keep):
    path = os.path.join(root, os.path.basename(url))
    if not os.path.exists(path):
        print(f"downloading {url}", flush=True)
        urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if keep(m)]
        zf.extractall(root, members)
    print(f"extracted {len(members)} files from {os.path.basename(url)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/semantickitti")
    ap.add_argument("--seqs", default="08,00,01,02,03,04,05,06,07,09,10")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    seqs = args.seqs.split(",")
    os.makedirs(args.root, exist_ok=True)

    in_seqs = lambda m: any(f"sequences/{s}/" in m for s in seqs)
    fetch_small_zip(CALIB_URL, args.root, in_seqs)
    fetch_small_zip(LABELS_URL, args.root, in_seqs)

    infos = [i for i in remote_zip().infolist() if i.filename.endswith(".bin") and in_seqs(i.filename)]
    order = {s: k for k, s in enumerate(seqs)}
    infos.sort(key=lambda i: (order[i.filename.split("sequences/")[1][:2]], i.filename))
    total = sum(i.file_size for i in infos)
    print(f"{len(infos)} scans, {total / 1e9:.1f} GB", flush=True)

    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, i, args.root) for i in infos]
        for n, fut in enumerate(as_completed(futures), 1):
            done += fut.result()
            if n % 500 == 0 or n == len(infos):
                print(f"{n}/{len(infos)} scans, {done / 1e9:.1f} GB fetched", flush=True)


if __name__ == "__main__":
    sys.exit(main())
