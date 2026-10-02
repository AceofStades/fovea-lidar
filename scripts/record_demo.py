"""Record a scripted walkthrough of the running simulator to MP4 (backup for live demos).

    uv run lidar sim --seq 08 --checkpoint models/spunet-5cm-mix.pt &
    python scripts/record_demo.py --out docs/demo.mp4

The script drives the UI like a presenter would: chase view, the four map layers, top-down
resolution rings, the cell inspector and the orbit view, while the drive replays at 10 Hz.
"""
import argparse
import asyncio
import glob
import os
import shutil
import subprocess
import tempfile

from playwright.async_api import async_playwright

W, H = 1680, 980


def chromium_path():
    hits = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")))
    return hits[-1] if hits else None


async def scene(page, *clicks, wait=4.0):
    for sel in clicks:
        await page.click(sel)
        await page.wait_for_timeout(400)
    await page.wait_for_timeout(int(wait * 1000))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/")
    ap.add_argument("--start-frame", type=int, default=1380)
    ap.add_argument("--out", default="docs/demo.mp4")
    args = ap.parse_args()
    tmp = tempfile.mkdtemp()
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=chromium_path(), headless=True,
                                          args=["--use-angle=vulkan", "--enable-gpu", "--ignore-gpu-blocklist"])
        ctx = await browser.new_context(viewport={"width": W, "height": H}, record_video_dir=tmp,
                                        record_video_size={"width": W, "height": H})
        page = await ctx.new_page()
        await page.goto(f"{args.url}?frame={args.start_frame}&view=chase&color=0")
        await page.wait_for_timeout(3000)
        await scene(page, "#btn-about", wait=6)                                   # what you are looking at
        await scene(page, "#about .close", wait=7)                                # semantic map, chase view
        await scene(page, "#color-mode [data-m='1']", wait=7)                     # drivability
        await scene(page, "#color-mode [data-m='2']", wait=6)                     # height
        await scene(page, "#view-mode [data-v='top']", "#color-mode [data-m='3']", wait=8)  # foveated rings
        await scene(page, "#color-mode [data-m='0']", wait=4)
        await scene(page, "#view-mode [data-v='chase']", wait=3)
        await page.mouse.move(800, 700)                                           # cell inspector
        for x in range(800, 980, 30):
            await page.mouse.move(x, 700 - (x - 800) // 3)
            await page.wait_for_timeout(450)
        await page.mouse.move(W - 10, H // 2)
        await scene(page, "#point-mode [data-p='2']", wait=6)                     # prediction errors
        await scene(page, "#point-mode [data-p='0']", "#view-mode [data-v='orbit']", wait=10)
        video = page.video
        await ctx.close()
        await browser.close()
        webm = await video.path()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", webm, "-c:v", "libx264", "-preset", "slow",
                    "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", args.out)


if __name__ == "__main__":
    asyncio.run(main())
