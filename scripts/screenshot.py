"""Headless screenshot of the running simulator (for README images and visual checks).

    python scripts/screenshot.py --out docs/sim.png --wait 6 [--click "#color-mode [data-m='1']"]
"""
import argparse
import asyncio
import glob
import os

from playwright.async_api import async_playwright


def chromium_path():
    hits = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")))
    return hits[-1] if hits else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/")
    ap.add_argument("--out", default="screenshot.png")
    ap.add_argument("--wait", type=float, default=6.0)
    ap.add_argument("--width", type=int, default=1680)
    ap.add_argument("--height", type=int, default=980)
    ap.add_argument("--click", action="append", default=[], help="CSS selector to click before the shot")
    args = ap.parse_args()
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=chromium_path(), headless=True,
                                          args=["--use-angle=vulkan", "--enable-gpu", "--ignore-gpu-blocklist",
                                                "--enable-unsafe-swiftshader"])
        page = await browser.new_page(viewport={"width": args.width, "height": args.height})
        logs = []
        page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
        page.on("pageerror", lambda e: logs.append(f"pageerror: {e}"))
        await page.goto(args.url)
        await page.wait_for_timeout(2000)
        for sel in args.click:
            await page.click(sel)
            await page.wait_for_timeout(800)
        await page.wait_for_timeout(args.wait * 1000)
        await page.screenshot(path=args.out)
        await browser.close()
        for line in logs[-20:]:
            print(line)


if __name__ == "__main__":
    asyncio.run(main())
