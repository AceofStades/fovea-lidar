"""Render docs/figures/architecture_overview.d2 to SVG (d2) and a 2x PNG (headless Chromium).

    python scripts/figures/render_d2.py

Needs the d2 binary on PATH (go install oss.terrastruct.com/d2@latest) and Playwright's Chromium.
"""
import asyncio
import glob
import os
import re
import subprocess
from pathlib import Path

from playwright.async_api import async_playwright

FIG = Path(__file__).resolve().parents[2] / "docs" / "figures"
SRC = FIG / "architecture_overview.d2"
SVG = FIG / "architecture_overview.svg"
PNG = FIG / "architecture_overview.png"


async def svg_to_png():
    svg = SVG.read_text()
    m = re.search(r'viewBox="[-\d.]+ [-\d.]+ ([\d.]+) ([\d.]+)"', svg)
    w, h = (int(float(v)) + 1 for v in m.groups())
    exe = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")))
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=exe[-1] if exe else None)
        page = await browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
        await page.goto(SVG.as_uri())
        await page.screenshot(path=str(PNG), full_page=False, omit_background=False)
        await browser.close()


def main():
    subprocess.run(["d2", str(SRC), str(SVG)], check=True)
    os.chmod(SVG, 0o644)
    asyncio.run(svg_to_png())
    print("wrote", SVG, "and", PNG)


if __name__ == "__main__":
    main()
