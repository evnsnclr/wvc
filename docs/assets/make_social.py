"""Render social-preview.png (1280 x 640) from hero.svg, for the repository's social preview.

    uv run --with matplotlib python docs/assets/make_social.py

hero.svg must be what make_figures.py draws from the receipts today; the script checks that,
then shows the hero's final frame (animations off, as for readers who ask for reduced motion)
on its own background with headless Chrome. Set CHROME to the browser binary if it is not in
the default macOS location. GitHub takes the image in Settings > Social preview; upload by hand.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from make_figures import HERE, THEMES, hero

W, H, HERO_W = 1280, 640, 1200
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


def page(svg: str) -> str:
    c = THEMES["dark"]
    body = svg.split("?>", 1)[-1]
    return f"""<!doctype html><meta charset="utf-8">
<style>
  * {{ animation: none !important; }}
  html, body {{ margin: 0; width: {W}px; height: {H}px; background: {c["bg"]}; overflow: hidden; }}
  main {{ display: flex; flex-direction: column; align-items: center; justify-content: center;
         height: {H}px; gap: 26px; }}
  main > svg {{ width: {HERO_W}px; height: auto; }}
  p {{ margin: 0; color: {c["muted"]}; font: 26px ui-monospace, SFMono-Regular, Menlo, monospace; }}
</style>
<main>{body}<p>github.com/evnsnclr/wvc</p></main>
"""


def main() -> None:
    svg = (HERE / "hero.svg").read_text(encoding="utf-8")
    if svg != hero():
        raise SystemExit("hero.svg differs from make_figures.py's output; run that first")
    if not Path(CHROME).exists():
        raise SystemExit(f"no Chrome at {CHROME}; set CHROME")
    out = HERE / "social-preview.png"
    with tempfile.TemporaryDirectory() as tmp:
        html = Path(tmp) / "social.html"
        html.write_text(page(svg), encoding="utf-8")
        shot = Path(tmp) / "shot.png"
        subprocess.run(  # noqa: S603  # the local browser binary, named by the user or the default
            [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
             "--force-device-scale-factor=1", f"--window-size={W},{H}",
             f"--screenshot={shot}", html.as_uri()],
            check=True, capture_output=True, timeout=60,
        )  # fmt: skip
        from PIL import Image

        image = Image.open(shot).convert("RGB")
        if image.size != (W, H):
            raise SystemExit(f"Chrome drew {image.size}, expected {(W, H)}")
        image.save(out, optimize=True)
    print(f"wrote docs/assets/{out.name} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
