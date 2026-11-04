"""Render demo.mp4 (1920 x 1080, H.264): the terminal replay of demo.svg as a video, for X.

    uv run --with matplotlib python docs/assets/make_clip.py

demo.svg types the Evil-BF16 Hub diff and prints the output its receipt recorded
(experiments/E008/diff-default-cap.txt); this script checks that demo.svg is what
make_figures.py draws today, then pauses its CSS animations at each frame time in headless
Chrome, screenshots the frame, and joins the frames with ffmpeg. It is a replay of a recorded
run, and the frame says so. Set CHROME and FFMPEG if they are not in the default places.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from make_figures import HERE, THEMES, demo

W, H, FPS, END, HOLD = 1920, 1080, 15, 13.0, 2.5  # frames 0..END s, then the last held
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"


def page(svg: str, t: float) -> str:
    c = THEMES["dark"]
    return f"""<!doctype html><meta charset="utf-8">
<style>
  * {{ animation-play-state: paused !important; animation-delay: -{t:.4f}s !important; }}
  html, body {{ margin: 0; width: {W}px; height: {H}px; background: {c["bg"]}; overflow: hidden;
               font-family: Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; }}
  main {{ display: flex; flex-direction: column; justify-content: center; height: {H}px;
         padding: 0 80px; box-sizing: border-box; gap: 34px; }}
  h1 {{ margin: 0 0 14px; color: {c["text"]}; font-size: 56px; font-weight: 600; }}
  p.sub {{ color: {c["body"]}; font-size: 30px; }}
  p {{ margin: 0; color: {c["muted"]}; font-size: 27px; }}
  main > svg {{ width: 100%; height: auto; }}
</style>
<main>
  <div><h1>One tensor of 775 changed. Three of four shards never downloaded.</h1>
  <p class="sub">A 9B upload whose card says “finetune” and names no tensor, diffed against its
  base on the Hugging Face Hub.</p></div>
  {svg}
  <p>Replay of a recorded run: the command and the output its receipt recorded, typed out in
  time. Run live, it takes 15 to 40 s from the cache.</p>
</main>
"""


def frame(svg: str, t: float, out: Path) -> None:
    html = out.with_suffix(".html")
    html.write_text(page(svg, t), encoding="utf-8")
    subprocess.run(  # noqa: S603  # the local browser binary, named by the user or the default
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
         "--force-device-scale-factor=1", f"--window-size={W},{H}",
         f"--screenshot={out}", html.as_uri()],
        check=True, capture_output=True, timeout=60,
    )  # fmt: skip


def main() -> None:
    svg = (HERE / "demo.svg").read_text(encoding="utf-8")
    if svg != demo():
        raise SystemExit("demo.svg differs from make_figures.py's output; run that first")
    for tool in (CHROME, FFMPEG):
        if not (Path(tool).exists() or shutil.which(tool)):
            raise SystemExit(f"{tool} not found; set CHROME or FFMPEG")
    out = HERE / "demo.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        n = round(END * FPS) + 1
        paths = [Path(tmp) / f"f{i:04d}.png" for i in range(n)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda i: frame(svg, i / FPS, paths[i]), range(n)))
        for k in range(round(HOLD * FPS)):
            shutil.copy(paths[-1], Path(tmp) / f"f{n + k:04d}.png")
        subprocess.run(  # noqa: S603  # ffmpeg from PATH or FFMPEG
            [FFMPEG, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i",
             str(Path(tmp) / "f%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-preset", "slow", "-crf", "20", "-r", "30", "-movflags", "+faststart",
             "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact", str(out)],
            check=True,
        )  # fmt: skip
        shutil.copy(paths[-1], HERE / "demo-last-frame.png")
    print(f"wrote docs/assets/{out.name} ({out.stat().st_size:,} bytes, {n} frames + hold)")


if __name__ == "__main__":
    main()
