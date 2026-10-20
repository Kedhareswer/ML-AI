"""Record the real arena UI with Playwright: one screenshot per game plus one screen recording.

The recording is assembled from browser frames captured continuously (~5 fps)
during the session, then encoded with the system ffmpeg.

Start the server first:  arena ui --artifacts artifacts/arena --port 7861
Every screenshot shows a real model response; the page reports the verifier verdict.
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

STEPS = [
    # (tab, button text, wait ms, screenshot name)
    ("sudoku", "Solve: network + backtracking", 6000, "sudoku-search"),
    ("mini_sudoku", "Solve: network only (no backtracking)", 4000, "mini-sudoku-neural"),
    ("tango", "Solve: network only (no backtracking)", 4000, "tango-neural"),
    ("queens", "Solve: network only (no backtracking)", 4000, "queens-neural"),
    ("zip", "Solve: network + backtracking", 6000, "zip-search"),
    ("patches", "Solve: network only (no backtracking)", 4000, "patches-neural"),
    ("wend", "Solve: network + backtracking", 5000, "wend-search"),
    ("chess", "Solve: network only (no search)", 7000, "chess-neural"),
    ("checkers", "Play: network vs material engine", 0, "checkers-game"),
    ("pinpoint", "Reveal clues and guess", 6000, "pinpoint"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7861")
    ap.add_argument("--output", default="evidence/arena")
    ap.add_argument("--chromium", default="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium, args=["--no-sandbox"])
        ctx = browser.new_context(viewport={"width": 1180, "height": 760})
        page = ctx.new_page()
        frames = out / "frames"
        shutil.rmtree(frames, ignore_errors=True)
        frames.mkdir()
        count = [0]

        def film(ms, until=None):
            """Capture frames for `ms` milliseconds, or until the JS condition holds."""
            end = time.time() + ms / 1000
            while time.time() < end or (until and not page.evaluate(until)):
                page.screenshot(path=str(frames / f"{count[0]:05d}.jpg"), type="jpeg", quality=80)
                count[0] += 1
                time.sleep(0.12)
                if until and time.time() > end + 180:
                    break
        page.goto(args.url)
        page.wait_for_selector("nav button")
        for kind, button, wait, name in STEPS:
            page.click(f"nav button[data-k='{kind}']")
            film(1500)
            with page.expect_response(lambda r: "/api/" in r.url and r.request.method == "POST", timeout=180_000) as resp:
                page.click(f"#buttons button:has-text('{button}')")
            body = resp.value.json()
            if kind == "checkers":  # film the whole game playback
                film(1000, "document.getElementById('status').innerText.includes('after')")
            else:
                film(wait)
            page.screenshot(path=str(out / f"{name}.png"))
            summary = {k: body.get(k) for k in ("solved", "mode", "expanded_states", "ms", "result", "plies", "words", "played", "solution")}
            if kind == "pinpoint":
                summary = {"clues": body["clues"], "top_guess_after_each_clue": [g[0][0] if g else None for g in body["after"]]}
            log.append({"tab": kind, "button": button, "status_text": page.inner_text("#status"), "api": summary})
            print(name, json.dumps(summary)[:200], flush=True)
        film(1500)
        ctx.close()
        browser.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5", "-i", str(frames / "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=1180:-2", str(out / "arena-demo.mp4")], check=True)
    shutil.rmtree(frames, ignore_errors=True)
    (out / "captures.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
