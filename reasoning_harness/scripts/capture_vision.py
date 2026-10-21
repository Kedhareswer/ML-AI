"""Record the UI solving puzzles from uploaded screenshots (real browser, real models).

The uploaded images are made fresh here: held-out puzzles screenshotted in a
second browser tab (dark mode, JPEG-compressed, rescaled), and a Lichess board
image in a piece set that the classifiers never saw.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from playwright.sync_api import sync_playwright  # noqa: E402

from scripts.bench_vision import lichess_image, phone_like  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7861")
    ap.add_argument("--output", default="evidence/arena/vision")
    ap.add_argument("--chromium", default="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(11)
    frames = out / "frames"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir()
    log = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium, args=["--no-sandbox"])
        maker = browser.new_page(viewport={"width": 1000, "height": 900}, color_scheme="dark")
        page = browser.new_page(viewport={"width": 1180, "height": 760})
        count = [0]

        def film(ms, until=None):
            end = time.time() + ms / 1000
            while time.time() < end or (until and not page.evaluate(until)):
                page.screenshot(path=str(frames / f"{count[0]:05d}.jpg"), type="jpeg", quality=80)
                count[0] += 1
                time.sleep(0.12)
                if time.time() > end + 120:
                    break

        def shot_of(kind, js):
            maker.goto(f"{args.url}/?kind={kind}")
            maker.wait_for_selector("nav button.on")
            maker.wait_for_timeout(400)
            maker.evaluate(js)
            maker.wait_for_timeout(200)
            return phone_like(maker.screenshot(), rng)

        cases = []
        for kind in ("sudoku", "queens", "zip", "tango"):
            row = json.loads(open(f"benchmarks/arena/{kind}/test.jsonl").read().splitlines()[-41])
            show = {k: v for k, v in row.items() if k not in ("solution", "seed", "rating", "source")}
            cases.append((kind, f"{kind}-input.jpg", shot_of(kind, f"row={json.dumps(show)};draw(row,null,0)"), None))
        import chess
        crow = json.loads(open("benchmarks/arena/chess/daily.jsonl").read().splitlines()[3])
        b = chess.Board(crow["FEN"])
        b.push_uci(crow["Moves"].split()[0])
        gif = lichess_image(b.fen(), "california", "blue", "white" if b.turn else "black")
        cases.append(("chess", "chess-input.gif", gif, "w" if b.turn else "b"))
        from arena.games import checkers as ck
        _, hist, _ = ck.play({"kind": "random"}, {"kind": "material", "depth": 1}, random.Random(5), record=True)
        board, side = hist[30]
        cases.append(("checkers", "checkers-input.jpg",
                      shot_of("checkers", f"drawCheckers({json.dumps(ck.absolute(board, side).tolist())})"),
                      "dark" if side == 0 else "light"))

        page.goto(args.url)
        page.wait_for_selector("nav button")
        for kind, name, data, to_move in cases:
            (out / name).write_bytes(data)
            page.click(f"nav button[data-k='{kind}']")
            film(1200)
            if to_move:
                page.select_option("#tomove", to_move)
            with page.expect_response(lambda r: "/api/read" in r.url, timeout=120_000) as resp:
                page.set_input_files("#imgfile", str(out / name))
            body = resp.value.json()
            film(1500 if kind in ("chess", "checkers") else 5000,
                 "!document.getElementById('status').innerText.includes('…')")
            film(1500 if kind in ("chess", "checkers") else 9000)  # let the answer finish animating
            page.screenshot(path=str(out / f"{kind}-from-image.png"))
            log.append({"kind": kind, "input": name, "status": page.inner_text("#status"),
                        "read": {k: body.get(k) for k in ("n", "confidence", "corrections", "fen", "puzzle") if k in body}})
            print(kind, page.inner_text("#status").replace("\n", " | ")[:160], flush=True)
        browser.close()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5", "-i", str(frames / "%05d.jpg"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "scale=1180:-2", str(out / "image-solving-demo.mp4")], check=True)
    shutil.rmtree(frames, ignore_errors=True)
    (out / "captures.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
