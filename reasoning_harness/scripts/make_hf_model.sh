#!/bin/sh
# Assemble the Hugging Face model repo: trained weights, the code that loads them, and the model card.
# usage: scripts/make_hf_model.sh [output_dir]   (run from reasoning_harness/)
set -e
out=${1:-build/hf-model}
rm -rf "$out"
mkdir -p "$out/artifacts"
cp -r artifacts/arena "$out/artifacts/arena"
cp -r arena "$out/arena"
find "$out/arena" -name __pycache__ -prune -exec rm -rf {} +
cp pyproject.toml "$out/"
cat deploy/huggingface/MODEL_META.md ARENA_MODEL_CARD.md > "$out/README.md"
# The Hub counts a download each time config.json is fetched (no library-specific rule applies here),
# so the repo carries an index of its models under that name.
python - "$out" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
models = {}
for m in sorted((root / "artifacts/arena").glob("*/manifest.json")):
    info = json.loads(m.read_text())
    files = sorted(p.relative_to(root).as_posix() for p in m.parent.iterdir() if p.suffix in (".pt", ".npy", ".npz"))
    models[m.parent.name] = {"kind": info.get("kind"), "parameters": info.get("parameters"), "files": files,
                             "manifest": m.relative_to(root).as_posix()}
vision = sorted(p.stem for p in (root / "artifacts/arena/vision").glob("*.pt"))
cfg = {"name": "puzzle-arena", "framework": "pytorch", "code": "arena/", "models": models,
       "screenshot_readers": {"dir": "artifacts/arena/vision", "classifiers": vision},
       "space": "https://huggingface.co/spaces/Kedhareswer/puzzle-arena"}
(root / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
PY
cat > "$out/.gitattributes" <<'ATTR'
*.pt filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
ATTR
du -sh "$out"
