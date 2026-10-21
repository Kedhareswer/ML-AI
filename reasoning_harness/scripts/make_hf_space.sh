#!/bin/sh
# Assemble a Hugging Face Space folder with only what the app needs at runtime.
# usage: scripts/make_hf_space.sh [output_dir]   (run from reasoning_harness/)
set -e
out=${1:-build/hf-space}
rm -rf "$out"
mkdir -p "$out/benchmarks/arena"
cp deploy/huggingface/app.py deploy/huggingface/requirements.txt "$out/"
# Space page = Space metadata + the model card
cat deploy/huggingface/README.md ARENA_MODEL_CARD.md > "$out/README.md"
cp -r arena "$out/arena"
find "$out/arena" -name __pycache__ -prune -exec rm -rf {} +
mkdir -p "$out/artifacts"
cp -r artifacts/arena "$out/artifacts/arena"
for d in sudoku mini_sudoku tango queens zip patches wend chess pinpoint; do
  mkdir -p "$out/benchmarks/arena/$d"
  cp benchmarks/arena/$d/*.jsonl "$out/benchmarks/arena/$d/"
done
cat > "$out/.gitattributes" <<'ATTR'
*.pt filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
ATTR
du -sh "$out"
