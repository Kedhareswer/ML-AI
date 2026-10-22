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
cat > "$out/.gitattributes" <<'ATTR'
*.pt filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
ATTR
du -sh "$out"
