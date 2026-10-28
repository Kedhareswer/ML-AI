#!/bin/sh
# Assemble the docparse Hugging Face model repo and Space. The card on both is DOCPARSE.md.
# usage: scripts/make_hf_docparse.sh   (run from reasoning_harness/) -> build/hf-docparse-model, build/hf-docparse-space
set -e
D=deploy/huggingface-docparse
card() {  # Hub metadata + DOCPARSE.md
  cat "$1" DOCPARSE.md
}
copy_code() {
  mkdir -p "$1/artifacts"
  cp -r docparse "$1/docparse"
  cp -r artifacts/docparse "$1/artifacts/docparse"
  mkdir -p "$1/evidence" && cp -r evidence/docparse "$1/evidence/docparse"  # charts on the card
  find "$1/docparse" -name __pycache__ -prune -exec rm -rf {} +
}

m=build/hf-docparse-model
rm -rf "$m"
copy_code "$m"
mkdir -p "$m/benchmarks" "$m/scripts" "$m/tests"
cp -r benchmarks/docparse benchmarks/parsebench benchmarks/extractbench "$m/benchmarks/"
cp scripts/build_doc_corpus.py scripts/parsebench_docparse.py scripts/parsebench_summary.py scripts/extractbench_docparse.py scripts/plot_docparse.py scripts/docparse_block_types.py "$m/scripts/"
cp tests/test_docparse.py "$m/tests/"
cp pyproject.toml "$m/"
card "$D/MODEL_META.md" > "$m/README.md"
# The Hub counts a download each time config.json is fetched, so the repo carries a small index under that name.
python - "$m" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
man = json.loads((root / "artifacts/docparse/manifest.json").read_text())
cfg = {"name": "docparse", "framework": "pytorch", "code": "docparse/", "weights": "artifacts/docparse/ocr.pt",
       "architecture": man["model"], "parameters": man["parameters"], "input": man["input"], "charset": man["charset"],
       "space": "https://huggingface.co/spaces/Kedhareswer/docparse"}
(root / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
PY
printf '*.pt filter=lfs diff=lfs merge=lfs -text\n' > "$m/.gitattributes"

s=build/hf-docparse-space
rm -rf "$s"
copy_code "$s"
cp "$D/app.py" "$D/requirements.txt" "$D/packages.txt" "$s/"
card "$D/SPACE_META.md" > "$s/README.md"
printf '*.pt filter=lfs diff=lfs merge=lfs -text\n' > "$s/.gitattributes"
du -sh "$m" "$s"
