"""Run LlamaIndex ExtractBench with the two-stage docparse / pypdf -> local LLM pipelines registered.

    DOCPARSE_LLM_URL=http://127.0.0.1:8088 python scripts/extractbench_docparse.py run docparse_extract_local --test
    DOCPARSE_LLM_URL=http://127.0.0.1:8088 python scripts/extractbench_docparse.py run pypdf_extract_local --test
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import docparse.extractbench  # noqa: E402,F401  (registers the provider and pipelines)
from extract_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
