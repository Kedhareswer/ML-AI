"""Run LlamaIndex ParseBench with the docparse pipeline registered.

    python scripts/parsebench_docparse.py download [--test]
    python scripts/parsebench_docparse.py run docparse [--test] [--group table]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import docparse.parsebench  # noqa: E402,F401  (registers the provider, pipeline and layout adapter)
from parse_bench.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
