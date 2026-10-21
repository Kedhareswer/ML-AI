#!/bin/sh
# Copy a finished run's inference files and its held-out test set into the repo.
# usage: package_arena.sh <kind> <runs_dir> <data_dir>
set -e
kind=$1; runs=$2; data=$3
mkdir -p artifacts/arena/$kind benchmarks/arena/$kind
cp $runs/$kind/weights.pt $runs/$kind/manifest.json $runs/$kind/metrics.jsonl artifacts/arena/$kind/
[ -f $data/$kind/test.jsonl ] && cp $data/$kind/test.jsonl benchmarks/arena/$kind/test.jsonl
sha256sum artifacts/arena/$kind/weights.pt
