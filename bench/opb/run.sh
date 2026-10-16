#!/bin/sh
# One Omni Parse Bench run of the Go port: render -> OCR -> score.
#
#   bench/opb/run.sh <outdir> [prep.py args...]
#   bench/opb/run.sh /tmp/opb --limit 300
#
# For the upstream baseline, use bench/opb_indra.py from datalab-to/indra with
# --mode ocr (see README.md).
set -e
OUT="${1:-/tmp/opb}"
shift || true
mkdir -p "$OUT"

uv run --with 'omni-parse-bench[benchmark]' --with pypdfium2 --with opencv-python-headless \
    bench/opb/prep.py --out "$OUT" "$@"

go run ./bench/opb -manifest "$OUT/manifest.json" -out "$OUT/go.jsonl" -workers 8

uv run --with 'omni-parse-bench[benchmark]' --with opencv-python-headless \
    bench/opb/score.py --jsonl "$OUT/go.jsonl" --imgs "$OUT/imgs" --name go
