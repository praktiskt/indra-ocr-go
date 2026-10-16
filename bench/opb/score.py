#!/usr/bin/env python3
"""Score a Go-port Omni Parse Bench JSONL with the official scorer.

Reads the JSONL from `bench/opb/main.go` plus the PNGs (for page dimensions),
builds the `{html, blocks}` outputs the benchmark expects, and prints the
`headline` / `text` / `tables` / `layout` summary. Writes <name>_summary.json.

Usage (uv):
  uv run --with 'omni-parse-bench[benchmark]' --with opencv-python-headless \
      bench/opb/score.py --jsonl /tmp/opb/go.jsonl --imgs /tmp/opb/imgs --name go
"""
import argparse
import json
import os

import cv2

from omni_parse_bench import benchmark, markdown
from omni_parse_bench import score as opb_score, summarize

# Go layout classes -> the benchmark's block labels.
LABELS = {
    "Text": "text", "Section-Header": "heading", "Page-Header": "page_furniture",
    "Page-Footer": "page_furniture", "List-Group": "list", "Table": "table",
    "Caption": "caption", "Footnote": "footnote", "Table-Of-Contents": "text", "Form": "form",
}


def clip(v):
    return min(1.0, max(0.0, v))


def outputs_of(row, w, h):
    blocks = []
    for b in row.get("blocks") or []:
        x0, y0, x1, y1 = b["bbox"]
        blocks.append({
            "label": LABELS.get(b["label"], "text"),
            "bbox": [clip(x0 / w), clip(y0 / h), clip(x1 / w), clip(y1 / h)],
            "text": b["text"],
        })
    return {"html": markdown.to_html(row.get("markdown") or ""), "blocks": blocks}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--imgs", required=True, help="dir with <page_id>.png from prep.py")
    ap.add_argument("--name", default="go")
    ap.add_argument("--dataset", default=benchmark.DEFAULT_DATASET)
    a = ap.parse_args()

    _, pages, by_page = benchmark.load(a.dataset)
    by_id = {p.page_id: p for p in pages}

    rows = [json.loads(l) for l in open(a.jsonl)]
    verdicts, scored, ms = [], [], []
    for row in rows:
        pid = row["page_id"]
        if pid not in by_id:
            continue
        img = cv2.imread(os.path.join(a.imgs, pid + ".png"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        verdicts += opb_score(outputs_of(row, w, h), by_page[pid])
        scored.append(by_id[pid])
        ms.append(row["ms"])

    s = summarize(verdicts, [t for p in scored for t in by_page[p.page_id]], scored)
    with open(f"{a.name}_summary.json", "w") as f:
        json.dump(s, f, indent=1, default=str)
    ms.sort()
    print(json.dumps({
        "runner": a.name,
        "pages": len(rows),
        "headline": s["headline"],
        "ms_median": ms[len(ms) // 2] if ms else None,
        "per_suite": {k: round(v, 4) for k, v in s.get("per_suite", {}).items()},
    }, indent=1))


if __name__ == "__main__":
    main()
