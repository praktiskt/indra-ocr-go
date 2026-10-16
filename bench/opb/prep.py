#!/usr/bin/env python3
"""Render Omni Parse Bench pages to PNG for the Go runner.

Pages are rasterised at the same dpi the upstream pipeline uses (200) so the Go
port sees the same pixels; image documents pass straight through as grayscale.
Writes <out>/manifest.json (one {page_id, png, suite} per page) and
<out>/imgs/<page_id>.png.

Usage (uv):
  uv run --with 'omni-parse-bench[benchmark]' --with pypdfium2 --with opencv-python-headless \
      bench/opb/prep.py --out /tmp/opb [--limit N] [--suites a,b] [--dpi 200]
"""
import argparse
import json
import os

import cv2
import pypdfium2 as pdfium

from omni_parse_bench import benchmark


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir (imgs/ + manifest.json)")
    ap.add_argument("--dataset", default=benchmark.DEFAULT_DATASET)
    ap.add_argument("--limit", type=int, default=0, help="first N pages (0 = all)")
    ap.add_argument("--suites", default="", help="comma-separated suites (default: all)")
    ap.add_argument("--dpi", type=float, default=200.0)
    a = ap.parse_args()

    imgs = os.path.join(a.out, "imgs")
    os.makedirs(imgs, exist_ok=True)
    root, pages, _ = benchmark.load(a.dataset)
    keep = set(a.suites.split(",")) - {""} if a.suites else None
    if keep:
        pages = [p for p in pages if p.suite in keep]
    if a.limit:
        pages = pages[: a.limit]

    man = []
    for i, p in enumerate(pages):
        src = benchmark.page_file(a.dataset, root, p.doc_path)
        dst = os.path.join(imgs, p.page_id + ".png")
        if not os.path.exists(dst):
            if str(src).lower().endswith(".pdf"):
                doc = pdfium.PdfDocument(str(src))
                doc[0].render(scale=a.dpi / 72.0).to_pil().save(dst)
                doc.close()
            else:
                cv2.imwrite(dst, cv2.imread(str(src), cv2.IMREAD_GRAYSCALE))
        man.append({"page_id": p.page_id, "png": dst, "suite": p.suite})
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(pages)}", flush=True)

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f)
    print(f"wrote {len(man)} pages -> {a.out}/manifest.json")


if __name__ == "__main__":
    main()
