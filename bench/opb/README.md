# Omni Parse Bench (OPB)

Runs the Go port against the official upstream benchmark
[`datalab-to/omni_parse_bench`](https://github.com/datalab-to/omni_parse_bench):
2937 pages, 22 suites (arxiv math, tables, forms, multi-column, multilingual,
handwriting, rotation/skew, old scans, tiny text...). Each page's outputs
(HTML rendered from the markdown + labeled blocks) are scored against per-page
tests; the **headline** is the mean of the `text`, `tables` and `layout`
families. The distribution is the one the models were tuned for, so it catches
page types the FUNSD corpus and brutal-print gauntlet do not.

Python runs through `uv` — nothing to install by hand. The dataset
(`hf://datasets/datalab-to/omni_parse_bench`) downloads on first use.

## Go port

```sh
bench/opb/run.sh /tmp/opb                 # full 2937 pages
bench/opb/run.sh /tmp/opb --limit 300     # first 300 (prep.py args pass through)
```

or the three steps separately:

```sh
uv run --with 'omni-parse-bench[benchmark]' --with pypdfium2 --with opencv-python-headless \
    bench/opb/prep.py --out /tmp/opb                 # render PNGs @ dpi 200 + manifest
go run ./bench/opb -manifest /tmp/opb/manifest.json -out /tmp/opb/go.jsonl -workers 8
uv run --with 'omni-parse-bench[benchmark]' --with opencv-python-headless \
    bench/opb/score.py --jsonl /tmp/opb/go.jsonl --imgs /tmp/opb/imgs --name go
```

`prep.py` rasterises PDFs at **dpi 200** (the dpi the upstream pipeline uses) so
the port sees the same pixels; image documents pass through as grayscale.
`main.go` runs **one page at a time** (matching the benchmark's methodology)
through the pool and writes `{page_id, markdown, blocks, ms}` JSONL. `score.py`
builds the `{html, blocks}` outputs (Go block classes map 1:1 onto the
benchmark's labels) and prints the summary; the full summary lands in
`<name>_summary.json`.

## Upstream baseline

Use `bench/opb_indra.py` from `datalab-to/indra`:

```sh
uv run --with indra-ocr --with 'omni-parse-bench[benchmark]' --with pypdfium2 \
    python bench/opb_indra.py --out runs/py --models <models> --mode ocr --procs 8
```

**Use `--mode ocr`.** With `--mode auto` upstream reads the embedded PDF text
layer (`page.source` = `text`/`hybrid`); the Go port is OCR-only, so `auto` is
not a like-for-like OCR comparison — it is a missing feature, not a regression.

## Reference results (dev box, full 2937 pages)

| run | headline | text | tables | layout | median ms |
|---|---|---|---|---|---|
| **Go port (OCR only)** | **0.5687** | 0.311 | 0.526 | 0.870 | 234 |
| upstream `--mode ocr` | 0.5714 | 0.326 | 0.512 | 0.876 | 591 |
| upstream `auto` | 0.6155 | 0.405 | 0.563 | 0.879 | 504 |

Go is at parity with upstream on OCR (-0.26pp headline) at ~2.5x lower latency,
and ahead on tables (+1.0pp). The only material per-tag gap is `header_footer`
(Go 0.146 vs 0.606, 226 tests): Go drops page furniture
(`keepFurniture=false`) while upstream's `opb_indra` passes `furniture=True`.

Upstream's published numbers (x86, 4 vCPU, one page at a time) are `auto` 62.1
and OCR-only 57.5 at 228/303 ms — the runs above reproduce them within ~0.5.

## Caveats

- **Text layer:** upstream `auto` uses digital PDFs' embedded text; the Go port
  has no text-layer path, so on digital-born pages upstream scores higher.
  Compare with `--mode ocr`.
- **Furniture:** Go markdown drops headers/footers; the `header_footer` tag is
  the only place this bites (and it is the whole Go-vs-upstream headline gap).
- **Math:** Go has no LaTeX path; math-heavy suites (`olmo_arxiv_math`) score
  low for both engines.
- **Render parity:** page PNGs must be dpi 200 to match indra's rasterisation.
