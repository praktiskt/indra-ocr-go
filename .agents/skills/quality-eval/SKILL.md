---
name: quality-eval
description: >
  Quality evaluation for indra-ocr-go work. Scores word recall over a 199-page
  FUNSD corpus instead of the 4 testdata pages, so changes can't silently
  overfit a narrow sample. Use when touching detection sides, adaptive passes,
  skew/orient preprocessing, batching, concurrency/pipelining, quantized models,
  or asking "is this safe on broad data". Also covers the A/B procedure, the
  reading-order metric, the settled-decisions log, the 150-page brutal-print
  gauntlet (receipts, multilingual forms, synthetic degrade/photo), and the
  official Omni Parse Bench (2937 pages, 22 suites).
---

## Corpus

199 diverse pages (forms, faxes, ads) + word-level ground truth, ~15MB.

Fetch once (needs `uv`):

```sh
mkdir -p corpus && cd corpus
for f in test train; do
  curl -sL -o $f.parquet "https://huggingface.co/datasets/nielsr/funsd/resolve/main/data/$f-00000-of-00001.parquet"
done
uv run --with pyarrow python - <<'EOF'
import pyarrow.parquet as pq, json, os
os.makedirs("img", exist_ok=True)
gt = {}
for split, f in [("test", "test.parquet"), ("train", "train.parquet")]:
    for row in pq.read_table(f).to_pylist():
        name = f"{split}_{row['id']}.png"
        open(f"img/{name}", "wb").write(row["image"]["bytes"])
        gt[name.removesuffix(".png")] = row["words"]
json.dump(gt, open("gt.json", "w"))
print("pages:", len(gt))
EOF
```

## Guard test (committed)

`corpus_test.go:TestCorpusRecall`, gated on `INDRA_CORPUS_DIR`:

```sh
INDRA_CORPUS_DIR=/path/to/corpus go test -run TestCorpusRecall -count=1 -v
```

Pass bar: word recall >= 0.80 (reference ~0.86 at v0.8.0, threads=2).
Below bar = real regression, not refactor noise. ~16s runtime.

## Reading order (the guard is blind to it)

Word recall is bag-of-words: it cannot see line *sequence*. Any change to
`order.onnx` (quantization, node pruning) needs a reading-order check too.
Score the ordered predicted lines against FUNSD's ordered GT words with Kendall
tau (matched tokens, in predicted order):

1. Run OCR over a dir with `img/` + `gt.json` and dump per-page ordered line
   texts to JSON (throwaway `_test.go`, never commit).
2. For each page map predicted tokens to their GT word index and compute
   Kendall tau of that sequence. Aggregate the mean over pages.

The corpus is deterministic (raw-vs-raw divergence is 0), so this is a
differential metric: compare candidate vs shipped. The shipped order.onnx
(asserted in the log) has mean tau 0.5057 on FUNSD and 0.8889 cord / 0.3650
xfund. Neutral = |delta| < ~0.01; quantify how many pages drop >2pp.

## A/B procedure for candidate changes

1. Baseline: run guard, save recall.
2. Candidate behind temp env knob or const flip. Run guard again.
3. Compare: recall delta + count of pages with changed line output.
   - Recall drop >0.01 or >10 pages degraded >5pp: reject.
   - Neutral recall + faster: accept, note divergent-page count in commit msg.
4. Revert all temp knobs before commit. `git diff` must show only the
   intended change. No permanent hidden env kill-switches.

Changes that only reorder/overlap work with no numeric effect (scheduling,
concurrency, session partitioning) skip steps 1-3's divergence harness: prove
output is bit-identical (equivalence test, e.g. `TestCallAllMatchesCall`, run
`-race`) plus recall parity on the guard. Bit-identical output also waives the
gauntlet and the cord-photo orient check.

Quick divergence harness pattern (throwaway `_test.go`, never commit):

```go
// one worker per OCR, CORPUS_CFG/CORPUS_DET env select config,
// write scratch/out_<cfg>.json {basename: {lines, ms}}
```

Score with bag-of-words recall: lowercase, `[A-Za-z0-9]+` tokens,
GT filtered to full-match words, hit = GT word present in predicted set.

## Brutal-print gauntlet (T1, 150 pages)

Built because FUNSD (0.86) is too clean to show receipt/photo/CJK failure.
Lives OUTSIDE the repo in a local scratch dir, e.g.
`$CORPUS_HARD/<subset>/img/*.png + gt.json` (same key convention as corpus:
basename without ext). Fetch/extract scripts live alongside the scratch data
(HF sets + synthetic generators). Downloaded parquets are deleted after
extract (re-downloadable); the scripts are the record.

| subset | source | pick | n |
|---|---|---|---|
| sroie | `arvindrajan92/sroie_document_understanding` train (652 rows, image+ocr) | 30 blurriest by Laplacian variance | 30 |
| cord | `mychen76/receipt_cord_ocr_v2` test (100 rows) | every ~3rd, spread | 30 |
| xfund | `nnul/xfund-multilingual` val (350 rows, 7x50) | 4/lang + zh,ja 5th (ids `<lang>_val_<i>`) | 30 |
| funsd-degraded | local FUNSD 5 bases x 6 corrupts (down55, jpeg18, noise16, fade, blur14, fold) | deterministic seeds | 30 |
| funsd-photo | local FUNSD 5 bases x 6 photo variants (warplight, rotshadow, glare, warpheavy, tiltblur, combo) | deterministic seeds | 30 |

GT words: sroie `ocr[].text`, cord `json.loads(raw_data)["ocr_words"]`
(literal_eval), xfund `words`, synth copied from FUNSD gt.

Scorer pattern (throwaway `hard_probe_test.go`, `HARD_DIR` env, never commit):
8 workers x `NewOCR(2)`, per page {recall, lines, empty, ms, rotation, skew},
log aggregate + worst-5, dump per-page JSON beside the scratch data.
Single-page dump pattern (throwaway, `HARD_DUMP=sub/page`): print
`lines/rot/skew + Markdown` for eyeballing failures.

Reference scores (threads=2):

| subset | recall | avg ms |
|---|---|---|
| xfund | 0.9411 | 833 |
| sroie | 0.8309 | 707 |
| funsd-degraded | 0.7848 | 671 |
| funsd-photo | 0.7579 | 655 |
| cord | 0.7055 | 272 |

Compare current runs against recall; `avg ms` is a rough guide (machine and
thermal state move it ~20%).

Known caveats (don't "fix" these as pipeline bugs without checking):
- sroie worst-list dominated by GT-sparse rows (denominators <=17);
  the enriched `ocr` field is entity-lean on some rows. OCR itself reads
  full receipts incl tables. Aggregate still meaningful (micro-average).
- cord GT `ocr_words` holds joined tokens (`Qty2.0060.000`) that the
  alnum filter drops -> tiny denominators, harsh metric. Consistent, keep.
- DIBCO deliberately excluded: ships binary masks, NO transcriptions.
  Replaced with funsd-photo. Don't re-add without a transcription source.

Failure classes mapped (eyeballed dumps):
1. CJK void + Latin hallucination (zh/ja ~0.4-0.6, digits carry them).
   Detection + digits right, glyphs confabulated. No CJK routing exists.
2. Orient false positive on photo receipts: upright photo + `rot=180`
   (>0.95 conf) = whole page garbage. Dark top background suspected bias.
   Status: handled — true rotations score >=0.998 vs FP 0.954, so
   `OrientMin` sits at 0.98 (cord +2.5pp, FUNSD neutral).
3. Strong perspective -> det collapse (~29 vs ~100 lines) + skew misread.
   No perspective path in pipeline.
4. Blur/noise/downscale -> graceful rec degrade (0.46-0.63).
5. `MinConf` wipeouts on faint print (e.g. 25/36 lines empty).

## Omni Parse Bench (official, 2937 pages, 22 suites)

The upstream `datalab-to/omni_parse_bench`: arxiv math, tables, forms,
multi-column, multilingual, handwriting, rotation/skew, old scans, tiny text,
receipts... It scores each page's outputs (HTML rendered from the markdown +
labeled blocks) against per-page tests; `headline` = fraction passing, split
into `text` and `layout` families. This is the distribution the models were
tuned for, so it catches page types neither our FUNSD corpus nor the gauntlet
carry (math, tables, rotation). Use it as the broad, official check.

Install (needs HF access; the dataset loads on first use, ~3k files):

```sh
uv pip install --python <venv>/bin/python "omni-parse-bench[benchmark]"
# dataset: hf://datasets/datalab-to/omni_parse_bench  (benchmark.DEFAULT_DATASET)
```

Two runners, one scorer:

1. **Upstream baseline** — `bench/opb_indra.py` from `datalab-to/indra`:
   ```sh
   python bench/opb_indra.py --out runs/py --models <models> --mode ocr --procs 8
   ```
   **Use `--mode ocr`.** With `--mode auto` upstream reads the embedded PDF
   text layer (`page.source` `text`/`hybrid`); the Go port is OCR-only (no
   text-layer path), so `auto` is a capability gap, not a like-for-like OCR
   comparison.
2. **Go port** (throwaway harness, scripts in scratch not the repo):
   - prep: render each page to PNG at **dpi 200** (matching indra) with
     pypdfium2; `.png` docs pass through; write a `manifest.json`.
   - OCR: a `tools/opb` main reads the manifest, runs the pool, and writes
     `{page_id, markdown, blocks:[{label,bbox,text}]}` JSONL (bbox in page px).
   - score: build `{"html": omni_parse_bench.markdown.to_html(markdown),
     "blocks": [{"label": LABELS[label], "bbox": normalised 0-1, "text"}]}` and
     call `omni_parse_bench.score(outputs, tests)` then
     `summarize(verdicts, tests, pages)`. Go `BlockClasses` map 1:1 onto OPB
     labels (`Table-Of-Contents` -> `text`).

Full-corpus result (dev box, all 2937 pages, 8 workers x 1 thread):

| run | headline | text | tables | layout | median ms |
|---|---|---|---|---|---|
| Go port (OCR only) | **0.5687** | 0.311 | 0.526 | 0.870 | 234 |
| upstream `--mode ocr` | 0.5714 | 0.326 | 0.512 | 0.876 | 591 |
| upstream `auto` | 0.6155 | 0.405 | 0.563 | 0.879 | 504 |

(Upstream's published numbers, x86 4 vCPU one-at-a-time: `auto` 62.1, OCR-only
57.5, median 228/303 ms — my runs reproduce them within ~0.5.)

Go is at **parity with upstream on OCR** (-0.26pp headline) at **~2.5x lower
latency**, and *ahead* on tables (+1.0pp). The only material per-tag gap is
`header_footer` (Go 0.146 vs upstream 0.606, 226 tests) — Go drops page
furniture; every other tag is within ~0.03. The `auto` row is higher only
because it reads PDF text layers (which Go lacks); it is not the like-for-like
target.

Caveats:
- Go markdown drops page furniture (`keepFurniture=false`) while upstream's
  `opb_indra` passes `furniture=True`; expect a small `page_furniture` delta.
- Go has no math/LaTeX path; math-heavy suites (arxiv math) score low for both.
- The PNG render must be dpi 200 to match indra's rasterisation.

## Settled decisions

Moved to [`docs/OPTIMIZATIONS_LOG.md`](../../../docs/OPTIMIZATIONS_LOG.md):
every accepted and rejected optimization with its evidence - int8 rec, the
Conv1d->Conv2d NCHWc promotion, skew stride-4, det resolution, the alternate CPU
backends (DNNL/OpenVINO/XNNPACK), AdaRound/activation-clip, batching, SE removal,
pool sizing, and the throughput ceiling + honest-benchmarking method.

**Read that log before re-attempting anything in the perf space**; it records
why each was accepted or rejected so settled questions aren't re-litigated.

## Rules

- No cutting quality mechanisms (adaptive, det side, deskew) without
  corpus numbers or a new eval regime that covers their target inputs.
- Numeric changes (resize, search schedules, thresholds) need A/B proof of
  identical-or-neutral output, not just "tests pass".
- Perf work reports time AND peak RSS. Worker count drives memory ~linearly
  (~120-260MB/handle); a time-only sweep hides multi-GB regressions.
- Measure speedups at the DEPLOYMENT thread count. int8 and fp32 cross:
  int8 rec is 2.4x at 1 thread (pool) but 1.3x at 4 and ~1.0x at 8, because
  fp32 scales with threads while VNNI saturates at one. A speedup measured at
  4+ threads can invert the decision (this is how int8 got wrongly rejected).
- 4 testdata pages are smoke only. Corpus decides.
- Concurrency: ORT sessions are NOT safe for concurrent `Run`. Every parallel
  path must give each concurrent Run its own session set, and prove
  bit-identical output (`-race` test) + recall parity, not just "tests pass".
- Orient/threshold changes must ALSO pass the cord photo subset (known
  `rot=180` false positive on dark-background receipts) plus the FUNSD guard.
- Downstream users of the gauntlet: GT-sparse sroie rows and joined-token
  cord GT make worst-lists noisy; always eyeball-dump worst pages before
  calling a number a pipeline bug (see caveats above).
