---
name: corpus-eval
description: >
  Broad quality evaluation for indra-ocr-go performance work. Scores word recall
  over a 199-page FUNSD corpus instead of the 4 testdata pages, so perf changes
  can't silently overfit a narrow sample. Use when touching detection sides,
  adaptive passes, skew/orient preprocessing, batching, or concurrency/pipelining,
  or asking "is this safe on broad data". Also covers the A/B procedure for numeric changes, the
  settled-decisions log, and the 150-page brutal-print gauntlet (receipts,
  multilingual forms, synthetic degrade/photo).
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

## Settled decisions (evidence, don't re-litigate without new data)

- Det side 1024 default. 768 shifts output on 178/199 pages, degrades 10
  pages 5-15pp (Latin forms gain avg +1.7pp, CJK/small-print loses).
- Skew proxy 600px + 3-level search. A/B identical to 0.01deg on 11 tilts;
  neutral recall on corpus, ~30% faster upright.
- Adaptive second det pass stays. Neutral on corpus (11/199 pages differ),
  decisive in tiny-print regime: 48 -> 137 lines on padded-canvas probe
  (locked by `TestAdaptiveSmallText`).
- Orient input fixed at 768x768 by model shape; inference ~5.6ms. No shrink.
- Rec padding waste 4-11%: batcher already tight, leave it.
- Parallel rec sessions: rejected. Rec saturates cores at 8 threads;
  16-thread single-session run collapses 248ms -> 681ms. More sessions =
  more contention for uncertain gain.
- Rec int8 static quant: ACCEPTED (supersedes the earlier rejection in git
  history). The old "1.25x / -0.78pp" was measured at 4+ intra-op threads,
  where int8 has no edge (fp32 scales with threads, VNNI does not). At the
  pool's 1 thread/worker int8 rec is ~2.4x (t2 1.8x, t4 1.3x). Accuracy was
  recoverable too, via three choices: quantize the RAW graph (quant_pre_process
  decomposes HardSwish and quantizes more tensors -> worse on both axes), use
  ASYMMETRIC activations (ActivationSymmetric=False, ~+0.6pp vs symmetric), and
  keep the stem + first blocks FP32. Shipped config (`tools/quantize_rec.py`,
  byte-reproducible): QDQ per-channel, QUInt8/QInt8 MinMax, asym, stem+blocks
  0-1 FP32. FUNSD 0.9134 vs 0.9150 (-0.16pp), 2/199 pages >5pp worse, 0 >10pp;
  gauntlet within 0.5pp (better on cord/indic, -0.5pp degraded/photo); pool
  17.8 -> 22.3 pages/s (+25%), peak RSS -13%. Keeping more leading blocks FP32
  trades speed for accuracy roughly monotonically (stem+blocks0-3: 21.4 p/s,
  -0.05pp; stem only: 23.2 p/s, -0.41pp but -2/-2.5pp on degraded/photo).
  Entropy/Percentile cal still fails on the dynamic-shape batches. QOperator,
  dynamic-quant all worse. FP8: dead on CPU - no ORT CPU fp8 kernels and no fp8
  hardware on Zen4 (avx512_vnni only).
- Rec AdaRound / activation MSE-clip: tried, no meaningful gain; don't re-run
  the standard forms. Layer-wise AdaRound (ONNX + numpy, layer output Gram:
  L(h)=sum_c s_c^2 (h_c-f_c)^T G_c (h_c-f_c), sequential binary coordinate
  descent from round-to-nearest) cut per-layer output MSE 63% (14% of weights
  re-rounded) but moved FUNSD word recall only +0.04pp (b1) / +0.09pp (full
  int8); line-exact and char-edit distance flat. MSE-optimal activation ranges
  beat ORT MinMax on 0/127 tensors (ranges already tight). So the residual int8
  loss is intrinsic to 8-bit activation quant, not weight rounding - the FP32
  stem/blocks0-1 are what buy the last ~0.2pp. Untried lever, if ever needed:
  quantized-input reconstruction (minimize ||conv(W_hat, Xq) - conv(W, X)||^2
  with the model's own activation quantiser), which can see activation error.
- Pipeline overlap: accepted. `OCR.CallAll`/`Pool.CallAll` run per-page stages
  (orient+det / rec / assemble) as a 3-stage goroutine pipeline, one page per
  stage, sessions partitioned by stage. Output bit-identical to per-page `Call`
  (DeepEqual, threads=1; `-race` clean), corpus neutral 0.8616. Single handle
  +1.36-1.43x pages/s (1t 5.6->7.6, 4t 9.2->13.2); pool reaches machine peak
  (~18 p/s) with 4-6 workers vs 8-12 (fewer session sets). This is *stage*
  overlap, NOT parallel rec sessions (see above) - only one page is in rec
  at a time.
- Pool sizing: default is `NumCPU/2` workers x 1 thread (`AutoWorkers`,
  `NewPoolAuto`/`NewPoolBuilder`; explicit `NewPool` still available). Worker
  count dominates intra-op threads: pool configs with threads=2 lose
  (6w x 2t 15.4 vs 6w x 1t 17.4 pages/s on 16 cores). Throughput plateaus from
  ~6 to ~16 workers (~17.5 p/s), so half the cores is the throughput knee with
  headroom for the pipeline's small stages. RAM scales ~linearly with workers:
  ~120MB/handle on small pages, ~260MB on large, plus a ~380MB runtime/first-
  handle floor - hence `MemBudget` (caps workers at `PerHandleBytes`=256MB
  each; handles only, not in-flight pages). Measure memory too, not just time.
- Incremental warp walks: `sharpRows`/`rotateRows`/`resizeLinearRows`
  advance coords by addition instead of recomputing per pixel. Tilt sweep
  13/13 identical to 1e-4deg; rec texts bit-identical on 4 testdata pages;
  corpus neutral (0.8616). skew -16%, rotate -21%, crops -10%. Safe pattern
  for future warp-math work (drift ~1e-11px, needs same A/B proof each time).
- Order `tab` head pruned: allocated+computed, never read (tables
  render geometrically). a/cls/same bit-identical at N=43/138/400. No win at
  N=138, ~4% order-stage at N=400 + drops up to ~2MB dead alloc. Kept as hygiene.
- Gauntlet scope: Latin print + receipts + photo warp covered;
  CJK/handwriting/scene-text are known voids, quantified not fixed. CJK pages
  score 0.4-0.6 on digits alone.
- Alphabet probe (synthetic, single-font, clean): ru 0.97, el 0.97, hi 0.95,
  bn 0.98 — the Cyrillic/Greek/Devanagari/Bengali vocab classes read fine on
  clean print. Real-scan robustness for these scripts untested; Arabic/Hebrew/
  Thai have no vocab classes at all.

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
