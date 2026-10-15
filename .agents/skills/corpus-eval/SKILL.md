---
name: corpus-eval
description: >
  Broad quality evaluation for indra-ocr-go performance work. Scores word recall
  over a 199-page FUNSD corpus instead of the 4 testdata pages, so perf changes
  can't silently overfit a narrow sample. Use when touching detection sides,
  adaptive passes, skew/orient preprocessing, batching, or asking "is this safe
  on broad data". Also covers the A/B procedure for numeric changes, the
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

Reference scores (threads=2, main @ `2cd25cb`):

| subset | recall | avg ms |
|---|---|---|
| xfund | 0.9411 | 833 |
| sroie | 0.8309 | 707 |
| funsd-degraded | 0.7848 | 671 |
| funsd-photo | 0.7579 | 655 |
| cord | 0.6810 | 272 |

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

## Rules

- No cutting quality mechanisms (adaptive, det side, deskew) without
  corpus numbers or a new eval regime that covers their target inputs.
- Numeric changes (resize, search schedules, thresholds) need A/B proof of
  identical-or-neutral output, not just "tests pass".
- 4 testdata pages are smoke only. Corpus decides.
- Orient/threshold changes must ALSO pass the cord photo subset (known
  `rot=180` false positive on dark-background receipts) plus the FUNSD guard.
- Downstream users of the gauntlet: GT-sparse sroie rows and joined-token
  cord GT make worst-lists noisy; always eyeball-dump worst pages before
  calling a number a pipeline bug (see caveats above).
