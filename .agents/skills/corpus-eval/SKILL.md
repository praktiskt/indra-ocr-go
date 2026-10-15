---
name: corpus-eval
description: >
  Broad quality evaluation for indra-ocr-go performance work. Scores word recall
  over a 199-page FUNSD corpus instead of the 4 testdata pages, so perf changes
  can't silently overfit a narrow sample. Use when touching detection sides,
  adaptive passes, skew/orient preprocessing, batching, or asking "is this safe
  on broad data". Also covers the A/B procedure for numeric changes and the
  settled-decisions log.
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
// write /tmp/.../out_<cfg>.json {basename: {lines, ms}}
```

Score with bag-of-words recall: lowercase, `[A-Za-z0-9]+` tokens,
GT filtered to full-match words, hit = GT word present in predicted set.

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

## Rules

- No cutting quality mechanisms (adaptive, det side, deskew) without
  corpus numbers or a new eval regime that covers their target inputs.
- Numeric changes (resize, search schedules, thresholds) need A/B proof of
  identical-or-neutral output, not just "tests pass".
- 4 testdata pages are smoke only. Corpus decides.
