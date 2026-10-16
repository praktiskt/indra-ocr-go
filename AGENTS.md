# AGENTS.md

Guide for AI agents (and humans) working in this repo.

## What this is

Native Go port of the [Indra OCR pipeline](https://github.com/datalab-to/indra)
(models `datalab-to/indra-ocr` v3.1): line detection, CTC recognition, reading
order and orientation on ONNX Runtime, **CPU only**. Models and the native
ONNX Runtime libraries are embedded; nothing is downloaded at runtime.

## Build & test

```sh
go build ./...          # Go 1.26+, CGO enabled
go test ./...           # unit tests (image/model tests are skipped)
INDRA_TEST_IMAGES=1 go test ./...   # + model tests on testdata/*.png
```

Integration/quality tests are env-gated (see below). CI-grade validation needs a
document corpus + the gauntlet.

## Test environment variables

| var | purpose |
|---|---|
| `INDRA_TEST_IMAGES=1` | enable model tests (`testdata/*.png`, smoke only) |
| `INDRA_CORPUS_DIR` | dir with `img/*.png` + `gt.json` -> `TestCorpusRecall` (FUNSD guard) |
| `INDRA_TEST_PAGE` | a single real page -> `TestOCRIntegrationRealPage` |
| `INDRA_REF_JSON` | Python-reference JSON for `TestDiff*` (default `/tmp/opencode/order_ref.json`) |
| `INDRA_HARD_DIR` | gauntlet root (subset dirs with `img/`+`gt.json`); used by a throwaway harness |
| `ORT_LIB_PATH` | override the bundled `libonnxruntime.so` |

## Validation — required for any perf or quality change

1. Read **[`docs/OPTIMIZATIONS_LOG.md`](docs/OPTIMIZATIONS_LOG.md)** (what was
   already accepted/rejected and why) and
   **[`.agents/skills/corpus-eval/SKILL.md`](.agents/skills/corpus-eval/SKILL.md)**
   (the corpus, guard and A/B method).
2. Run the guard: `INDRA_CORPUS_DIR=/path/to/corpus go test -run TestCorpusRecall -count=1 -v`.
3. Reject rule: recall drop > 0.01 or > 10 pages degraded > 5 pp. Neutral + faster
   = accept (note the divergent-page count in the commit message).
4. Scheduling/layout-only changes: prove **bit-identical** output
   (`TestCallAllMatchesCall`, `-race`) plus recall parity; that waives the gauntlet.
5. Numeric/threshold changes need the **gauntlet** too (receipts, multilingual,
   degraded/photo) — the FUNSD corpus alone is too clean to catch them.
6. Report **time AND peak RSS**, measured at the **deployment thread count**
   (pool = 1 thread/worker). int8 and fp32 cross over with thread count; a
   speedup measured at 4+ threads can invert the decision.

## Rules

- **ORT sessions are not safe for concurrent `Run`.** Every parallel path must
  give each concurrent `Run` its own session set.
- **Never commit throwaway harnesses.** A/B and divergence probes are
  `*_test.go` scratch files kept out of the tree (put scratch data in `/tmp`).
  `git diff` before committing must show only the intended change — no hidden
  env kill-switches.
- **Models are embedded** (`models/`). Regenerate them with `tools/`, never
  hand-edit. See [`tools/README.md`](tools/README.md).
- **Don't cut quality mechanisms** (adaptive detection, det side, deskew) without
  corpus + gauntlet numbers, or a new eval regime covering their target inputs.
- 4 `testdata` pages are smoke only. The corpus decides.

## Layout

| path | contents |
|---|---|
| `models/` | embedded ONNX models + `meta.json` (det long side, vocabs, rec height) |
| `lib/` | embedded ONNX Runtime + C++ runtime (linux/amd64) |
| `tools/` | model regeneration: `quantize_rec.py`, `calib/` (calibration crops) |
| `det.go` | detector decode (connected components) |
| `ocr_ort.go` | pipeline: detect / recognize / upright / `Call` / `CallAll` / assemble |
| `session_ort.go` | ORT sessions, embedded-lib loading, per-model threads |
| `pool.go` | worker pool (`NewPool`, `NewPoolAuto`, `NewPoolBuilder`, `CallAll`, `CallFiles`) |
| `image.go` | grayscale, resize, rotate, skew, crop; `WidthBatches`/`PadBatch` |
| `skew_py.go` | python-exact skew estimator |
| `order.go`, `layout.go` | reading order, layout blocks, markdown |
| `docs/` | `OPTIMIZATIONS_LOG.md` (perf history) |
| `.agents/skills/` | `corpus-eval` skill (guard, gauntlet, A/B method) |

## More

- [`README.md`](README.md) — usage/API.
- [`docs/OPTIMIZATIONS_LOG.md`](docs/OPTIMIZATIONS_LOG.md) — perf history.
- [`tools/README.md`](tools/README.md) — regenerating the shipped models.
- [`.agents/skills/corpus-eval/SKILL.md`](.agents/skills/corpus-eval/SKILL.md) — evaluation method.
