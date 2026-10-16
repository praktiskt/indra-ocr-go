# AGENTS.md

Guide for AI agents (and humans) working in this repo.

## What this is

Native Go port of the [Indra OCR pipeline](https://github.com/datalab-to/indra)
(models `datalab-to/indra-ocr` v3.1): line detection, CTC recognition, reading
order and orientation on ONNX Runtime, **CPU only**. Models and the native
ONNX Runtime libraries are embedded; nothing is downloaded at runtime.

## Keep these documents current

`AGENTS.md` (this file), [`docs/OPTIMIZATIONS_LOG.md`](docs/OPTIMIZATIONS_LOG.md)
and [`.agents/skills/quality-eval/SKILL.md`](.agents/skills/quality-eval/SKILL.md)
are living documents. **Update them in the same commit as the change, never
"later":**

- Accepted or rejected an optimization -> add/adjust its entry in
  `docs/OPTIMIZATIONS_LOG.md` (status, config, numbers, why). This is what stops
  settled questions from being re-litigated.
- Changed files, env vars, commands, conventions or layout -> update this
  `AGENTS.md`.
- Changed the corpus, guard, gauntlet or A/B procedure -> update the skill.
- A doc that contradicts the code is wrong: the code wins, fix the doc.

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
   **[`.agents/skills/quality-eval/SKILL.md`](.agents/skills/quality-eval/SKILL.md)**
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
7. The official **Omni Parse Bench** ([`bench/opb/`](bench/opb/)) is the broad
   periodic check. It is **heavy** (~30 min+ end to end), so run it sparingly —
   at releases or after a model / pipeline change — not on every commit.

## Rules

- **ORT sessions are not safe for concurrent `Run`.** Every parallel path must
  give each concurrent `Run` its own session set.
- **Keep `SessionOptions` in sync with upstream `indra_ocr/pipeline.py`**
  (CPU memory arena, inter-op threads, graph-opt level, per-model threads).
  Runtime *config* drifts silently and moves RSS/throughput without touching a
  model or a test — that is how the arena got re-enabled. Changed a setting?
  A/B it and log it, like any other perf change.
- **Never commit throwaway harnesses.** A/B and divergence probes are
  `*_test.go` scratch files kept out of the tree (put scratch data in `/tmp`).
  `git diff` before committing must show only the intended change — no hidden
  env kill-switches. The one committed harness is [`bench/opb/`](bench/opb/) (the
  official Omni Parse Bench); everything ad hoc stays throwaway.
- **Models are embedded** (`models/`). Regenerate them with `tools/`, never
  hand-edit. See [`tools/README.md`](tools/README.md).
- **Don't cut quality mechanisms** (adaptive detection, det side, deskew) without
  corpus + gauntlet numbers, or a new eval regime covering their target inputs.
- 4 `testdata` pages are smoke only. The corpus decides.

## Layout

| path | contents |
|---|---|
| `models/` | embedded ONNX models + `meta.json` (det long side, vocabs, rec height) |
| `lib/` | embedded native libs (linux/amd64): ONNX Runtime, PDFium (PDF render + text layer) + C++ runtime |
| `tools/` | model regeneration: `quantize_rec.py`, `quantize_order.py`, `calib/` (calibration crops) |
| `bench/opb/` | committed Omni Parse Bench harness (Go runner + `uv`-run prep/score); see [`bench/opb/README.md`](bench/opb/README.md) |
| `det.go` | detector decode (connected components) |
| `ocr_ort.go` | pipeline: detect / recognize / upright / `Call` / `CallAll` / assemble |
| `session_ort.go` | ORT sessions, embedded-lib loading, per-model threads |
| `pool.go` | worker pool (`NewPool`, `NewPoolAuto`, `NewPoolBuilder`, `CallAll`, `CallFiles`) |
| `image.go` | grayscale, resize, rotate, skew, crop; `WidthBatches`/`PadBatch` |
| `skew_py.go` | python-exact skew estimator |
| `order.go`, `layout.go` | reading order, layout blocks, markdown |
| `docs/` | `OPTIMIZATIONS_LOG.md` (perf history) |
| `.agents/skills/` | `quality-eval` skill (guard, gauntlet, reading-order, Omni Parse Bench, A/B method) |

## More

- [`README.md`](README.md) — usage/API.
- [`docs/OPTIMIZATIONS_LOG.md`](docs/OPTIMIZATIONS_LOG.md) — perf history.
- [`tools/README.md`](tools/README.md) — regenerating the shipped models.
- [`.agents/skills/quality-eval/SKILL.md`](.agents/skills/quality-eval/SKILL.md) — evaluation method.
