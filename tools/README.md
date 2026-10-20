# tools

Regeneration tooling for the shipped models. These are not built into the
library (`tools/` only holds `calib`, a small `main` package).

- `models/rec.onnx` — base recognizer, mixed-precision int8 (`quantize_rec.py`,
  `--no-promote --rec-height 96`).
- `models/rec_cjk.onnx`, `models/rec_rtl.onnx`, `models/rec_world.onnx` — the
  script-group recognizers, shipped as upstream **FP32**. Group-rec int8 needs a
  per-script calibration set and a per-script guard; see below.
- `models/order.onnx` — reading order, tab-head-stripped + dynamic int8
  (`quantize_order.py`).
- `models/det.onnx`, `models/orient.onnx`, `models/script.onnx` and the math
  classifier (`mathcls`) — shipped as upstream FP32. The math encoder/decoder
  (`math_enc` + `math_dec_step`) are dynamic int8 (`quantize_math.py`; the
  decoder's logits head is excluded); `math_dec_init` stays FP32 (its MatMuls
  are not direct-initializer weights, so dynamic quant is a no-op). `script`
  routes each line to a group recognizer; the math models are used with
  `OCR.Math`.
- `compose_table.go` (repo root) — the spacing-diacritic composition table used
  by `foldSpacingAccents` in `pdfium.go` (`gen_compose.py`).

The recognizer in `models/rec.onnx` is a mixed-precision int8 build of the
upstream FP32 `rec.onnx`. Everything here reproduces it.

## Why this exact recipe

Three choices carried the accuracy (details and the A/B evidence are in
`.agents/skills/quality-eval/SKILL.md`):

1. **Quantize the raw graph.** Do not run ORT `quant_pre_process` first. It
   decomposes `HardSwish` into `HardSigmoid`+`Mul` and quantizes more tensors,
   costing both accuracy and speed (142 vs 132 `QuantizeLinear`).
2. **Asymmetric activations, per-channel weights** (`QUInt8`/`QInt8`, MinMax).
   Asymmetric beats `ActivationSymmetric` by ~0.6pp recall at equal speed.
3. **Mixed precision: stem + first blocks FP32.** The earliest layers are where
   int8 hurts most; leaving the stem (`/rec/stem`, its `HardSwish`/`Mul`) and
   the first two inverted-residual blocks float recovers most of the loss.
   `--fp32-blocks 0` quantizes everything (faster, less accurate); larger
   values trade speed back for accuracy.
4. **Promote Conv1d → Conv2d (`--no-promote` to skip).** ORT's layout optimizer
   only lowers 2-D Convs to the fast NCHWc layout; 1-D Convs stay on the slow
   generic path (microsoft/onnxruntime#16047). The recognizer's blocks are all
   `Conv1d`, so this adds a unit spatial axis and keeps the graph 4-D. Pure
   speed, no accuracy: `+22%` on the fp32 recognizer, `+8%` on int8, `0` CTC
   lines changed (diffs ~1e-6 from accumulation order). Before the promotion
   the optimized graph had ~2 NCHWc ops; after, ~117. **Not used for the v1.1
   base rec**: its new `/rec/context` module has a genuinely 1-D `.proj/Conv`
   whose input stays 3-D, so promoting it breaks shape inference — the base
   build skips promotion (`--no-promote`); still faster than FP32 at 1 thread.
5. **Fuse the confidence head (`--no-fuse-head` to skip).** `conf` is the
   argmax-class probability; replace its 6-node `ReduceMax/Sub/Exp/ReduceSum/
   Reciprocal/Mul` decomposition with `Softmax → ReduceMax`. `ids`/`lengths`
   stay bit-identical, `conf` moves ~2e-3, rec ~7% faster.

Measured on the FUNSD 199-page guard with the default pool (8 workers ×
1 thread, 16-core Zen4), **v0.3.2**: FP32 0.9150 recall / ~17.4 pages/s →
this build 0.9136 recall / ~22 pages/s (+~27% pages/s, −13% peak RSS).
**v1.1.0** (threads=2): FP32 `rec` 0.8658 → int8 0.8680 (neutral).

## Regenerate `models/rec.onnx`

```sh
# 1. FP32 source: upstream huggingface.co/datalab-to/indra-ocr (rec.onnx, v1.1.0),
#    or the pre-int8 file from git history.
git show <pre-int8-commit>:models/rec.onnx > /tmp/rec_fp32.onnx

# 2. Calibration crops, through the real pipeline (upright -> detect -> crop).
#    Any document images work; FUNSD is what the shipped build used.
INDRA_CORPUS_DIR=/path/to/corpus go run ./tools/calib -out /tmp/rec_calib

# 3. Quantize. The v1.1 base graph has a /rec/context module that the
#    Conv1d->Conv2d promotion does not handle (: .proj/Conv is genuinely 1-D and
#    make it 2-D breaks shape inference), so use --no-promote. The base rec reads
#    a 96px window (rec_window=3): --rec-height 96.
uv run --with onnx --with onnxruntime --with numpy \
  tools/quantize_rec.py --model /tmp/rec_fp32.onnx --calib /tmp/rec_calib \
  --out models/rec.onnx --fp32-blocks 2 --no-promote --rec-height 96

# 4. Validate against the guard (v1.1.0: must print word_recall >= 0.86).
INDRA_CORPUS_DIR=/path/to/corpus go test -run TestCorpusRecall -count=1 -v
```

`tools/quantize_rec.py --help` documents the arguments. The int8 model keeps the
FP32 model's input/output names (`x`, `widths` → `ids`, `conf`, `lengths`,
`blank`, `topv`, `topi`), so nothing in the Go code changes when the file is
swapped.

## Script-group recognizers (`rec_cjk`, `rec_rtl`, `rec_world`)

Shipped **FP32**. The v1.1.0 models are FP32 on the Hub; quantizing them needs
(1) calibration on *their* script — Latin FUNSD crops make `rec_cjk`
confabulate CJK (chinese_essay char-sim 0.62 vs FP32) — and (2) a per-script
quality guard (the FUNSD guard is Latin-only; use OPB `multilingual` +
`multiling_finepdfs`, or a scratch subset of those pages).

With a CJK calibration set (synthetic Noto CJK pages + real Hani/Jpan pages)
`rec_cjk` int8 recovers (chinese_essay 0.98, `cmn_Hani` 0.987). `rec_world`
(with Greek/Cyrillic/Devanagari/Armenian/Bengali) and `rec_rtl` (Arabic/Hebrew)
still regressed on 27 real OPB multilingual pages even with per-script
calibration (mean char-sim 0.83), so they stay FP32 until a guard exists.
The tooling is ready:

```sh
INDRA_CORPUS_DIR=/path/to/script_corpus go run ./tools/calib -out /tmp/g_calib -window 1
uv run --with onnx --with onnxruntime --with numpy \
  tools/quantize_rec.py --model /tmp/rec_<group>_fp32.onnx --calib /tmp/g_calib \
  --out models/rec_<group>.onnx --fp32-blocks 2 --rec-height 32
```

## Regenerate `models/order.onnx`

Two steps, both in `tools/quantize_order.py`: remove the dead `tab` head (the
port reads only `a`/`cls`/`same`), then dynamic-int8 the MatMul/Gemm weights.
2.97 MB → 0.76 MB, recall and reading order unchanged (see
`docs/OPTIMIZATIONS_LOG.md`).

```sh
# FP32 source: upstream huggingface.co/datalab-to/indra-ocr (order.onnx),
# or the shipped pre-quantization file from git history.
git show 293f1db:models/order.onnx > /tmp/order_fp32.onnx

uv run --with onnx --with onnxruntime --with numpy \
  tools/quantize_order.py --model /tmp/order_fp32.onnx --out models/order.onnx

# --lossless strips the tab head but keeps FP32 (bit-identical outputs).

# Validate (must print word_recall >= 0.86): the corpus guard is
# bag-of-words, so also check reading order if you change the quantization.
INDRA_CORPUS_DIR=/path/to/corpus go test -run TestCorpusRecall -count=1 -v
```

Outputs stay `a`/`cls`/`same` (the `tab` output is gone), so `session_ort.go`
loads it unchanged.

## Regenerate the math models

Dynamic int8 (per-channel `QUInt8` on MatMul/Gemm weights, no calibration — the
`order.onnx` recipe), via `tools/quantize_math.py`. The math models are a
Conformer encoder + autoregressive decoder used with `OCR.Math` (off by
default).

```sh
uv run --with onnx --with onnxruntime --with numpy \
  tools/quantize_math.py --model /tmp/math_dec_step_fp32.onnx \
  --out models/math_dec_step.onnx --exclude /out/MatMul
uv run --with onnx --with onnxruntime --with numpy \
  tools/quantize_math.py --model /tmp/math_enc_fp32.onnx --out models/math_enc.onnx
```

`--exclude /out/MatMul` keeps the decoder's final vocab projection FP32: it feeds
an argmax over 652 classes, and quantizing it flipped tokens (OPB
`olmo_arxiv_math` text 0.61484 -> 0.61272; excluding it 0.61449). Validate any
math-model change on OPB `olmo_arxiv_math` (514 pages, `-math`) — the `text`
family is the signal; FUNSD/xfund do not exercise the math models.

## Regenerate `compose_table.go`

`compose_table.go` (repo root) holds the spacing-diacritic → combining-mark map
and the canonical (mark, base) → precomposed table used by `foldSpacingAccents`
in `pdfium.go`. It is generated from Python's stdlib `unicodedata` (no
dependencies) — do not hand-edit:

```sh
python3 tools/gen_compose.py > compose_table.go
```

`gen_compose.py` also maps the legacy dotless-i/j bases (some fonts draw an
accented i/j as a dotless glyph) to the dotted base's result. To add an accent,
add its spacing glyph to `SPACING` and regenerate.

## Native libraries (`lib/`)

These are embedded (linux/amd64) and loaded at runtime — ONNX Runtime via
`onnxruntime_go`, PDFium via `pdfium.go` (both from an anonymous memfd). See
[`../LICENSE-LIBS`](../LICENSE-LIBS) for licenses.

- **`lib/libpdfium.so`** — pinned to the pdfium-binaries release
  **`chromium/8076`** (PDFium **156.0.8076.0**), sha256
  `fd4de9f91f183476d04b1bc3fd17dacd95aba349c55d2c376512d95e082b441e`.
  Reproduce/refresh with:

  ```sh
  sh tools/vendor_pdfium.sh
  ```

  To bump: set `RELEASE`/`SHA256` in the script and re-validate — rendering
  feeds detect/rec/order, so run the guard and the OPB PDF suite before
  committing (see [`../.agents/skills/quality-eval/SKILL.md`](../.agents/skills/quality-eval/SKILL.md)).
- **`lib/libonnxruntime.so`**, **`lib/libstdc++.so.6`**, **`lib/libgcc_s.so.1`** —
  ONNX Runtime **1.31.0** and its GCC 13 C++ runtime, embedded as shipped (no
  fetch script).
