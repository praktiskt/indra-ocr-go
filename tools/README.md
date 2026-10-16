# tools

Regeneration tooling for the shipped models. These are not built into the
library (`tools/` only holds `calib`, a small `main` package).

- `models/rec.onnx` — recognizer, mixed-precision int8 (`quantize_rec.py`).
- `models/order.onnx` — reading order, tab-head-stripped + dynamic int8
  (`quantize_order.py`).
- `models/det.onnx`, `models/orient.onnx`, `models/script.onnx`, `models/rec_cjk.onnx`
  — shipped as upstream FP32 (`script` routes CJK lines to `rec_cjk`).

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
   the optimized graph had ~2 NCHWc ops; after, ~117.
5. **Fuse the confidence head (`--no-fuse-head` to skip).** `conf` is the
   argmax-class probability; replace its 6-node `ReduceMax/Sub/Exp/ReduceSum/
   Reciprocal/Mul` decomposition with `Softmax → ReduceMax`. `ids`/`lengths`
   stay bit-identical, `conf` moves ~2e-3, rec ~7% faster.

Measured on the FUNSD 199-page guard with the default pool (8 workers ×
1 thread, 16-core Zen4): FP32 0.9150 recall / ~17.4 pages/s →
this build 0.9136 recall / ~22 pages/s (+~27% pages/s, −13% peak RSS).

## Regenerate `models/rec.onnx`

```sh
# 1. FP32 source: upstream huggingface.co/datalab-to/indra-ocr (rec.onnx, v3.1),
#    or the pre-int8 file from git history.
git show <pre-int8-commit>:models/rec.onnx > /tmp/rec_fp32.onnx

# 2. Calibration crops, through the real pipeline (upright -> detect -> crop).
#    Any document images work; FUNSD is what the shipped build used.
INDRA_CORPUS_DIR=/path/to/corpus go run ./tools/calib -out /tmp/rec_calib

# 3. Quantize.
uv run --with onnx --with onnxruntime --with numpy \
  tools/quantize_rec.py --model /tmp/rec_fp32.onnx --calib /tmp/rec_calib \
  --out models/rec.onnx --fp32-blocks 2

# 4. Validate against the guard (must print word_recall >= 0.913 for this recipe).
INDRA_CORPUS_DIR=/path/to/corpus go test -run TestCorpusRecall -count=1 -v
```

`tools/quantize_rec.py --help` documents the arguments. The int8 model keeps the
FP32 model's input/output names (`x`, `widths` → `ids`, `conf`, `lengths`), so
nothing in the Go code changes when the file is swapped.

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

# Validate (must print word_recall >= 0.913): the corpus guard is
# bag-of-words, so also check reading order if you change the quantization.
INDRA_CORPUS_DIR=/path/to/corpus go test -run TestCorpusRecall -count=1 -v
```

Outputs stay `a`/`cls`/`same` (the `tab` output is gone), so `session_ort.go`
loads it unchanged.
