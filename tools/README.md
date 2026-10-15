# tools

Regeneration tooling for the shipped model. These are not built into the
library (`tools/` only holds `calib`, a small `main` package).

The recognizer in `models/rec.onnx` is a mixed-precision int8 build of the
upstream FP32 `rec.onnx`. Everything here reproduces it.

## Why this exact recipe

Three choices carried the accuracy (details and the A/B evidence are in
`.agents/skills/corpus-eval/SKILL.md`):

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

Measured on the FUNSD 199-page guard with the default pool (8 workers ×
1 thread, 16-core Zen4): FP32 0.9150 recall / 17.8 pages/s →
this build 0.9134 recall / 22.3 pages/s (+25% pages/s, −13% peak RSS).

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
