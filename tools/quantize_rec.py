#!/usr/bin/env python3
"""Quantize the Indra recognizer (rec.onnx) to int8 for the Go port.

Reproduces the shipped models/rec.onnx: a QDQ, per-channel, mixed-precision
int8 build. The recipe matters more than the bits -- three choices carried the
accuracy (see .agents/skills/corpus-eval/SKILL.md):

  1. Quantize the RAW graph. Do NOT run ORT's quant_pre_process first: it
     decomposes HardSwish into HardSigmoid+Mul and quantizes more tensors,
     which costs both accuracy and speed here.
  2. Asymmetric activations, per-channel weights (QUInt8 activations, QInt8
     weights, MinMax). Asymmetric activations beat ActivationSymmetric by
     ~0.6pp corpus word recall at equal speed.
  3. Keep the stem and the first few blocks in FP32 (mixed precision). This is
     the "co-calibrated" slot: the earliest layers are where int8 hurts most,
     and leaving them float buys back most of the loss.

Measured (FUNSD 199-page guard, 8x1-thread pool on a 16-core Zen4):
  fp32 baseline          0.9150 recall   17.8 pages/s
  this build (2 blocks)  0.9134 recall   22.3 pages/s   (+25% pages/s)

Usage (needs onnx/onnxruntime/numpy; uv handles it):
  uv run --with onnx --with onnxruntime --with numpy \
      tools/quantize_rec.py --model rec_fp32.onnx --calib ./calib --out models/rec.onnx

The FP32 source is the upstream model (huggingface.co/datalab-to/indra-ocr,
rec.onnx), or the pre-int8 file from git history:
  git show <pre-int8-commit>:models/rec.onnx > rec_fp32.onnx
"""

import argparse
import glob
import os
import sys

import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

# Node-name prefixes kept in FP32. /rec/stem is the initial conv, /rec/HardSwish
# and /rec/Mul are its activation, and blocks.N are the inverted-residual
# blocks (stem + the first `--fp32-blocks` of them).
STEM_PREFIXES = ["/rec/stem", "/rec/HardSwish", "/rec/Mul"]


def load_calib(directory):
    """Calibration batches written by tools/calib: x_*.raw (float32, [B,1,32,W])
    and w_*.raw (int64 widths [B])."""
    samples = []
    for wf in sorted(glob.glob(os.path.join(directory, "w_*.raw"))):
        idx = os.path.basename(wf)[2:-4]
        w = np.fromfile(wf, dtype=np.int64)
        x = np.fromfile(os.path.join(directory, f"x_{idx}.raw"), dtype=np.float32)
        b = len(w)
        if b == 0:
            continue
        wdt = x.size // (b * 32)
        samples.append((x.reshape(b, 1, 32, wdt), w))
    return samples


class SampleReader(CalibrationDataReader):
    def __init__(self, samples):
        self.it = iter(samples)

    def get_next(self):
        try:
            x, w = next(self.it)
            return {"x": x, "widths": w}
        except StopIteration:
            return None


def fp32_nodes(model_path, fp32_blocks):
    import onnx

    names = [n.name for n in onnx.load(model_path).graph.node]
    prefixes = list(STEM_PREFIXES)
    prefixes += [f"/rec/blocks.{i}/" for i in range(fp32_blocks)]
    return [n for n in names if any(n.startswith(p) for p in prefixes)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="FP32 rec.onnx source")
    ap.add_argument("--calib", required=True, help="dir with x_*.raw / w_*.raw calibration batches")
    ap.add_argument("--out", required=True, help="output int8 onnx path")
    ap.add_argument("--fp32-blocks", type=int, default=2,
                    help="leading inverted-residual blocks kept FP32 (default 2; 0 = quantize all)")
    args = ap.parse_args()

    samples = load_calib(args.calib)
    if not samples:
        sys.exit(f"no calibration batches in {args.calib}")
    lines = sum(len(w) for _, w in samples)
    excl = fp32_nodes(args.model, args.fp32_blocks)
    print(f"calibration: {len(samples)} batches / {lines} lines")
    print(f"FP32 nodes ({len(excl)}): stem + blocks 0..{args.fp32_blocks - 1}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    quantize_static(
        args.model,
        args.out,
        SampleReader(samples),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False},
        nodes_to_exclude=excl,
    )
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
