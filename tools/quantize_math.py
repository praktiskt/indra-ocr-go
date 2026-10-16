#!/usr/bin/env python3
"""Shrink the Indra math models (encoder + decoder) for the Go port.

The math models are a small Conformer/transformer encoder (`math_enc`), a
cross-attention K/V projection (`math_dec_init`) and an autoregressive decoder
step (`math_dec_step`), plus a tiny CNN classifier (`mathcls`). Upstream ships
them FP32.

Two independent, mostly-lossless steps:

  1. **Dynamic int8** (per-channel QUInt8) on MatMul/Gemm weights -- the standard
     transformer recipe, no calibration needed (same as tools/quantize_order.py).
     This is the whole win for `math_dec_init` (runs once per batch) and most of
     it for `math_enc` / `math_dec_step`.

  2. **Protect the logits head.** `math_dec_step`'s final projection
     (`/out/MatMul`, channels -> vocab) feeds an argmax over 652 classes, so
     quantizing it flips tokens: excluding it moved OPB `olmo_arxiv_math` text
     0.61272 -> 0.61449 (fp32 0.61484) for +0.37 MB. Pass `--exclude /out/MatMul`
     for the shipped `math_dec_step`.

Usage (needs onnx/onnxruntime; uv handles it):
  uv run --with onnx --with onnxruntime --with numpy \\
      tools/quantize_math.py --model math_dec_step.onnx --out models/math_dec_step.onnx \\
      --exclude /out/MatMul
  uv run --with onnx --with onnxruntime --with numpy \\
      tools/quantize_math.py --model math_enc.onnx --out models/math_enc.onnx
  uv run --with onnx --with onnxruntime --with numpy \\
      tools/quantize_math.py --model math_dec_init.onnx --out models/math_dec_init.onnx

Validate on OPB `olmo_arxiv_math` (514 pages) with `-math`; the `text` family is
the signal. FUNSD/xfund are Latin/Latin+CJK and do not exercise the math models.
"""
import argparse
import os
import sys

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="FP32 math .onnx source")
    ap.add_argument("--out", required=True, help="output int8 onnx path")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="node names to keep FP32 (e.g. /out/MatMul for math_dec_step)")
    ap.add_argument("--weight-type", choices=["QUInt8", "QInt8"], default="QUInt8")
    ap.add_argument("--no-per-channel", action="store_true",
                    help="per-tensor weight scales instead of per-channel")
    args = ap.parse_args()

    wt = QuantType.QUInt8 if args.weight_type == "QUInt8" else QuantType.QInt8
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    quantize_dynamic(
        args.model, args.out,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=wt,
        per_channel=not args.no_per_channel,
        nodes_to_exclude=args.exclude,
    )
    # Sanity: input/output names must be unchanged so the Go sessions load it.
    a, b = onnx.load(args.model).graph, onnx.load(args.out).graph
    if [i.name for i in a.input] != [i.name for i in b.input] or \
       [o.name for o in a.output] != [o.name for o in b.output]:
        sys.exit("signature changed: refusing to ship")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB, "
          f"from {os.path.getsize(args.model) / 1e6:.2f} MB"
          + (f", excluded {args.exclude}" if args.exclude else "") + ")")


if __name__ == "__main__":
    main()
