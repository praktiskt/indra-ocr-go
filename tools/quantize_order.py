#!/usr/bin/env python3
"""Shrink the Indra reading-order model (order.onnx) for the Go port.

Reproduces the shipped models/order.onnx from the upstream FP32 graph
(huggingface.co/datalab-to/indra-ocr, or `git show 293f1db:models/order.onnx`).

Two lossless-or-near-lossless steps:

  1. Strip the dead `tab` head. The Go port renders tables geometrically and
     never reads it (session_ort.go asks for a/cls/same only). 38 tab-only
     nodes + 4 initializers; a/cls/same are bit-identical (verified).

  2. Dynamic int8 (per-channel QUInt8) on the MatMul/Gemm weights. order.onnx
     is 2.93 MB of fp32 transformer weights; int8 is ~4x smaller. Dynamic
     quantization (no calibration) is the standard transformer recipe.

Claimed (corpus + gauntlet in docs/OPTIMIZATIONS_LOG.md): recall identical on
the 199-page FUNSD guard and all 9 brutal-print subsets; reading order neutral
(the guard is bag-of-words, so a line-order Kendall-tau metric was added).

Usage (needs onnx/onnxruntime/numpy; uv handles it):
  uv run --with onnx --with onnxruntime --with numpy \\
      tools/quantize_order.py --model order_fp32.onnx --out models/order.onnx

  --lossless  strip the tab head but keep FP32 (bit-identical outputs)
  --no-strip  quantize without removing the tab head
"""
import argparse
import os
import sys

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def strip_tab(src, dst):
    m = onnx.load(src)
    g = m.graph
    prog = {o: n for n in g.node for o in n.output}

    def reach(outs):
        s, st = set(), list(outs)
        while st:
            t = st.pop()
            n = prog.get(t)
            if not n or n.name in s:
                continue
            s.add(n.name)
            st.extend(n.input)
        return s

    keep_names = reach([o.name for o in g.output if o.name != "tab"])
    kept = [n for n in g.node if n.name in keep_names]
    dropped = len(g.node) - len(kept)
    used = {i for n in kept for i in n.input}
    inits = [i for i in g.initializer if i.name in used or i.name in {o.name for o in g.output}]
    outputs = [o for o in g.output if o.name != "tab"]
    del g.node[:]
    g.node.extend(kept)
    del g.initializer[:]
    g.initializer.extend(inits)
    del g.output[:]
    g.output.extend(outputs)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    return dropped, len(g.initializer), len(inits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="FP32 order.onnx source")
    ap.add_argument("--out", required=True, help="output order.onnx")
    ap.add_argument("--lossless", action="store_true", help="strip tab head only, keep FP32")
    ap.add_argument("--no-strip", action="store_true", help="skip tab-head removal")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    work = args.model
    if not args.no_strip:
        work = args.out + ".stripped.onnx"
        dropped, ni, nj = strip_tab(args.model, work)
        print(f"stripped tab head: -{dropped} nodes, initializers {ni} -> {nj}")
    if args.lossless:
        if work == args.model:
            sys.exit("--lossless needs --no-strip to be off")
        os.replace(work, args.out)
    else:
        quantize_dynamic(work, args.out, op_types_to_quantize=["MatMul", "Gemm"],
                         weight_type=QuantType.QUInt8, per_channel=True)
        if work != args.model:
            os.remove(work)
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
