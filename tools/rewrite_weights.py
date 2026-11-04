#!/usr/bin/env python3
"""Rewrite individual weights of a shipped QDQ model.

Weight-space programming for this repo: the shipped recognizers keep the
classifier bias in QDQ form, so one class prior is one int32 value with a
float scale. This tool maps characters to class ids through `models/meta.json`,
applies the spec, and writes the model back. Nothing else changes.

Presets:
  vn-priors   raise the bias of the Vietnamese Extended Additional classes
              (U+1EA0..U+1EF9 plus Đ/đ) to --target. Those classes ship at the
              quantization floor (-15.96), which makes them unreachable: the
              head row exists but its prior is crushed. Restoring the prior is
              the minimal edit; every other class is untouched.

Usage (verify with the corpus guard afterwards):
  uv run --with onnx --with numpy tools/rewrite_weights.py \
      --model models/rec.onnx --out models/rec.onnx \
      --preset vn-priors --target -10

A generic JSON spec can also be passed with --spec:

  {"chars": "ẠẢẤ", "target": -10.0}
  {"range": [7810, 7930], "target": -10.0}   # decimal codepoints
  {"chars": "Ạ", "delta": 3.0}

`target` sets each listed class to at least that value (never lowers a higher
prior); `delta` adds to the current value. Class ids are 1-based (blank is 0),
matching the `meta.json` vocab order.
"""

import argparse
import json
import os

import numpy as np
import onnx
from onnx import numpy_helper as NH

VN_RANGE = (0x1EA0, 0x1EF9)
VN_EXTRA = "Đđ"


def load_chars(meta_path):
    meta = json.load(open(meta_path))
    vocab = meta["vocab"] if isinstance(meta["vocab"], dict) else json.loads(meta["vocab"])
    return vocab["chars"]


def find_class_bias(model, nclasses):
    """The per-class bias DQ: its quantized tensor is one value per class."""
    graph = model.graph
    init = {i.name: NH.to_array(i) for i in graph.initializer}
    for node in graph.node:
        if node.op_type != "DequantizeLinear" or not node.output:
            continue
        q, scale = node.input[0], node.input[1]
        if q not in init or scale not in init or init[q].size != nclasses:
            continue
        if not any(n.op_type == "Conv" and node.output[0] in n.input for n in graph.node):
            continue
        zp = node.input[2] if len(node.input) > 2 else None
        return q, scale, zp
    raise SystemExit(f"no per-class Conv bias of length {nclasses} found")


def ids_for(spec, chars):
    ids = []
    if "chars" in spec:
        for ch in spec["chars"]:
            if ch not in chars:
                raise SystemExit(f"character {ch!r} is not in the recognizer vocab")
            ids.append(chars.index(ch) + 1)
    if "range" in spec:
        lo, hi = spec["range"]
        ids += [i + 1 for i, ch in enumerate(chars) if len(ch) == 1 and lo <= ord(ch) <= hi]
    return sorted(set(ids))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "meta.json"))
    ap.add_argument("--preset", choices=["vn-priors"])
    ap.add_argument("--target", type=float, help="bias target for --preset")
    ap.add_argument("--spec", help="JSON edit spec (see module docstring)")
    args = ap.parse_args()

    chars = load_chars(args.meta)
    if args.preset == "vn-priors":
        if args.target is None:
            raise SystemExit("--preset vn-priors needs --target")
        lo, hi = VN_RANGE
        spec = {"range": [lo, hi], "target": args.target}
        extra = [ch for ch in VN_EXTRA if ch in chars]
        if extra:
            spec["chars"] = "".join(extra)
    elif args.spec:
        spec = json.loads(open(args.spec).read())
    else:
        raise SystemExit("pass --preset or --spec")

    model = onnx.load(args.model)
    qname, sname, zpname = find_class_bias(model, len(chars) + 1)
    init = {i.name: NH.to_array(i) for i in model.graph.initializer}
    q = init[qname].astype(np.int64)
    scale = init[sname].astype(np.float64)
    zp = np.asarray(init[zpname], dtype=np.float64).ravel() if zpname else np.zeros_like(scale)
    if zp.size == 1:
        zp = np.full_like(scale, zp[0])
    bias = (q - zp) * scale

    ids = ids_for(spec, chars)
    changed = 0
    for cid in ids:
        old = bias[cid]
        new = max(old, float(spec["target"])) if "target" in spec else old + float(spec["delta"])
        if new != old:
            q[cid] = int(round((new - zp[cid]) / scale[cid]))
            bias[cid] = (q[cid] - zp[cid]) * scale[cid]
            changed += 1
    print(f"{len(ids)} classes selected, {changed} biases rewritten")

    for t in model.graph.initializer:
        if t.name == qname:
            t.CopyFrom(NH.from_array(q.astype(np.int32), qname))
            break
    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
