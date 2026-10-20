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
import onnx
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


def load_calib(directory, rec_height=32):
    """Calibration batches written by tools/calib: x_*.raw (float32, [B,1,H,W])
    and w_*.raw (int64 widths [B])."""
    samples = []
    for wf in sorted(glob.glob(os.path.join(directory, "w_*.raw"))):
        idx = os.path.basename(wf)[2:-4]
        w = np.fromfile(wf, dtype=np.int64)
        x = np.fromfile(os.path.join(directory, f"x_{idx}.raw"), dtype=np.float32)
        b = len(w)
        if b == 0:
            continue
        wdt = x.size // (b * rec_height)
        samples.append((x.reshape(b, 1, rec_height, wdt), w))
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


PROMOTE_HELP = """\
Promote 1-D Convs to 2-D before quantizing. ORT's layout optimizer only turns
2-D Convs into the fast NCHWc layout; 1-D Convs are left on the slow generic
path (microsoft/onnxruntime#16047). The recognizer's blocks are all Conv1d, so
we add a unit spatial axis: kernel [K]->[1,K], pads [p0,p1]->[0,p0,0,p1],
strides [s]->[1,s], dilations [d]->[1,d], weight [OC,IC,K]->[OC,IC,1,K]. The
graph is kept 4-D end to end (SE reduce axes -> (2,3), the widths mask is
unsqueezed to [B,1,1,W], a final Squeeze restores the CTC layout). Numerically
identical (diffs ~1e-6 from accumulation order; 0 CTC lines change)."""


def promote_conv2d(src, dst):
    """Rewrite a Conv1d recognizer graph so every Conv is NCHWc-eligible."""
    import numpy as np
    from onnx import helper as H, numpy_helper as NH

    m = onnx.load(src)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    nodes = list(g.node)
    const = {n.output[0]: list(NH.to_array(n.attribute[0].t).ravel())
             for n in nodes if n.op_type == "Constant" and n.attribute}

    def resolve(x):
        if x in init:
            return list(NH.to_array(init[x]).ravel())
        return const.get(x)

    def ints(n, name, default=None):
        for a in n.attribute:
            if a.name == name:
                return list(a.ints)
        return default

    STEM = "/rec/stem/Conv_output_0"
    SQ = "/rec/Squeeze_output_0"
    HEAD = "/rec/head/Conv_output_0"
    MASK3 = "/rec/Cast_4_output_0"

    for n in nodes:
        if n.op_type == "BatchNormalization" and n.input[0] == SQ:
            n.input[0] = STEM

    for n in nodes:
        if n.op_type != "Conv" or n.name == "/rec/stem/Conv":
            continue
        k = ints(n, "kernel_shape", [1])
        if len(k) != 1:
            continue
        pads = ints(n, "pads", [0, 0])
        strides = ints(n, "strides", [1])
        dil = ints(n, "dilations", [1])
        for a in n.attribute:
            if a.name == "kernel_shape":
                a.ints[:] = [1] + k
            elif a.name == "pads":
                a.ints[:] = [0, pads[0], 0, pads[1]]
            elif a.name == "strides":
                a.ints[:] = [1] + strides
            elif a.name == "dilations":
                a.ints[:] = [1] + dil
        wn = n.input[1]
        if wn in init:
            w = NH.to_array(init[wn])
            if w.ndim == 3:
                init[wn].CopyFrom(NH.from_array(w.reshape(w.shape[0], w.shape[1], 1, w.shape[2]), wn))

    producer = {o: n for n in nodes for o in n.output}
    for n in nodes:
        if n.op_type != "ReduceSum":
            continue
        p = producer.get(n.input[0])
        if p is not None and p.name.endswith("/se/Mul") and resolve(n.input[1]) == [2]:
            nm = "z_se_axes_" + n.name.split("/")[2]
            init[nm] = NH.from_array(np.array([2, 3], dtype=np.int64), nm)
            n.input[1] = nm

    for n in nodes:
        for i, inp in enumerate(n.input):
            if inp != MASK3:
                continue
            if n.op_type == "Mul":
                n.input[i] = "z_mask4"
            elif n.op_type == "ReduceSum":
                n.input[0] = "z_mask4"
                init["z_mask_count_axes"] = NH.from_array(np.array([2, 3], dtype=np.int64), "z_mask_count_axes")
                n.input[1] = "z_mask_count_axes"
    init["z_mask_uaxes"] = NH.from_array(np.array([2], dtype=np.int64), "z_mask_uaxes")
    mpi = next(i for i, n in enumerate(nodes) if n.output and n.output[0] == MASK3)
    nodes.insert(mpi + 1, H.make_node("Unsqueeze", [MASK3, "z_mask_uaxes"], ["z_mask4"], name="z_mask_unsq"))

    init["z_head_axes"] = NH.from_array(np.array([2], dtype=np.int64), "z_head_axes")
    for n in nodes:
        if n.input and n.input[0] == HEAD and n.op_type not in ("Conv", "Squeeze"):
            n.input[0] = "z_head3"
    hi = next(i for i, n in enumerate(nodes) if n.name == "/rec/head/Conv")
    nodes.insert(hi + 1, H.make_node("Squeeze", [HEAD, "z_head_axes"], ["z_head3"], name="z_head_sq"))

    have = {i.name for i in g.initializer}
    for k, v in init.items():
        if k not in have:
            g.initializer.append(v)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    return dst


def fuse_conf_head(src, dst):
    """Replace the recognizer's confidence branch with a fused softmax-max.

    `conf` is the probability of the argmax class, currently computed as
    ReduceMax + Sub + Exp + ReduceSum + Reciprocal + Mul (16 nodes with their
    Q/DQ wrappers) = several elementwise passes over [B,T,V]. ORT's Softmax
    kernel does the same math in one fused pass, and `ids`/`lengths` are on a
    different branch and stay bit-identical. conf differs only by accumulation
    order (~2e-3). Measured: rec ~7%, end-to-end ~4% faster.
    """
    from onnx import helper as H

    m = onnx.load(src)
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}

    def reach(outs):
        s, st = set(), list(outs)
        while st:
            t = st.pop()
            n = prod.get(t)
            if not n or n.name in s:
                continue
            s.add(n.name)
            st.extend(n.input)
        return s

    conf_only = reach(["conf"]) - reach([o.name for o in g.output if o.name != "conf"])
    # logits = the external input consumed by the softmax-max Sub node
    logits = None
    for n in g.node:
        if n.name in conf_only and n.op_type == "Sub":
            for inp in n.input:
                p = prod.get(inp)
                if p is None or p.name not in conf_only:
                    logits = inp
                    break
    if logits is None:
        raise ValueError("conf branch not found")
    kept = [n for n in g.node if n.name not in conf_only]
    kept += [H.make_node("Softmax", [logits], ["z_conf_sm"], name="z_conf_sm", axis=-1),
             H.make_node("ReduceMax", ["z_conf_sm"], ["conf"], name="z_conf_max", axes=[-1], keepdims=0)]
    del g.node[:]
    g.node.extend(kept)
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    return len(conf_only)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="FP32 rec.onnx source")
    ap.add_argument("--calib", required=True, help="dir with x_*.raw / w_*.raw calibration batches")
    ap.add_argument("--out", required=True, help="output int8 onnx path")
    ap.add_argument("--fp32-blocks", type=int, default=2,
                    help="leading inverted-residual blocks kept FP32 (default 2; 0 = quantize all)")
    ap.add_argument("--no-promote", action="store_true", help="skip the Conv1d->Conv2d NCHWc promotion")
    ap.add_argument("--no-fuse-head", action="store_true", help="skip the softmax-max confidence-head fusion")
    ap.add_argument("--rec-height", type=int, default=32,
                    help="recognizer input height (32 band; 96 for the windowed base rec)")
    args = ap.parse_args()

    samples = load_calib(args.calib, args.rec_height)
    if not samples:
        sys.exit(f"no calibration batches in {args.calib}")
    lines = sum(len(w) for _, w in samples)
    excl = fp32_nodes(args.model, args.fp32_blocks)
    print(f"calibration: {len(samples)} batches / {lines} lines")
    print(f"FP32 nodes ({len(excl)}): stem + blocks 0..{args.fp32_blocks - 1}")

    src = args.model
    if not args.no_promote:
        src = args.out + ".promoted.onnx"
        promote_conv2d(args.model, src)
        print(f"promoted Conv1d->Conv2d -> {src}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    quantize_static(
        src,
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
    if not args.no_fuse_head:
        tmp = args.out + ".prehead.onnx"
        os.replace(args.out, tmp)
        n = fuse_conf_head(tmp, args.out)
        os.remove(tmp)
        print(f"fused confidence head (-{n} nodes -> Softmax+ReduceMax)")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
