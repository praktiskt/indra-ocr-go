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


def promote_int8(src, dst):
    """Promote the block Conv1ds of an ALREADY-QUANTIZED rec.onnx to Conv2d.

    Same goal as promote_conv2d, but working on the shipped QDQ int8 file:
    no re-quantization and no calibration — every Q/DQ pair is preserved, so
    the quantized numerics come out bit-identical. Only the 50 block convs are
    promoted; /rec/head/Conv and /rec/context/proj/Conv stay 1-D (one Squeeze
    bridges the backbone to head; the CTC tail is untouched). The stem region
    stays untouched and is bridged back to 4-D with Unsqueeze nodes; the
    widths mask, the SE ReduceSums and the Shape-derived Gather that feeds
    the mask length get 4-D-aware axes/indices.

    RESULT (see docs/OPTIMIZATIONS_LOG.md): lossless but NOT faster — ORT's
    NCHWc transformer is fp32-only, so quantized convs never get the blocked
    layout at any rank. Kept as a working capability for the day ORT extends
    NCHWc to int8; do not use for the shipped models (they stay 1-D).
    """
    import numpy as np
    from onnx import helper as H, numpy_helper as NH

    m = onnx.load(src)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    nodes = list(g.node)
    consumers = {}
    for n in nodes:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)
    # Original ranks (before any edit): the mask's length is derived at runtime
    # as Shape(stem-region)[2], and after the Conv1d -> Conv2d promotion the
    # stem region is 4-D, so any hardcoded positive Gather index pointing at
    # the old position axis (2) must be remapped to 3 (kept for context/head,
    # whose rank is unchanged).
    orig_ranks = {}
    try:
        mi = onnx.shape_inference.infer_shapes(m)
        for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output):
            orig_ranks[v.name] = len(v.type.tensor_type.shape.dim)
    except Exception:
        orig_ranks = {}

    def chain_last(t):
        """Follow a linear elementwise (Q/DQ/Cast) chain forward from t."""
        while True:
            cs = [n for n in consumers.get(t, []) if n.op_type not in ("Squeeze", "Unsqueeze")]
            if len(cs) != 1:
                return t
            c = cs[0]
            if c.op_type not in ("QuantizeLinear", "DequantizeLinear", "Cast"):
                return t
            t = c.output[0]

    def promote_init(name):
        """Reshape a [OC,IC,K] weight to [OC,IC,1,K], direct or behind DQ."""
        if name in init:
            if len(init[name].dims) == 4:
                return
            assert len(init[name].dims) == 3, f"{name}: dims {init[name].dims}"
            w = NH.to_array(init[name])
            init[name].CopyFrom(NH.from_array(w.reshape(w.shape[0], w.shape[1], 1, w.shape[2]), name))
            return
        p = prod.get(name)
        if p is not None and p.op_type == "DequantizeLinear":
            promote_init(p.input[0])
            return
        raise ValueError(f"cannot promote weight {name}")

    prod = {o: n for n in nodes for o in n.output}

    # 1. Promote block convs; head/proj stay 1-D, stem/context convs are 2-D.
    promoted = 0
    for n in nodes:
        if not n.name.startswith("/rec/blocks.") or n.op_type != "Conv":
            continue
        for a in n.attribute:
            if a.name == "kernel_shape" and len(a.ints) == 1:
                a.ints[:] = [1] + list(a.ints)
            elif a.name == "pads" and len(a.ints) == 2:
                a.ints[:] = [0, a.ints[0], 0, a.ints[1]]
            elif a.name == "strides" and len(a.ints) == 1:
                a.ints[:] = [1] + list(a.ints)
            elif a.name == "dilations" and len(a.ints) == 1:
                a.ints[:] = [1] + list(a.ints)
        promote_init(n.input[1])
        promoted += 1

    # axes-rewrite helper: axes can be an initializer or a Constant node output.
    def axes_producer(ax):
        """Return a callable writing a new axes array into ax's source."""
        if ax in init:
            return lambda arr: init[ax].CopyFrom(NH.from_array(arr, ax))
        p = prod.get(ax)
        if p is not None and p.op_type == "Constant":
            return lambda arr: p.attribute[0].t.CopyFrom(NH.from_array(arr, ax))
        raise ValueError(f"cannot rewrite axes {ax}")

    # 2. Widths mask: bridge to 4-D for every consumer.
    mask3 = chain_last("/rec/Cast_4_output_0")
    mask4 = "z_mask4"
    for n in nodes:
        if n.op_type in ("Mul", "ReduceSum"):
            for i, inp in enumerate(n.input):
                if inp == mask3:
                    n.input[i] = mask4
    # ReduceSums that consume the mask (the SE count sums): axes [2] -> [2, 3].
    for n in nodes:
        if n.op_type != "ReduceSum" or n.input[0] != mask4:
            continue
        axes_producer(n.input[1])(np.array([2, 3], dtype=np.int64))

    # 4-D mask source.
    nodes.insert(
        next(i for i, x in enumerate(nodes) if x.output and x.output[0] == mask3) + 1,
        H.make_node("Unsqueeze", [mask3, "z_mask_uaxes"], [mask4], name="z_mask_unsq"),
    )
    init["z_mask_uaxes"] = NH.from_array(np.array([2], dtype=np.int64), "z_mask_uaxes")

    # 3. SE pooling ReduceSums (axes [2] -> [2,3]; sum over the sequence axis).
    se_sums = [
        n for n in nodes
        if n.op_type == "ReduceSum" and n.name.startswith("/rec/blocks.") and n.input[0] != mask4
    ]
    for n in se_sums:
        axes_producer(n.input[1])(np.array([2, 3], dtype=np.int64))
    assert se_sums, "no SE pooling ReduceSums found"

    # 4. Bridge the 3-D stem region (and the 3-D context proj) into 4-D.
    #    base rec:   Squeeze -> Add, proj -> Add   (both ends of /rec/Add)
    #    group recs: Squeeze -> BatchNormalization
    bridges = []
    for n in nodes:
        for idx, inp in enumerate(n.input):
            src = None
            if n.op_type == "Add" and inp == chain_last("/rec/Squeeze_output_0"):
                src = "/rec/Squeeze_output_0"
            elif n.op_type == "BatchNormalization" and inp == chain_last("/rec/Squeeze_output_0"):
                src = "/rec/Squeeze_output_0"
            elif n.op_type == "Add" and inp == chain_last("/rec/context/proj/Conv_output_0"):
                src = "/rec/context/proj/Conv_output_0"
            if src is not None:
                bridges.append((n, idx, chain_last(src)))
    src4 = "z_bridge_axes"
    init[src4] = NH.from_array(np.array([2], dtype=np.int64), src4)
    for j, (n, idx, source) in enumerate(bridges):
        out = f"z_bridge_{j}"
        nodes.insert(
            next(i for i, x in enumerate(nodes) if x.output and x.output[0] == source) + 1,
            H.make_node("Unsqueeze", [source, src4], [out], name=f"z_bridge_unsq_{j}"),
        )
        n.input[idx] = out

    # 5. Bridge blocks.9 -> head conv (head stays 1-D, CTC tail untouched).
    head = next(n for n in nodes if n.name == "/rec/head/Conv")
    blk9 = chain_last("/rec/blocks.9/Mul_output_0")
    assert head.input[0] == blk9, f"head conv input moved: {head.input[0]}"
    nodes.insert(
        next(i for i, x in enumerate(nodes) if x.output and x.output[0] == blk9) + 1,
        H.make_node("Squeeze", [blk9, "z_blk9_axes"], ["z_blk9_sq"], name="z_blk9_sq"),
    )
    head.input[0] = "z_blk9_sq"
    init["z_blk9_axes"] = NH.from_array(np.array([2], dtype=np.int64), "z_blk9_axes")

    # 6. Topological order (insertions above can violate it).
    out2nodes = {o: i for i, n in enumerate(nodes) for o in n.output}
    dep = [sum(1 for inp in n.input if inp in out2nodes) for n in nodes]
    adj = {i: [] for i in range(len(nodes))}
    for i, n in enumerate(nodes):
        for inp in n.input:
            if inp in out2nodes:
                adj[out2nodes[inp]].append(i)
    order = [i for i in range(len(nodes)) if dep[i] == 0]
    pos = 0
    while pos < len(order) and len(order) < len(nodes):
        x = order[pos]
        pos += 1
        for y in adj[x]:
            dep[y] -= 1
            if dep[y] == 0:
                order.append(y)
    assert len(order) == len(nodes), "cycle introduced"
    new_nodes = [nodes[i] for i in order]

    have = {i.name for i in g.initializer}
    for k, v in init.items():
        if k not in have:
            g.initializer.append(v)
    # Drop the fp32 export's declared intermediate shapes (whose rank changed)
    # and the output shapes: ORT re-infers 4-D from the rewritten graph.
    del g.value_info[:]
    for o in g.output:
        if o.type.tensor_type.HasField("shape"):
            for d in o.type.tensor_type.shape.dim:
                d.ClearField("dim_value"); d.ClearField("dim_param")
    del g.node[:]
    g.node.extend(new_nodes)

    # Remap hardcoded positive Gather indices over Shape chains whose source
    # changed rank 3 -> 4 (axis 2 was the position axis in [B,C,L]; it is 3 in
    # [B,C,1,L]). Negative indices and rank-unchanged sources are left alone.
    if orig_ranks:
        try:
            mi = onnx.shape_inference.infer_shapes(m)
            new_ranks = {}
            for v in list(mi.graph.value_info) + list(mi.graph.input) + list(mi.graph.output):
                new_ranks[v.name] = len(v.type.tensor_type.shape.dim)
            for n in m.graph.node:
                if n.op_type != "Gather":
                    continue
                t, remapped = n.input[0], False
                for _ in range(8):
                    p = next((x for x in m.graph.node if t in x.output), None)
                    if p is None:
                        break
                    if p.op_type == "Shape":
                        r0, r1 = orig_ranks.get(p.input[0]), new_ranks.get(p.input[0])
                        remapped = r0 == 3 and r1 == 4
                        break
                    t = p.input[0]
                idx_src = n.input[1]
                setidx = None
                if idx_src in init:
                    setidx = lambda arr: init[idx_src].CopyFrom(NH.from_array(arr, idx_src))
                    cur = orig_i = NH.to_array(init[idx_src]).ravel().tolist()
                else:
                    cp = next((x for x in m.graph.node if idx_src in x.output), None)
                    if cp is None or cp.op_type != "Constant":
                        continue
                    def setidx(arr): cp.attribute[0].t.CopyFrom(NH.from_array(arr, idx_src))
                    cur = NH.to_array(cp.attribute[0].t).ravel().tolist()
                if remapped and cur == [2]:
                    # axis 2 held the position axis; channels (1) and batch (0) are
                    # unchanged, -1 (last) is still the position axis.
                    setidx(np.array([3], dtype=NH.to_array(init[idx_src]).dtype if idx_src in init else np.int64))
                    print(f"remapped Shape-derived Gather index 2 -> 3 at {n.name}")
        except Exception as e:
            onnx.checker.check_model(m)
            onnx.save(m, dst)
            raise SystemExit(f"gather remap failed: {e}")
    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"promoted {promoted} block convs ({m.producer_name if hasattr(m, 'producer_name') else ''})")
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
    ap.add_argument("--calib", help="dir with x_*.raw / w_*.raw calibration batches")
    ap.add_argument("--out", required=True, help="output int8 onnx path")
    ap.add_argument("--fp32-blocks", type=int, default=2,
                    help="leading inverted-residual blocks kept FP32 (default 2; 0 = quantize all)")
    ap.add_argument("--no-promote", action="store_true", help="skip the Conv1d->Conv2d NCHWc promotion")
    ap.add_argument("--no-fuse-head", action="store_true", help="skip the softmax-max confidence-head fusion")
    ap.add_argument("--promote-only", action="store_true",
                    help="MODEL is an already-quantized int8 rec; only run the Conv1d->Conv2d "
                         "promotion on it (no re-quantization; every Q/DQ pair is preserved)")
    ap.add_argument("--rec-height", type=int, default=32,
                    help="recognizer input height (32 band; 96 for the windowed base rec)")
    args = ap.parse_args()

    if args.promote_only:
        promote_int8(args.model, args.out)
        print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")
        return

    samples = load_calib(args.calib, args.rec_height) if args.calib else []
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
