# Optimizations log

Every performance change attempted on this port, with the evidence that
accepted or rejected it. The point is to avoid re-litigating settled questions
and to keep the *method* honest.

- **Method, guard, gauntlet, A/B procedure:** see
  [`.agents/skills/corpus-eval/SKILL.md`](../.agents/skills/corpus-eval/SKILL.md).
- **Reproduce the shipped models:** see [`tools/README.md`](../tools/README.md).
- **Honest benchmarking:** one model/config per process, warm up, best-of-N;
  report **time and peak RSS**, measured at the **deployment** thread count
  (pool = 1 thread/worker). Never trust a speedup measured at one thread count.

"pp" = percentage points (absolute recall fraction delta x 100). "relative" =
delta / baseline.

---

## Accepted (shipped)

### Recognizer: mixed-precision int8 — `v0.14.0`

- **Config** (`tools/quantize_rec.py`, byte-reproducible): QDQ, per-channel
  weights, `QUInt8` activations, `QInt8` weights, MinMax, **asymmetric**
  activations, quantize the **raw** graph (no `quant_pre_process`), keep the
  **stem + first two blocks FP32**.
- **Speed:** rec ~2.4x at 1 thread (t2 1.8x, t4 1.3x). Pool 17.8 -> 22.3 pages/s
  (+25%), peak RSS -13%.
- **Quality:** FUNSD 0.9134 vs 0.9150 (-0.16pp); 2/199 pages >5pp worse, 0 >10pp;
  gauntlet within 0.5pp.
- **Why it worked when the first attempt didn't:** the original rejection was
  measured at 4+ intra-op threads, where int8 has no edge (fp32 scales with
  threads, VNNI saturates at one). Three choices recovered accuracy: raw graph
  (`quant_pre_process` decomposes HardSwish and quantizes more tensors, worse on
  both axes), asymmetric activations (~+0.6pp vs symmetric), and FP32
  stem/blocks 0-1. Keeping more leading blocks FP32 trades speed for accuracy
  monotonically (stem+blocks0-3: 21.4 p/s, -0.05pp; stem only: 23.2 p/s, -0.41pp
  but -2/-2.5pp on degraded/photo).

### Recognizer: fused softmax-max confidence head — `v0.19.0`

- `conf` (probability of the argmax class) was `ReduceMax + Sub + Exp + ReduceSum
  + Reciprocal + Mul` = 16 nodes incl. their Q/DQ wrappers, i.e. several
  elementwise passes over the `[B,T,V]` logits. ORT's `Softmax` kernel does the
  same math in one fused pass and `ReduceMax` picks the max probability.
  `tools/quantize_rec.py` (`fuse_conf_head`, on by default; `--no-fuse-head` to
  skip) rewrites the branch to `Softmax -> ReduceMax`.
- **Numerically:** `ids` and `lengths` are on a different branch and stay
  bit-identical (0 differing CTC lines on 60 real crops); `conf` differs only by
  accumulation order (~2e-3).
- **Speed:** rec ~7% (60 real crops, 1 thread); end-to-end guard `avg_ms`
  411 -> 395 (-4%), reproducible across runs.
- **Quality:** FUNSD 0.9137 -> 0.9136 (-0.01pp, one word via a MinConf flip);
  brutal-print gauntlet identical on all 9 subsets.

### ORT session config: CPU memory arena OFF (restored) — `v0.20.0`

- `newOptions` now sets `SetCpuMemArena(false)`, matching upstream
  `indra_ocr` (`pipeline.py`: *"No CPU memory arena ... it kept growing ...
  bought no measurable speed"*).
- **It had silently regressed:** `daf601e` set it `false`, then `6ac5c04`
  (the rec 512px batches) flipped it back to `true` **without saying so**, while
  leaving the "Arena off" comment. The arena choice was never in this log.
- **Measured** (FUNSD corpus, 8 workers x 1 thread, deployment config, no result
  retention): arena ON 22.95 pages/s / 2.88 GB peak RSS (3.32 GB at 10 reps);
  OFF 22.12 / 1.96 GB (2.17 GB at 10 reps). ON is ~3-4% faster but RSS rises
  ~245 -> ~360 MB per handle; `PerHandleBytes` is calibrated for arena off.
- Verdict: keep OFF (upstream parity + memory budget + this project's stated
  memory priority); the ~3% is the price, now documented instead of accidental.

### Pool sizing re-validated (arena context)

- Grid (arena OFF, FUNSD, 597 pages, pages/s): 4w x 1t 18.8, **8w x 1t 23.0**,
  16w x 1t 23.3, 8w x 2t 18.9, 4w x 2t 17.6, 16w x 2t 17.8. Confirms
  `AutoWorkers = NumCPU/2` x 1 thread: the knee, and intra-op threads lose.
- Single-page latency does **not** scale with intra-op threads (139/118/115/127 ms
  at 1/2/4/8 threads), so upstream's "few pages -> all cores on each page"
  fast-path is not worth porting; the pool already spreads pages.
- **Per-stage split** (199 pages, threads=2, serial): det 35%, rec 35%,
  upright 28% (of which skew 80% / orientation 20%), assemble 2%. det and skew
  are as expensive as rec — do not assume rec is the whole story.

### Recognizer: Conv1d -> Conv2d promotion (NCHWc) — `v0.16.0`, LOSSLESS

- ORT's layout optimizer only lowers **2-D** Convs to the fast **NCHWc** layout;
  **1-D** Convs are left on the slow generic path
  (microsoft/onnxruntime#16047, maintainer: *"just promote it to Conv2d"*). The
  whole rec backbone is `Conv1d`.
- `tools/quantize_rec.py` (`promote_conv2d`, on by default; `--no-promote` to
  skip) adds a unit spatial axis and keeps the graph 4-D (SE reduce axes ->
  (2,3), widths mask -> `[B,1,1,W]`, final Squeeze for CTC).
- Optimized graph: **~2 -> ~117 NCHWc ops**. Numerically identical: 0 CTC lines
  differ, corpus 0.9134 -> 0.9136, gauntlet within +/-0.02pp.
- Rec-only: fp32 1.22x, int8 1.08x. End-to-end pool 8x1t +~2.5%.
- Bonus: NCHWc makes fp32 convs **scale with threads** (fp32 rec t1 18548ms ->
  t4 8362ms), so at 4-8 threads fp32+NCHWc ~= int8 Conv1d.

### Skew: coarse pass at stride 4 — `v0.17.0`

- The 800px estimator is ~96% sharpness passes (coarse x21 = 64%, fine x11 =
  33%; the resize is ~3%). The coarse pass only locates the basin, so it runs at
  **stride 4** (2x cheaper than stride 2, which is argmax-identical to stride 1
  on 199/199 pages).
- Single-handle 10.4 -> 12.0 pages/s (1.14x; skew is on the single-handle
  critical path). Skew CPU work -31% (portable).
- Divergence: 1/199 pages shifts final skew <=0.8deg (one deskew decision flip);
  corpus 0.9136 -> 0.9137, gauntlet within 0.02pp. Pool on the dev box unchanged
  (wall-bound).

### Pipeline overlap (stage pipelining) — accepted

- `OCR.CallAll` / `Pool.CallAll` run per-page stages (orient+det / rec / assemble)
  as a 3-stage goroutine pipeline, one page per stage, sessions partitioned by
  stage. Output bit-identical to per-page `Call` (DeepEqual, threads=1; `-race`
  clean).
- Single handle +1.36-1.43x pages/s. This is **stage** overlap, NOT parallel rec
  sessions (only one page is in rec at a time).

### Pool sizing defaults

- Default `NumCPU/2` workers x 1 thread (`AutoWorkers`, `NewPoolAuto`,
  `NewPoolBuilder`). Worker count dominates intra-op threads (threads=2 loses).
  Throughput plateaus from ~6-16 workers; half the cores is the knee with
  headroom for the pipeline's small stages. RAM scales ~linearly with workers
  (~120-260MB/handle + ~380MB floor) -> `MemBudget` / `PerHandleBytes`.

### Warp-math incremental walks

- `sharpRows` / `rotateRows` / `resizeLinearRows` advance coords by addition
  instead of recomputing per pixel. Tilt sweep 13/13 identical to 1e-4deg; rec
  texts bit-identical on testdata; corpus neutral. skew -16%, rotate -21%,
  crops -10%. Safe pattern for future warp-math work (needs the same A/B proof).

### Order: `tab` head stripped + dynamic int8 — `v0.18.0`

- **What:** `tools/quantize_order.py` (a) physically removes the dead `tab`
  head (38 tab-only nodes + 4 initializers; the port only reads a/cls/same,
  tables render geometrically), then (b) dynamic-quantizes the MatMul/Gemm
  weights to per-channel QUInt8.
- **Size:** `models/order.onnx` 2.97 MB -> 0.76 MB (-74%), ~-2.2 MB of embedded
  model bytes. This is the largest model but runs once per page, so it is a
  size/startup win, not a throughput one.
- **Speed:** order stage 2.16 -> 1.92 ms (N=140 lines, 1 thread). Session
  create for the 4-model set drops ~10 ms from the smaller node count.
- **Quality:** FUNSD guard 0.9137 -> 0.9137 (identical); brutal-print gauntlet
  identical on all 9 subsets (cord 0.7178, sroie 0.8456, xfund 0.9516, ...).
- **Reading order:** the guard is bag-of-words and blind to sequence, so a
  Kendall-tau metric against FUNSD's ordered GT words was added (see the
  corpus-eval skill). Quantization perturbs adjacency near threshold: 51/199
  FUNSD pages re-sequence (median 5 lines, pure permutation). The metric shows
  it is quality-neutral: mean tau +0.0007 on FUNSD, 0 pages worse >2pp;
  cord/xfund/alpha identical; sroie -0.0057 mean, 2/29 pages >2pp.
- `--lossless` strips `tab` only (bit-identical outputs) if the quantization is
  ever unwanted.

### Order `tab` head pruned at fetch (superseded)

- The earlier hygiene change only skipped fetching `tab`; the nodes still lived
  in the file and ORT still built them. Superseded by the physical strip above.

---

## Rejected (do not re-run without new data)

### Detector resolution below 1024

- 768: shifts output on 178/199 pages, degrades 10 pages 5-15pp (Latin forms gain
  avg +1.7pp, CJK/small-print loses).
- 896: FUNSD neutral (0.9137), det 1.32x, **but** gauntlet cord (receipts)
  **-3.7pp / -5.1% relative** (xfund -0.45pp; sroie +4.0 / alpha-hi +4.6). A
  real regression on small print. Raising `adaptiveMinH` did not rescue cord.
  Net: +4.5% end-to-end for a receipts regression — not worth it.
- Det is already 2-D and NCHWc'd; keep 1024.

### Detector int8

- **Every** int8 form (QDQ and QOperator, per-channel and per-tensor, U8U8,
  S8S8) yields **0 NCHWc ops** for det -> 46 ms/page vs 35 ms fp32. The NCHWc
  transformer does not lower quantized convs, and the int8 path without NCHWc is
  slower. Dead.

### Alternate CPU execution providers (all three fail)

- **DNNL EP**: slower than MLAS. Allocates 16 OpenMP threads regardless of ORT's
  thread setting; even constrained, 87-132 ms vs 53 ms on det. Known poor ORT
  integration (microsoft/onnxruntime#5615).
- **OpenVINO EP**: huge *single-session* speedups (det 9.2 ms vs 32 at 1 thread)
  are a **thread artifact** — OpenVINO ignores `intra_op` and used 16 threads.
  At 1 thread it equals MLAS (32.2 vs 31.7), and it **collapses in the pool**
  (~4 p/s vs 25) in every worker x `num_of_threads` config. It parallelizes one
  inference internally; our throughput is many concurrent single-thread workers.
  (It is numerically lossless: corpus 0.9150.)
- **XNNPACK EP**: wash (det 1.05x, rec 1.0x). Unsupported ops (SE/CTC/Resize)
  fall back to CPU and the layout churn cancels the conv win.

### FP8 / FP16 / BF16 on CPU

- ORT CPU has no fp8 kernels and no fp16 ops; BF16 Conv is unsupported, and
  Quark BF16 is GPU/NPU-only. Zen4 has no fp8 hardware (avx512_vnni only). Dead.

### Rec: AdaRound / activation MSE-clip

- Layer-wise AdaRound (ONNX + numpy, output-Gram quadratic, sequential binary
  coordinate descent from round-to-nearest) cut per-layer output MSE 63% (14% of
  weights re-rounded) but moved FUNSD recall only +0.04pp (b1) / +0.09pp (full
  int8); line-exact/char-edit flat. MSE-optimal activation ranges beat ORT MinMax
  on **0/127** tensors (ranges already tight). The residual int8 loss is
  intrinsic to 8-bit **activation** quant, not weight rounding. Untried lever if
  ever needed: quantized-input reconstruction (`||conv(W_hat,Xq)-conv(W,X)||^2`).

### Rec: full int8 (drop the FP32 stem/blocks 0-1)

- +4.5% pool but -0.25pp FUNSD, -2pp on degraded/photo. A quality trade, not
  free. Not taken.

### Rec: remove the SE (squeeze-excite) blocks

- SE is ~16% of rec, but bypassing the SE scale **destroys** the model: corpus
  recall 0.9137 -> **0.0187**. SE is load-bearing. Dead.

### Rec: batching retune

- Bigger batches (fewer ORT runs) are **slower** — rec is pixel-compute-bound,
  not launch-bound; padding dominates. The current `WidthBatches` (ratio 1.3,
  area 40960) is near-optimal; tighter buckets (r1.15) only ~5% rec better.
  Leave it.

### Rec: shape-static model / parallel rec sessions / int8-only-Conv

- Shape-static (baked fixed shapes / free-dim overrides): 0.96-1.02x on real
  shapes. Parallel rec sessions: rec saturates cores at 8 threads; 16-thread
  single-session collapses 248 -> 681 ms. Conv-only int8 (leave elementwise
  fp32): slower (10.3 vs 9.7 s).

### Skew / misc micro-optimizations

- Skew half-size coarse: 1.84x skew but FUNSD -0.10pp / photo -0.53pp, and no
  pool gain. Superseded by the accepted stride-4 coarse (which is near-lossless).
- Bit-identical scalar rewrite of `sharpRowsStride` (hoist `cos*stride`, replace
  `math.Floor` with exact truncate+correct): verified identical by fuzz, but
  **0.995x** — the loop is memory/branch-bound, not arithmetic.
- ORT `allow_spinning=0`: nil. Per-model threads `det=2`: worse (18.4 vs 20.0).
- Serial pool vs pipelined pool: same throughput (~20 p/s); the pipeline burns
  ~30% more CPU (matters only where CPU is the binding constraint).
- 1x1 Conv as MatMul: 1x1 Conv is **faster** (0.87 vs 0.93 ms / 15 layers).
  Don't rewrite pointwise convs as MatMul.

---

### Pre-baked (ORT-optimized) graphs / `.ort` format

- Pre-baking ORT's in-memory optimized graph into the embedded files (constant
  folding + NCHWc layout + QDQ->QLinear + GELU/LayerNorm fusions) cuts session
  create for the 4-model set 70.8 -> 44.7 ms (keeping level=ALL) or 37 ms
  (disable), with identical corpus recall. **Not shipped:** the files are
  slightly *larger* (det 390 -> 402 KB), keep-ALL re-derives everything at load
  anyway, and a baked graph version-locks the NCHWc layout to the bundled ORT.
  The order size win above already captures the useful part.
- `.ort` flatbuffer/`.optimized.onnx`: larger than ONNX here (det 390 -> 459 KB).
  No size win; not shipped.

### Re-audit: the shipped models are already maximally fused at load

Done while hunting for further surgery (do not redo):

- **det / orient:** ORT's NCHWc transformer absorbs `Conv+HardSwish` into the
  conv activation (31 / 21 HardSwish -> 0), plus conv+BN / conv+Add folding.
  Node counts 164 -> 96 and 55 -> 30.
- **rec int8:** QDQ -> `QLinearConv`/`QLinearMul`/`QLinearAdd`/`QLinearSigmoid`;
  BN folded; `se/fc1` Relu fused into its conv. The only unfused activation is
  HardSwish in the int8 region (no `QLinearHardSwish` exists) -> a
  dequant+HardSigmoid+Mul+quant island, ~1% of rec.
- **order:** Erf-GELU -> `BiasGelu`, `LayerNorm` kept, `FusedMatMul`. The port
  never reads `tab`.
- **rec profile** (40 real crops, 1 thread): `QLinearConv` 8.6%, fp32 `Conv`
  5.1%, `DequantizeLinear` 2.9%, `QLinearMul` 2.8%. The CTC confidence head
  (ReduceMax/Sub/Exp/ReduceSum/Reciprocal/Mul, ~14% of rec) is already the
  minimal softmax decomposition -- `Softmax`+`ReduceMax` would do the same work.
- **SE `Div` is not foldable:** its divisor is the *runtime* valid-element count
  (`blocks.0/se/Clip(ReduceSum(mask))`), shared by every block, not a constant.

### Re-audit: rec `--fp32-blocks` did NOT move post-NCHWc

- Re-swept after the Conv1d->Conv2d promotion on the theory that cheaper fp32
  convs would shift the accuracy/speed knee. They did not.
- rec latency (2063 lines, 1 thread): blk0 1548, blk1 1562, **blk2 1641**,
  blk3 1725, blk4 1805 ms.
- FUNSD: blk0 0.9110, blk1 0.9119, **blk2 0.9137**, blk3 0.9136, blk4 0.9142.
- Gauntlet: blk3/blk4 are dominated -- slower and no consistent gain (blk4 is
  *worse*: cord -1.8pp, sroie -0.4pp). blk1 is 4.8% faster rec but -0.18pp FUNSD
  and mixed on the gauntlet (-0.4pp degraded/photo, +0.6pp cord).
- Verdict: keep `--fp32-blocks 2`. Reproduced the shipped rec exactly from the
  `cal` calibration dir (ids+conf bit-identical), so the comparison base is the
  shipped model.

---

## Corrections / stale notes

- **Skew estimator**: an older note said "600px + 3-level search". The shipped
  estimator is `EstimateSkewPy` (800px thumbnail, 0.5deg coarse then 0.1deg fine,
  black border). The 600px `EstimateSkew` exists but is unused.
- **`quant_pre_process`**: do NOT run it before quantizing rec — it decomposes
  HardSwish and quantizes more tensors, hurting both accuracy and speed.

---

## Throughput ceiling & measurement (dev box)

Dev box: Ryzen 7 PRO 7840U (8 physical / 16 logical, 28W laptop part). The pool
is hard-capped at ~20-25 p/s OCR-only / ~19 p/s end-to-end (`Pool.CallFiles`).

- **8 independent single-thread *processes* = the pool** (~20.6 p/s) -> the Go
  pool/goroutine model is NOT the limit.
- Pure integer CPU scales 8x, but **both Go and Python scale only ~3.2x** from
  1->8 cores: the workload saturates a shared resource (memory/power). Per-core
  work reduction does **not** raise pool throughput here.
- End-to-end at 8 cores: Go ~18.9 vs Python `convert_many` ~14.8 p/s = **1.26x**
  (~1.4x OCR-only). Single-handle wins (skew, pipeline) show here even when the
  pool does not.
- Consequence: prefer changes that reduce total work (they help unconstrained
  many-core hardware) and validate scaling on the target hardware, not this box.

---

## Go vs upstream: per-stage parity (threads=1)

Same corpus, Go `NewOCR(1)` serial vs original `indra_ocr.OrtOCR(threads=1)`
with `cv2.setNumThreads(1)`:

| stage | Go | upstream |
|---|---|---|
| orientation | 10.2 ms | 11.3 ms |
| skew | 28.4 ms | 24.1 ms |
| det | 63.0 ms | 66.3 ms |
| rec | 53.4 ms (int8) | 134.0 ms (fp32) |
| **per-page core** | **149 ms** | **270 ms** |

- No Go stage is meaningfully slower than upstream at the deployment thread
  count; rec is 2.5x faster (shipped int8 vs upstream fp32). The end-to-end
  1.26x (vs ~1.8x core-only) is upstream spending the difference on rendering,
  the text layer and layout, not on OCR.
- Compare at **threads=1**: at threads>=2 upstream's `cv2.warpAffine`
  parallelizes the skew search while the Go estimator is single-threaded on
  purpose (parallel angle eval was tried and rejected, -2.4%). Threads=2 makes
  Go skew look ~2x slower; at deployment it is ~1.2x.

## Reference numbers (dev box)

FUNSD 199-page guard (`TestCorpusRecall`, threads=2): shipped recall **0.9137**.

Brutal-print gauntlet (threads=2), current:

| subset | recall |
|---|---|
| alpha-bn | 1.0000 |
| alpha-el | 0.9286 |
| alpha-hi | 0.9302 |
| alpha-ru | 0.9400 |
| cord | 0.7178 |
| funsd-degraded | 0.8501 |
| funsd-photo | 0.8117 |
| sroie | 0.8456 |
| xfund | 0.9518 |
