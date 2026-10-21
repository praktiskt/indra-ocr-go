# Optimizations log

Every performance change attempted on this port, with the evidence that
accepted or rejected it. The point is to avoid re-litigating settled questions
and to keep the *method* honest.

- **Method, guard, gauntlet, A/B procedure:** see
  [`.agents/skills/quality-eval/SKILL.md`](../.agents/skills/quality-eval/SKILL.md).
- **Reproduce the shipped models:** see [`tools/README.md`](../tools/README.md).
- **Honest benchmarking:** one model/config per process, warm up, best-of-N;
  report **time and peak RSS**, measured at the **deployment** thread count
  (pool = 1 thread/worker). Never trust a speedup measured at one thread count.

"pp" = percentage points (absolute recall fraction delta x 100). "relative" =
delta / baseline.

---

## Accepted (shipped)

### Language-model beam search and line decoding: three bugs that kept the LM off — `v0.31.0`

- **Found by A/B-ing against the Python reference** (upstream `indra-ocr` 1.1.2
  run on the same pages with the same FP32 models), not by the corpus: the port
  produced *plausible* output, so the guards could not see it.
- **Bug 1 — the beam search read log-probabilities as probabilities.** The
  model's `topv` output is `LogSoftmax -> TopK`, i.e. **log**-probabilities
  (upstream's `beam_search` docstring says so). The port compared them against
  `candFloor` (a probability, 3e-3) and then took `log()` of them again, so
  every candidate was either dropped or NaN. Every line the LM touched came
  back **empty** — the language model has been effectively off since the port.
  Fixed: compare in log space (`v >= log(candFloor)`) and use the value as-is.
- **Bug 2 — the text came from the greedy CTC ids, the tokens from the beam.**
  Upstream reassigns `ids` to the beam-search result, so the line text and the
  tokens are the same reading; the port kept the greedy text and the beam
  tokens. Fixed (`recognize.go`).
- **Bug 3 — an unsplit line kept the raw recognizer text.** Upstream always
  decodes the line through `_decode` (whitespace collapse, NFC, and the
  right-to-left flip for group tokens; the text-layer tokens get the collapse
  only). The port only decoded split lines. Fixed (`assemble.go`, `ocr.go`).
- **Effect (FUNSD 199-page guard, int8):** 0.8678 → **0.9318**. Gauntlet: cord
  0.6442 → 0.7301, sroie 0.8457 → 0.8743, xfund 0.9458 → 0.9634,
  funsd-degraded 0.6610 → 0.7400, funsd-photo 0.5874 → 0.6169. Per-script
  guard (character recall): world 0.8203 → 0.9246, cjk 0.8378 → 0.9276, rtl
  0.9641 (unchanged). The v1.1 numbers now sit **above** the v0.3.2 reference
  (FUNSD 0.9150, cord 0.7055, sroie 0.8309, xfund 0.9601).
- **Upstream A/B (14 pages, FP32 both sides):** word agreement 94.8% → **99.5%**
  (2248/2259 upstream words present in ours, 8 words only in ours), line-level
  LCS 79.7% → **93.1%**. The residual is the detector's input resize: our
  `INTER_AREA` port rounds differently from cv2's (max 1/255 per pixel, 34% of
  pixels differ), which flips a borderline pixel and occasionally a connected
  component. **A/B against upstream is the check the corpus cannot do** — it
  compares *order and wording*, not just word presence.
- **PDF text layer A/B** (`testdata/multipage12.pdf`, 12 pages, upstream
  `indra-ocr` vs the Go `CallPDF`, FP32 both sides): the page sources match
  12/12 (`text`/`hybrid`) and **38/40 lines are character-identical**. With
  the shipped int8 build only 11/40 match: on 5 pages int8 reads a tiny,
  otherwise empty box as a character ("D"), which turns a `text` page into a
  `hybrid` one. That is the int8 noise floor (the same conf-boundary effect
  the gauntlet sees on receipts), not a text-layer bug — with FP32 the pages
  match upstream exactly.
- **Omni Parse Bench, direct against upstream** (120 pages of
  `multiling_finepdfs`/`multilingual`/`realdoc_pages`, `--mode ocr`, the
  upstream runner `bench/opb_indra.py` with the same model dir and page
  selection): Go **0.8214** headline (text 0.6845, layout 0.9583) vs upstream
  **0.8165** (text 0.6817, layout 0.9514) — the port is +0.5pp and **3.3x
  faster** (215 ms vs 705 ms median per page, int8 vs upstream's FP32). The
  full 2937-page run is the real number and was not repeated here (it takes
  ~30 min); this subset is the fast check.
- **Right-to-left A/B** (49 pages: 8 real OPB Arabic/Hebrew pages + 40
  synthetic RTL pages, upstream vs Go, FP32 both sides): **99.9% token
  agreement** (7117/7124 upstream tokens present in ours) and 95% of the lines
  character-identical — the v1.1 `rtl.go` flip (visual↔logical, mirroring,
  presentation forms) matches upstream. The **vertical-column** path (orient
  class 4) could not be A/B'd: no vertical document exists in the test data
  and the orient model does not call a synthetic one vertical either.

### Upstream model upgrade to indra v1.1.0 — `v0.29.0`

- **What:** the port tracks upstream package `indra-ocr` **1.1.2** on the model
  set `datalab-to/indra-ocr` tag **v1.1.0** (1.1.1 and 1.1.2 are code-only
  releases: the visual-order RTL layer check and the legacy-font script check).
  `models/` is the v1.1.0 tag: `det.onnx` retrained for handwriting (same size,
  3 channels — picture path dormant upstream), a 4-class `script.onnx` router
  (`script_groups = [base, cjk, rtl, world]`), three new script-group
  recognizers (`rec_cjk`/`rec_rtl`/`rec_world`, `rec.onnx` retrained), a 5-class
  `orient.onnx` (class 4 = vertical columns), and three n-gram LMs (`lm.npz`
  order-5, `lm_cjk.npz` order-3, `lm_world.npz` order-4; no `lm_rtl`).
- **Pipeline ports (all in this commit range):**
  - `rec_window = 3`: the base recognizer reads a 96 px window (the line band
    plus a band above/below); group recognizers read the 32 px band. Light-on-dark
    is decided once per box (`lineDarkness`).
  - **script routing + LM beam search:** `script.onnx` picks a group per line
    (probability over `RouteSure = 0.95`); a line the router is unsure of is
    read by base *as well* and the more confident reading wins (base first, so it
    wins ties). Lines with `0.5 <= conf < gate` are re-read by the CTC prefix
    beam search over the LM (`gate` 0.9 base / 0.95 cjk+world, `alpha` 0.5 / 0.2,
    `beta` 1.5, beam 8).
  - **orientation/deskew up to 45°** (`upright.go`, replaces `skew.go`/
    `skew_py.go`): preview 1600, coarse ±45° step-2, then 0.5° and 0.1° refine;
    5-class orient (vertical → `rot90` CCW read as CJK columns). Vertical pages
    assemble via XY-cut blocks (the order model saw no vertical pages).
  - **`rtl.go`** (`rtl.py`): a bidi subset for Arabic/Hebrew — visual↔logical
    `flip`, mirroring, presentation-form and tatweel normalisation. `lm.go`
    (`lm.py`): npz/npy reader + `ngramLM` + the beam search; the three LMs are
    embedded as-is (~46 MB) and parsed lazily, one shared copy per process.
  - **assembly:** `tidy_lines` (leader/marker rules), page base direction from
    the strong-character counts, `split_line` at wide gaps, vertical `xyCut`.
  - **PDF text layer:** the whole v1.1 phase (quarter turns, `/Rotate`,
    vertical, visual-RTL, legacy-font check, sideways runs) — see
    `DEVIATIONS.md`.
- **Quantization:** base `rec.onnx` mixed-precision int8 (raw graph, QDQ,
  per-channel `QUInt8`/`QInt8`, MinMax, asymmetric, stem+blocks 0-1 FP32) **and
  `order.onnx` dynamic int8 + `tab`-strip**. The new `/rec/context` module breaks
  the Conv1d→Conv2d NCHWc promotion (`tools/quantize_rec.py --no-promote`), so
  the base build is the raw graph; it is still faster than FP32.
  The three **script-group** recognizers (`rec_cjk`/`rec_rtl`/`rec_world`) are
  int8 too, but each needs **its own script's calibration set** (see below).
- **Quantization A/B (per page, the rule from the skill):** base `rec.onnx` int8
  vs FP32 on the 199-page FUNSD corpus: **0.9341 → 0.9318** (−0.23pp) and
  **1 page** degraded >5pp (out of 199) — inside the reject rule (drop >0.01
  or >10 pages >5pp). Per-script guard (int8 vs FP32 group recognizers):
  rtl 0.9644 → 0.9641, world 0.9243 → 0.9246, cjk 0.9311 → 0.9276, **0 pages**
  degraded >5pp in any of the three. The gauntlet's sroie/cord subsets move by
  ~1.7pp between FP32 and int8, but they hold 175/163 words — three words —
  so the per-page A/B on the 199-page corpus is the number that decides.
- **Speed and memory** (deployment config: pool = 8 workers x 1 intra-op
  thread, 199-page FUNSD corpus, 16-core box, idle, best of 3 runs x 2 process
  runs): FP32 `rec`+`order`+groups **13.9 pages/s, peak RSS 3.26 GB** →
  int8 **16.6 pages/s, peak RSS 3.07 GB** (**+19% pages/s, −6% RSS**; the
  embedded models shrink by 18 MB). Smaller than the v0.3.2 recipe's +25%,
  because the v1.1 base rec gained the `/rec/context` module and cannot use
  the NCHWc promotion that made the old graph fast.
- **Evidence (this machine, FUNSD 199-page guard, threads=2):** FP32 v1.1.0
  `rec` 0.8658 and int8 `rec` 0.8680 when the upgrade landed (the LM was
  broken then — see `v0.31.0`); with the LM fixed the shipped int8 build
  scores **0.9318**. Reading order (order.onnx int8 vs FP32, same rec/det):
  **144/199 pages identical**, mean Kendall tau **0.9939**, 2 pages < 0.9.
- **Group recognizers:** the first int8 attempt failed because each group model
  was calibrated on the wrong script; the shipped builds are per-script
  calibrated — see the entry below.
- **Bug fix (found by the guard):** `assemble` passed `tidy_lines` the
  *pre-split* boxes with the *post-split* recs (panic on any page whose line
  split at a wide gap). Fixed to pass the split boxes.
- **Bug fix (found by a unit test):** the npz reader assumed fixed-width central
  directory entries, so every LM failed to load and `lmFor` returned `nil` —
  the beam search was silently off. Entries are now walked with their
  name/extra/comment lengths; all three LMs load.
- **Gauntlet (recreated from the spec; scratch, `/tmp/opencode/hard`, 150
  pages):** int8 vs FP32 v1.1 is neutral — sroie 0.8514/0.8229, cord
  0.6687/0.6871, xfund 0.9072/0.9012 (micro word recall, threads=2). The
  synthetic `funsd-degraded`/`funsd-photo` generators were rebuilt differently
  from the originals, so their absolute scores (0.67/0.57) are **not**
  comparable to the historical 0.78/0.76; only the int8-vs-FP32 differential is
  meaningful there (neutral). Those numbers predate the LM fix (`v0.31.0`),
  which lifted every subset: sroie 0.8743, cord 0.7301, xfund 0.9634,
  funsd-degraded 0.7400, funsd-photo 0.6169.
- **Orientation bar re-measured on the v1.1 `orient.onnx` (kept at 0.98):** the
  new model is *less* discriminative on receipts than the v0.3.2 one — on 80
  cord photos (20 receipts x 4 photo variants) it calls 52 of them "upside
  down", and the false-positive confidence reaches **0.995** (the old model
  topped out at 0.954). At our 0.98 bar that destroys 3/80 receipt pages and
  fixes 3/80 genuinely inverted ones (net 0); upstream's 0.95 would destroy 8
  and fix 11 (net +3 on receipts) but recover 4.8pp more on *synthetic*
  upside-down documents (FUNSD rotated 180°: 0.7646 at 0.98 vs 0.8125 at
  0.95). Real rotated pages (OPB `rotation_skew`, 60 pages) are unaffected
  (0.8974 at both bars), and upright pages are unaffected (0.8680 at both).
  Kept at 0.98: on **real** data the bar is neutral-to-better, the upside-down
  gain comes from a synthetic set, and the bar destroys fewer pages outright.

### Script-group recognizers (`rec_cjk`/`rec_rtl`/`rec_world`): int8 with per-script calibration — `v0.29.0`

- **What:** the three group recognizers ship int8 (same recipe as the base rec:
  raw graph, QDQ, per-channel `QUInt8`/`QInt8`, MinMax, asymmetric,
  stem+blocks 0-1 FP32, `--no-promote`, `--rec-height 32`). Size: 10.18 + 4.81
  + 5.63 MB → 3.62 + 2.11 + 2.34 MB (**−12.6 MB embedded**).
- **The calibration set must be the script the model reads.** A group model
  calibrated on the wrong script is not merely inaccurate, it *confabulates*:
  `rec_cjk` calibrated on Latin (FUNSD) crops reads CJK as garbage
  (`chinese_essay` character similarity 0.62 vs FP32, lines like
  `ő束了，z于ầ束了`). With a CJK calibration set the same build is identical to
  FP32 on that page.
- **Which script belongs to which model** (from the v1.1.0 `meta.json` vocabs):
  `rec_rtl` = Arabic + Hebrew; `rec_world` = Devanagari, Bengali, Armenian,
  Georgian, Ethiopic; `rec_cjk` = Han + Kana; **Greek and Cyrillic are the
  base model's**, not `rec_world`'s. Calibrating `rec_world` on Greek/Cyrillic
  pages (they read fine, but the model never sees them) halves the useful
  calibration data — that mistake cost 5pp of character overlap in the first
  attempt.
- **Guard built for it** (throwaway, `INDRA_SCRIPT_DIR`): synthetic pages per
  script, rendered with the system Noto fonts (raqm shaping) from real OPB
  multilingual text, with **exact ground truth**. Character recall
  (int8 / FP32): rtl **0.9641 / 0.9641**, world **0.9246 / 0.9246**,
  cjk **0.9276 / 0.9276** (word recall: rtl 0.9372, world 0.8702,
  cjk 0.4583 — the CJK word metric is whole-token and therefore harsh).
  On 35 *real* OPB multilingual pages the int8 output keeps **99.2%** of the
  FP32 characters (worst page 97.2%); the clean real CJK page
  (`chinese_essay`) is character-identical.
- **Tripwires:** FUNSD guard 0.8680 → 0.8678 (Latin path untouched); gauntlet
  cord 0.6687 → 0.6626, sroie 0.8514 (unchanged), xfund 0.9072 (unchanged),
  funsd-degraded 0.6661 → 0.6686, funsd-photo 0.5624 → 0.5610 — all noise
  (numbers from before the LM fix, `v0.31.0`). Re-measured with the LM
  working: character recall int8/FP32 rtl 0.9641/0.9644, world 0.9246/0.9243,
  cjk 0.9276/0.9311, with **0 of 180 pages degraded >5pp** — accept.
- `--fp32-blocks 4` was tried for `rec_cjk` (the worst case) and did not help
  (character overlap 0.9815 → 0.9821 for +0.55 MB), so the recipe stays at 2.

### PDFium concurrency: one library instance per concurrent document — `v0.28.1`

- **Bug:** PDFium's C API is not thread-safe — the API header allows only one
  call in flight against a loaded copy. Every entry point went through a single
  process-wide library (`pdfium_initlib` called plain `FPDF_InitLibrary`), so
  concurrent `CallPDF` (`Pool.CallPDF` from several goroutines, `OCR.CallPDF`
  from several goroutines) raced PDFium's globals and corrupted its heap
  (`benchthr -mode pdf` drives 16 concurrent `CallPDF`).
- **Fix:** PDFium's globals live in the *loaded library image*, so the port now
  opens N independent copies — a separate memfd and `dlopen(..., RTLD_LOCAL)`
  each, with its own resolved entry-point table — and lends one per concurrent
  document (`pdfiumPool`, cap `min(NumCPU, 8)`, opened lazily, so a
  single-threaded caller still opens exactly one). Distinct instances render in
  parallel; nothing is serialized. This is **not** an init flag:
  `FPDF_LIBRARY_CONFIG` (v1–v3) has no thread-safety field — upstream's own
  guidance is a mutex or separate processes.
- **Evidence (dev box, 16 cores):** a scratch probe driving **one** instance from
  8 goroutines segfaults PDFium; with the pool, 8 goroutines render
  bit-identical to the sequential baseline. Committed `TestPDFiumConcurrentRender`
  (8 goroutines x 2 rounds, hash of every page's pixels + runs) catches a
  regression to a shared instance. 8 instances = 32 distinct libpdfium mapped
  ranges, +~6 MB RSS each (touched) — negligible against ~120–260 MB/handle.
  Concurrent `CallPDF` (8 goroutines x 2 pages): 47.7 -> 16.2 ms/page (2.95x);
  0/80 divergent `CallPDF` runs.
- **Quality:** scheduling/resources only — rendering itself is untouched, and
  single-threaded output is bit-identical (scheduling-only change, waives the
  gauntlet per the skill). The **OPB runner is sequential** (one page at a time,
  `bench/opb`), so the official benchmark takes one uncontended instance: no
  timing or score change. Image/OCR paths are untouched.
- **Why no big lock:** a single global mutex would make every concurrent PDF
  workload single-threaded in its render+text stage and cap throughput at one
  renderer; per-instance copies keep the parallel path parallel.

### PDF text layer: `mode="text"` + the unreadable-script rescue — `v0.27.0`

- **What:** `PDFOptions.Mode` gains `"text"` (use the layer even when the trust
  checks fail) and, in `"auto"`, an untrusted layer is now used when the
  recognizer cannot read its script (`readable()` false, i.e. < 90% of the
  layer's letters are known to `vocab`/`vocab_cjk`) and its font map is not
  broken — upstream `document.pdf_page`'s `unreadable` rule. A layer the
  recognizer cannot read beats confabulating OCR; a broken font map does not.
  Wired in `CallPDF` (`useTextLayer`); `CallText` keeps its trust gate. The
  image/OCR path is untouched.
- **Why it matters:** on the full OPB it fires on **3/2937** pages, all garbled
  by OCR before: a Gurmukhi (`pan_Guru`) sheet and two Japanese NTA forms now
  read correctly from the layer.
- **Quality** (full OPB, `-pdf -furniture`, 8x1t, deterministic; base = shipped):
  headline 0.6052 -> **0.6054** (+0.02pp), text 0.4012 -> 0.4022 (+0.09pp),
  tables flat, layout -0.04pp; `multilingual` 0.5109 -> **0.5543 (+4.34pp)**,
  `multiling_finepdfs` 0.5740 -> 0.5784 (+0.44pp). 3 pages change, none
  degrades > 5pp. No speed change.
- **`mode="text"` is a capability add** (opt-in), not exercised by the OPB run.
- Unit test `TestUseTextLayer` covers the decision (force / trusted / unreadable
  rescue / readable fallback / broken-font-map exclusion).
- **Not ported (still):** the broken-font-map *turn* nuance — when an untrusted
  layer's map is broken and the glyphs are mostly upright, upstream skips
  orientation. Go always runs `Upright`. A preprocessing quality mechanism
  (orient false positives), so it needs the guard + cord-photo + gauntlet, not
  just the OPB; left for its own change.

### Math decode loop: reuse constant-shape tensors — `v0.26.0`

- `math.go` `decodeBatch` allocated and destroyed the `token`/`pos`/`logits`
  tensors **every decode step** (up to `max_tokens-1` = 319 steps per line
  batch). `token` and `logits` have constant shape across steps, so they are now
  allocated once and reused (onnxruntime_go tensors wrap the Go slice, so writing
  the slice updates the tensor). `pos` stays per-step: it is a true **0-dim
  scalar** input, and only `ort.NewScalar` builds one (a rank-1 `[1]` tensor
  mis-shapes the decoder and `Run` errors, silently dropping all math — caught by
  the math suite, since the pipeline then falls back to plain rec text).
- **Bit-identical:** OPB `olmo_arxiv_math` (514 pages) markdown+blocks **0/514
  differ** pre/post. (The math path is run-to-run deterministic; the only
  unstable field in the JSONL is the reported per-page `ms`, so compare
  markdown+blocks, not bytes.) Math is opt-in, so the default path is untouched.
- Not separately timed (opt-in path; this is a quality-first batch), but it is a
  pure allocation-churn reduction in the decoder's inner loop.
- Considered and left alone: `newMathReader`'s hardcoded `newOptions(2)` for all
  four math models ignores the pool's `Threads` (oversubscribes at `Threads(1)`),
  but changing intra-op thread counts can perturb ORT accumulation order and thus
  the autoregressive argmax, so it is not quality-neutral and needs its own A/B.

### Math encoder (`math_enc`): dynamic int8 — `v0.26.0`

- Same recipe as `math_dec_step` (`tools/quantize_math.py`, dynamic per-channel
  `QUInt8` MatMul/Gemm, no calibration): the 12 attention/FFN `MatMul` weights
  quantize; the 34 `Conv` stay FP32 (that is most of the file). 9.89 -> 7.29 MB
  (-26%).
- Measured **stacked on the shipped `math_dec_step` int8** (both int8):
  OPB `olmo_arxiv_math` (514 pages) text 0.61484 -> 0.61413 (-0.07pp), headline
  0.78932 -> 0.78784 (-0.15pp; `math_dec_step` alone is -0.02pp, so `math_enc`'s
  marginal is ~-0.13pp). 260/514 pages differ; 36 pages > 5% char change
  (worst 0.25) — 8-bit on the encoder's attention perturbs some math spans that
  the decoder cross-attends to. All deltas are inside the project's accept bar
  (< 1pp), but this is the least lossless math model; a calibrated static-int8
  build (which could protect the attention `out_proj`) is the untried lever.
- **`math_dec_init` stays FP32:** its 6 `MatMul` read
  `memory x transpose(sliced in_proj_weight)` — the weight is not a direct
  initializer, so ORT's dynamic quantizer skips them (1.36 -> 1.37 MB, no-op).
  Quantizing it would need constant-folding the weight transposes first; it runs
  once per batch, so the payoff is ~1 MB — deferred.

### Math decoder step (`math_dec_step`): dynamic int8, logits head protected — `v0.26.0`

- `math_dec_step.onnx` is the hot math model: it is run **once per decoded
  token** (up to `max_tokens-1` = 319 serial ORT runs per line batch), 31
  `MatMul` (16 with constant weights), 10 `LayerNorm`.
- **Dynamic int8** (per-channel `QUInt8` on MatMul/Gemm weights, no calibration —
  the `order.onnx` recipe, `tools/quantize_math.py`): 8.43 -> 3.82 MB.
- **But quantizing the logits head flips tokens.** Full dynamic int8 moved OPB
  `olmo_arxiv_math` (514 pages) text 0.61484 -> 0.61272, headline 0.78932 ->
  0.78826 (-0.11pp): 193/514 pages differ, all near-identical (char sim > 0.97)
  — the final projection (`/out/MatMul`, 192 -> 652 classes) feeds an argmax, and
  8-bit rounding on the class logits crosses decision boundaries.
- **Excluding `/out/MatMul` recovers it:** text 0.61449, headline 0.78914
  (-0.02pp vs fp32), per_suite 0.66167 (fp32 0.66198). Size 4.19 MB (-50%).
- Shipped: `tools/quantize_math.py --model <fp32> --out models/math_dec_step.onnx
  --exclude /out/MatMul` (byte-reproducible).
- FUNSD/xfund not run here: the math models are off the default path (`OCR.Math`).

### CJK recognizer (`rec_cjk`): mixed-precision int8 — `v0.26.0`

- **The rec recipe applies verbatim.** `rec_cjk.onnx` is exported from the same
  graph as `rec.onnx`: identical node/tensor names (`/rec/stem`, `/rec/blocks.N`,
  the `conf` branch), identical `x`/`widths` -> `ids`/`conf`/`lengths` signature
  and rec height (32), 52 `Conv1d`. So `tools/quantize_rec.py` was run unchanged:
  QDQ, per-channel, asymmetric `QUInt8`, raw graph, stem+blocks 0-1 FP32, the
  Conv1d->Conv2d NCHWc promotion and the fused confidence head.
- **Calibration (the one thing that is new).** CJK line crops through the real
  pipeline: 60 synthetic CJK pages (Noto Sans/Serif CJK, zh+ja, mixed
  Latin/digits) plus the 10 real `xfund` zh/ja pages ->
  `INDRA_CORPUS_DIR=<cjk corpus> go run ./tools/calib -out /tmp/cjk_calib`
  (328 batches / 3658 lines). `tools/calib` is rec-height/`x`/`widths` already,
  so it needs no change.
- **Size:** `models/rec_cjk.onnx` 9.26 -> 3.36 MB (-64%, ~-5.9 MB embedded).
- **Quality** (OPB `multilingual` + `multiling_finepdfs`, 200 pages,
  `-furniture`): the `text`/`html` family is **byte-identical** (0.22037;
  `Hani` tag 0.2727 both) and `multilingual` per-suite is identical (0.3696);
  `multiling_finepdfs` 0.3861 -> 0.3891 (noise). Headline 0.4961 -> 0.5005
  (all of it `layout`, noise). The clean printed `cmn_Hani` page is
  character-identical.
- **Regression tripwires unchanged:** FUNSD guard 0.9137 -> 0.9134 and xfund
  0.9601 -> 0.9597 (both noise; `rec_cjk` is off the Latin path).
- Not swept: `--fp32-blocks` (kept at rec's `2`; quality was already neutral, so
  the sweep is a speed/size tune, not a quality one — the batch is quality-first).

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

### PDF input: embedded PDFium + auto text layer — `v0.24.0`

- `OCR.CallPDF` / `Pool.CallPDF` render PDF bytes and, in `mode="auto"`, use
  each page's text layer where sound (the `CallText` hybrid path) — upstream
  `document.py`. Ports `textlayer.py` extraction: per-char boxes mapped through
  the page-to-device affine, char angles (upright check), the invisible-text
  fraction (render mode 3), the broken-font / no-ink trust checks and
  `assign_runs`.
- The renderer **PDFium** (BSD-3-Clause, 7.8 MB) is embedded and loaded from
  memfd like ONNX Runtime; every entry point is resolved with `dlsym`, so the Go
  module gains no PDF dependency. `loadPDFium` is lazy.
- **Quality:** OPB `realdoc_pdfs` (88 pages, `auto`) headline **0.6964** (text
  0.643, layout 0.750) vs upstream **0.7121** (-1.6pp), deterministic. The
  default OCR path is unaffected.
- `mode="ocr"` ignores the text layer; `PDFOptions.DPI` defaults to 200.
- **`pdf_order` (content order) not shipped — measured neutral/negative:** the
  earlier port looked neutral-to-worse (ON 0.670 vs OFF 0.682) but that was
  measured while the buffer bug below was dropping/blanking pages, so it was not
  a result. Re-ported faithfully on the fixed build (deterministic text path):
  full OPB `-pdf -furniture` headline 0.6052 -> 0.6048 and `olmo_multi_column`
  0.8324 -> 0.8210 (-1.14pp). Rejected; see the Rejected section for numbers and
  the untried lever (Go does not split text-layer lines at wide gaps, so the
  boxes fed to `pdf_order` differ from upstream's).
- **Resolved (was "not bit-reproducible"): the buffer fix below explains it.** OPB
  `realdoc_pdfs` used to flip ~5/88 pages run to run; that was the GC freeing the
  PDF buffer mid-render, not the `CallText` assembly. Now 0/88 differ and the
  headline is stable.
- **Fix — `v0.24.1`: PDFium buffer lifetime.** `pdfOpen` handed
  `FPDF_LoadMemDocument64` a Go heap slice; pdfium does **not** copy it (the
  caller must keep it valid for the document's lifetime), so the GC could free
  it mid-render and `CallPDF` silently dropped/blanked pages (12-page PDF -> 3
  pages, 2 blank under `GOGC=1`; intermittent otherwise). The bytes are now
  copied into C memory (`C.CBytes`, `C.free`d by `pdfClose`), which also satisfies
  the cgo pointer rules. `TestCallPDFPages` (`testdata/multipage12.pdf`, forced
  GC) guards it. Audited the ORT paths: `CreateOrtTensorWithShape` and
  `CreateSessionFromArray` copy, and `Tensor` keeps its slice — no other
  retained-buffer sites.

### Math / LaTeX: embedded encoder-decoder — `v0.23.0`

- Ports upstream `math.py`: `mathcls.onnx` flags lines that hold math and a
  small encoder/decoder (`math_enc` + `math_dec_init` + `math_dec_step`, with a
  self/cross K/V cache stepped one token at a time) reads them as text with
  inline LaTeX. `OCR.Math` / `PoolBuilder.Math` enable it — off by default like
  upstream `math=True`; the ~20 MB of models are embedded but load lazily, so an
  unused handle pays nothing.
- Binding detail: the ORT Go binding rejects 0-dim tensors, so the scalar `pos`
  input is fed with `ort.NewScalar[int64]` (a rank-1 `[1]` tensor made the
  decoder mis-shape and error).
- **Quality:** OPB `olmo_arxiv_math` suite (514 pages) headline **0.7893** with
  math vs upstream `--math` **0.7899** (-0.06pp); *without* math both are
  exactly 0.5069. Median 921 ms vs upstream 4378 ms (~4.7x).
- Off the default path: only enabled with `Math=true`.

### Furniture + caller-supplied PDF text layer — `v0.21.0`

- `OCR.Furniture` / `PoolBuilder.Furniture(bool)` keep page headers/footers in
  the markdown (upstream `load(furniture=True)`); the default drops them. On OPB
  this is the whole `header_footer` gap (0.146 -> 0.589 vs upstream 0.606).
- `TextLayer` / `TextRun` / `TextChar` + `OCR.CallText` / `Pool.CallText` port
  `textlayer.py`'s trust checks (invisible layer, broken font map, runs with no
  ink, `readable`) and `document.assign_runs`: a caller that extracts a PDF's
  text runs gets the hybrid "auto" path (`Source` text/hybrid). Go carries no
  PDF engine here — the caller supplies the runs.
- `Page.Text()` (upstream `Page.text`).

### CJK routing: script.onnx -> rec_cjk.onnx — `v0.22.0`

- The port ran only the Latin/multi recognizer; CJK lines were confabulated
  (gauntlet failure class 1: zh/ja ~0.4-0.6). It now matches upstream
  `recognize`: `script.onnx` classifies each line's script (p > 0.5 -> Chinese)
  and those lines go to `rec_cjk.onnx`, decoded with the already-embedded
  `vocab_cjk`.
- `script.onnx` (0.19 MB) + `rec_cjk.onnx` (9.3 MB, fp32) are now embedded;
  upstream ships them but the port had omitted them.
- **Quality:** brutal-print gauntlet xfund 0.9516 -> **0.9601** (+0.85pp); FUNSD
  guard 0.9136 -> 0.9135 (noise) and every other subset unchanged (Latin, Greek,
  Cyrillic, Devanagari lines still route to rec).
- Not quantized yet (`rec_cjk` stays fp32); `rec_cjk` int8 needs its own CJK
  calibration.

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
  quality-eval skill). Quantization perturbs adjacency near threshold: 51/199
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

### Det input shape bucketing (coarser than /32) — no churn to fix

- **Hypothesis:** every page produces a distinct `det.onnx` input shape
  (`pageToInput` pads the resized page to a /32 multiple), and ORT builds a
  per-shape execution plan, so rounding the padded dims to a coarser grid
  (/64, /128) would reuse plans and cut det (35% of the pipeline) time.
- **Census kills it:** pages cluster into very few shapes. Over the whole
  FUNSD corpus (199 pages) there are **4** distinct det shapes at /32; over the
  full OPB (2937 pages) **34** at /32, collapsing to 19 (/64), 10 (/128),
  5 (/256). Each shape is reused ~86x on OPB, so the per-shape setup is
  amortized to nothing.
- **Direct timing (1 thread, det only):** repeating one warm shape vs
  alternating two warm shapes of equal area differs by **-0.4%** (noise). Cold
  (first-encounter) vs warm per shape over the 34 OPB shapes: **-3.7 ms/shape**
  (negative = thermal drift) — i.e. **no measurable per-shape plan-build cost**
  at all. ORT's plan/kernel selection for a new shape is cheap here.
- **Verdict:** nothing to gain; bucketing to /128 would change det input
  padding (a numeric change needing the guard + gauntlet) to save ~24 one-time
  plan builds over 2937 pages. Rejected. `pageToInput` stays at /32.

### Reuse output tensors across `Run` — bit-identical, but throughput/RSS neutral

- **What was tried:** every `runDet`/`recBatch`/`runScript`/`runOrder` did
  `ort.NewEmptyTensor` per call, i.e. a fresh `make([]T,n)` (zeroed) for the
  output, then GC. det output is ~3 MB/page (x2 with the adaptive pass), the
  padded rec batch up to ~5 MB. Held one reusable output tensor per
  (session, output), recreated only when the shape changes. ORT writes every
  element of a fully-defined output, so reuse is **bit-identical**.
- **Correctness:** `TestCallAllMatchesCall` passes, `-race` clean, FUNSD guard
  recall **0.9134** (== shipped). No reader touches `ids`/`conf` beyond
  `lengths`; `decodeDet` consumes the det buffer before the adaptive re-run;
  `runOrder` results are copied out by `reshape2`.
- **Speed (FUNSD, 199 pages, 8 workers x 1 thread, best-of-5):**
  - baseline: **18.98 pages/s**, peak RSS 2139 MB
  - reuse: **19.07 pages/s**, peak RSS 2176 MB
  - **+0.5% / +1.7% — both within run-to-run noise** (rep spread 18.2-19.1).
- **Why neutral:** this box is memory/power-saturated (see "Throughput
  ceiling"), and the saved allocation is recycled by the Go allocator, so the
  zeroing lands on cache-hot memory — no reduction in DRAM traffic. ~3-6 MB of
  alloc/zero per page at ~19 p/s is ~0.2% of memory bandwidth. The log's rule
  is "neutral + faster = accept"; this is neutral + neutral.
- **Verdict:** rejected (adds ~60 lines of reuse machinery for no measured
  gain). Untested lever if the target ever becomes a high-core-count or
  bandwidth-bound-elsewhere machine: re-measure there, where the binding
  resource differs.

### `pdf_order` (keep the PDF content order) — neutral, regresses multi-column

- **What was ported** (faithful to upstream `textlayer.order_backtracks` +
  `pdf_order_coherent` and `document.assign_runs`): the smallest run (content)
  index per detected line, the pairwise order logits `a`, and in `assemble` the
  content order kept when every line has a content index and
  `pdf_order_coherent` accepts it (`order_backtracks <= max(1, 0.02n)` and, on
  >= 3 boxes, the model agrees with >= `MIN_MODEL_AGREEMENT` = 0.85 of
  consecutive pairs). Wired only into the text-layer path (`CallText` /
  `CallPDF` `auto`); the image path is untouched.
- **Full OPB, 2937 pages, `-pdf -furniture`, 8x1t, deterministic:**
  - no `pdf_order`: headline **0.6052** (text 0.4012, tables 0.5364, layout 0.8781)
  - +`pdf_order`: headline **0.6048** (text 0.4008, tables 0.5356, layout 0.8781)
  - **-0.04pp headline, -0.07pp tables, layout flat** — neutral-to-slightly-worse.
- **582/2937 pages re-sequence** (tables_derived 291, olmo_arxiv_math 120,
  olmo_table_tests 45, synth_copies 41, olmo_multi_column 38, ...), but only 4
  suites move at all, and the material one is a **regression**:
  `olmo_multi_column` 0.8324 -> **0.8210 (-1.14pp)** on 137 pages (38
  re-sequenced), reproduced on an isolated subset run; `olmo_table_tests`
  -0.28pp, `tables_derived` -0.04pp. No speed change (order is once per page).
- **The old "residual is `pdf_order`" note was wrong.** On the suite it named
  (`realdoc_pdfs`) the text layer applies to ~1 page (upstream sources there: 83
  ocr / 4 hybrid / 1 text), and `pdf_order` differs on **0/88** pages. The
  `realdoc_pdfs` gap with furniture is 0.7076 vs upstream 0.7121 (0.45pp), not
  `pdf_order`.
- **Untried lever before any re-run:** Go's text-layer lines are **not** split at
  wide gaps (upstream splits them because its tokens carry per-char x), so the
  box set fed to `pdf_order` differs from upstream's. A like-for-like comparison
  needs that port first — that, not the coherence rule, is the likely reason the
  multi-column verdict differs from upstream's. Port the text-layer line split,
  then re-measure.
- Not shipped (implementation reverted).

### `script.onnx` / `mathcls.onnx` int8

- Both are 0.19 MB, 4-conv 2D CNNs run once per line batch (`script` routes CJK
  lines, `mathcls` flags math lines). Dynamic int8 (including the Convs) shrinks
  each to 0.06 MB (-0.13 MB, -0.26 MB total).
- **Slower AND flips decisions.** OPB `cjk36` median 215 -> 234 ms (+9%): ORT's
  NCHWc transformer does not lower quantized convs, so the tiny net drops to the
  slow generic conv path (the same effect that killed **Detector int8** above),
  and 8-bit rounding crosses the binary threshold on marginal inputs — 1/36 CJK
  pages rerouted (`script`; a Korean page the FP32 build already garbles) and
  6/48 math pages changed (`mathcls`).
- 0.26 MB saved for a measurable slowdown and routing sensitivity on the CJK/math
  paths — not worth it. (`script` and `mathcls` share the architecture.)

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
  estimator is `estimateSkewPy` (800px thumbnail, 0.5deg coarse then 0.1deg fine,
  black border). The 600px `estimateSkew` exists but is unused.
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

## Official benchmark: Omni Parse Bench (OPB)

- The upstream `datalab-to/omni_parse_bench` — 2937 pages, 22 suites (arxiv
  math, tables, forms, multi-column, multilingual, handwriting, rotation/skew,
  old scans, tiny text...) — is the distribution the models were tuned for.
  Method + how to run: see the quality-eval skill.
- **Full 2937-page OPB, current build (`-furniture`, default pool):**
  - **Go OCR (image input): headline 0.5784** (text 0.332, tables 0.523, layout
    0.880) vs upstream `--mode ocr` **0.5714** — **+0.7pp, the port beats the
    reference on OCR.**
  - **Go `auto` (PDF text layer via `CallPDF`): headline 0.6052** (text 0.4012,
    tables 0.5364, layout 0.8781, current build) vs upstream `auto` **0.6155** —
    ~1.0pp. The residual is **not** the unported `pdf_order`: porting it changes
    582 pages but is neutral on the headline (0.6052 -> 0.6048) and regresses
    `olmo_multi_column` (-1.14pp) — see the Rejected section. Likely lever: Go
    does not split text-layer lines at wide gaps (upstream does).
  - (The `0.6048`/`0.5784` figures below are from the pre-`rec_cjk`-int8 build;
    re-run for a same-build OCR number.)
  - The original Go default was **0.5687**; since then CJK routing (+0.52pp),
    furniture (~+0.4pp) and the PDF text layer (~+2.6pp) took it to 0.6048.
- `header_footer` opened the last OCR gap (0.146 -> 0.589 vs upstream 0.606);
  `OCR.Furniture` / `PoolBuilder.Furniture(true)` is the switch.
- Upstream's published numbers reproduce locally: `auto` 61.6 (published 62.1),
  OCR-only 57.1 (57.5).
- **Text layer (was a capability gap):** upstream `mode=auto` reads embedded
  PDF text layers (`source` = text/hybrid); the port now does too, via the
  embedded PDFium (`CallPDF` / `bench/opb -pdf`), reaching 0.6052 vs upstream
  `auto` 0.6155. Compare OCR-only with `--mode ocr`. `pdf_order` (keep the PDF
  content order) was ported and rejected (neutral headline, multi-column
  regression); the residual is elsewhere (text-layer line splitting).

## Reference numbers (dev box)

FUNSD 199-page guard (`TestCorpusRecall`, threads=2): shipped recall **0.9134**.

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
| xfund | 0.9601 |
