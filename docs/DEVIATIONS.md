# Deviations from upstream indra

The port aims to reproduce the Python reference (`indra_ocr`,
<https://github.com/datalab-to/indra>) on the same inputs. This file records the
**intentional** places where its output differs, so a diff against the reference
is not mistaken for a bug. Everything else is expected to match; the model-level
changes are in [`OPTIMIZATIONS_LOG.md`](OPTIMIZATIONS_LOG.md) and the evaluation
method in [`../.agents/skills/quality-eval/SKILL.md`](../.agents/skills/quality-eval/SKILL.md).

## Text layer: spacing diacritics are composed

- **What:** in the PDF text layer, a spacing accent glyph is folded into the
  precomposed form of the base letter it belongs to. The spacing forms are the
  TeX text-companion glyphs — grave `U+0060`/`U+02CB`, acute `U+00B4`/`U+02CA`,
  circumflex `U+02C6`, tilde `U+02DC`, macron `U+00AF`/`U+02C9`, breve `U+02D8`,
  dot above `U+02D9`, diaeresis `U+00A8`, ring above `U+02DA`, double acute
  `U+02DD`, caron `U+02C7`, cedilla `U+00B8` and ogonek `U+02DB` — composed
  through the full Unicode canonical table, so every base is handled:
  `Cicho´n` -> `Cichoń`, `dissertaç˜ao` -> `dissertação`, `potˆencia` ->
  `potência`, `Gr¨obner` -> `Gröbner`, `Huˇsková` -> `Hušková`,
  `Dvoˇr´ak` -> `Dvořák`, `¯Riga` -> `Rīga`, `˙e` -> `ė`, `˚A` -> `Å` (Angstrom),
  `˛a` -> `ą`, `ˆc` -> `ĉ` (Esperanto). Fonts that draw an accented i/j as a
  dotless glyph (`Mej´ıa` -> `Mejía`, `ˆȷ` -> `ĵ`) are covered too.
- **Both glyph orders.** Some producers draw the accent *after* its base
  (`Schu¨tz`, `Geocieˆncias`), some *before* (`Cicho´n`). The text layer carries a
  box per character, so the accent is joined to whichever neighbour its glyph box
  overlaps; with no clear overlap (or none) it defaults to the prefix case.
- **Not folded:** a mark whose neighbour has no precomposed form, a prime/trailing
  symbol that does not overlap the preceding letter (`SplExtp´,`, `5´`), and the
  ASCII operators `^`/`~` (never spacing marks). Multi-mark stacking
  (Vietnamese `ế`) composes when the marks are adjacent to their base.
- **Why:** PDFs draw the accent as a separate glyph positioned by its base, and
  pdfium emits characters in content order, so the raw layer reads
  `Cicho´n` / `Gr¨obner` / `Schu¨tz`. Upstream passes that through, but the
  recognizer does not — its vocab is NFC (`indra_ocr/vocab.py`), so the OCR path
  already produces the precomposed form. This makes the two paths agree.
- **Where:** `pdfium.go` (`foldSpacingAccents`); the spacing-mark set and the
  composition table live in `compose_table.go`, generated from Python's
  `unicodedata` by `tools/gen_compose.py`. Text-layer extraction only; the OCR
  path and image input are unchanged.
- **Impact:** Latin (and any canonically-composable) accents in PDF text-layer
  text are normalized; primes, unit symbols and ASCII math are not. Precomposed
  math decorations (`ḡ`, `ẋ`, `Ā`, `Å`) are a side effect of composing over a
  letter and read the same. Measured over all 2149 OPB PDFs: the edit composes
  29 macrons, 39 dots-above, 19 rings, 1 breve, 2 carons, 119 circumflexes, 96
  tildes, 157 acutes, 78 diaereses and 12 graves that previously leaked; the
  Portuguese prefix (`CallPDF` on the motivating thesis) is unchanged, and the
  postfix Brazilian/PMC papers (`Geociências`, `Schröder`,
  `Kansaneläkelaitos`) are fixed.
- **Validation:** `TestFoldSpacingAccents` (prefix, marks/bases, primes, ASCII)
  and `TestFoldSpacingAccentsOrder` (prefix vs postfix from boxes, stacking).
- **Version:** the composition landed in `v0.25.0` (acute / diaeresis / double
  acute / cedilla); extended to every mark and both glyph orders in `v0.25.1`.

## `pdf_order` (PDF content order) not ported

- Upstream keeps the PDF content order when it is coherent (`order=pdf`); the Go
  port orders with the order model. **Ported and measured, then rejected:** on
  the full OPB (`-pdf -furniture`) it changes 582/2937 pages but is neutral on the
  headline (0.6052 -> 0.6048) and regresses `olmo_multi_column` (0.8324 ->
  0.8210, -1.14pp). It also barely applies to `realdoc_pdfs` (~1 text-layer page;
  0/88 pages differ). Not shipped. See the Rejected section of
  `OPTIMIZATIONS_LOG.md`.

## PDF text layer: ported fully (v1.1)

- `PDFOptions.Mode` is `"auto"` / `"ocr"` / `"text"`. `"text"` uses the layer
  even when the trust checks fail (upstream `mode="text"`), and in `"auto"` an
  untrusted layer is used when the recognizer cannot read its script and the
  font map is not broken (upstream `document.pdf_page`'s `unreadable` rule).
  `PDFOptions.NoImageOCR` mirrors `image_ocr=False` (drop detected lines the
  layer does not cover instead of OCRing them).
- The v1.1 text-layer phase is ported whole: per-character quarter turns
  (`textlayer._quarter` over the page-to-device affine, `L = F·A·F`), the
  `/Rotate` reset branch (`FPDFPage_Get/SetRotation`), the vertical-layout
  branch, the legacy-font `layer_agrees` check (`box_groups` over the router),
  sideways runs, the broken-font-map turn nuance (`turn = not(reason ==
  BROKEN_FONT_MAP and rotated < 0.5·chars)`), the unreadable-script rescue, and
  `Page.Note` (the text-layer verdict and the "chars outside detected lines"
  count).
- **One render, not two:** upstream renders a check image at `CHECK_DPI = 72`
  and, separately, the text page at the page dpi. The Go port renders once at
  the page dpi (`min(dpi/72, sqrt(MAX_PAGE_PIXELS/area))`) and reuses it for the
  ink checks and the text page. The trust thresholds are scale-robust; boxes are
  kept in scaled pixels throughout. A `/Rotate`-reset page re-renders once more
  (as upstream does).

## Detector: the `picture` channel is dormant; pictures are not ported

- Upstream's `det.onnx` can emit a fourth channel (picture/region hints) and
  `detect_all` returns picture boxes; `pipeline.recognize`/`assemble` carry them
  into layout. The **shipped v1.1.0 `det.onnx` has three channels** (core/top/
  bottom) and upstream's own picture path is dormant (`picture_boxes` gets
  `None`), so the port does not implement it. Decode is unchanged (3 channels).

## Math on a PDF text layer: a garbled reading never replaces the layer's text

- **What:** upstream's `_read_math` replaces a text-layer line with the math
  decoder's reading whenever the reading contains `$` and is confident. The
  decoder often reads *prose* as math — it spells the words out one letter at a
  time (`D a v i d C . K r a k a u e r`) and drops the diacritics — so on a
  paper with superscript affiliations and bracketed citations the front matter
  and the reference lines come out mangled (`Ricard Sole` for `Ricard Solé`).
- **The port keeps the layer's characters when the reading is garbled:** a
  reading with a run of single letters in a row (`M a r c o T u c c i o`) and no
  LaTeX command in that part is the decoder's spelling of prose, and the PDF's
  own text (with its diacritics and its spacing) is better. A part with a LaTeX
  command (`\mathrm { L e m m a }`) is the decoder's own italic-text style and
  is left alone.
- **Scope:** the **text-layer path only**. The image path keeps upstream's
  behaviour on purpose: there the recognizer's reading of a math line is
  garbage, so the math reading is the only source of the LaTeX — dropping it
  costs real quality (measured below), and the benchmark wants that LaTeX.
- **Evidence** (upstream indra-ocr 1.1.2 on arXiv 2609.03344, 12 pages,
  `Math(true)`, PDF text layer): upstream and the port *without* the guard
  produce the same output — 103 lines with `$`, the letter-spaced
  `M a r c o T u c c i o`, and `Ricard Sole` (the `é` lost) — i.e. the port was
  faithful and the defect is upstream's. With the guard: 101 lines with `$`
  (the two mangled lines keep the layer's text), **0** letter-spaced runs, and
  `Ricard Solé` preserved.
- **Cost on the benchmark:** OPB `realdoc_pdfs` (88 pages) in PDF mode with
  `-math`: **0.7344** headline / 0.6523 per suite, *identical* with and without
  the guard. OPB `olmo_arxiv_math` (100 pages, image path, `-math`): 0.7763 /
  0.6690, unchanged (the image path is untouched). A blunt "the reading must be
  mostly math" gate was measured first and **rejected**: it cost 7.0pp headline
  and 15.2pp of the text family on the math suite, because the benchmark wants
  the inline LaTeX of prose lines.

## Quantized models

- `rec.onnx` (the base recognizer) is mixed-precision int8, the three
  script-group recognizers (`rec_cjk`/`rec_rtl`/`rec_world`) are int8 with
  **per-script** calibration, and `order.onnx` is dynamic int8; upstream ships
  FP32 (plus an optional int8 `rec`). This moves the last digits of the output.
  See `OPTIMIZATIONS_LOG.md` for the A/B evidence and the calibration rule.

## Skew/orientation: integer row sums and merge-order ties

- `estimateSkewPy`'s `rowSharpness` accumulates integer-valued pixel sums
  (`255 - v`, with the sampling stride folded back in) rather than Python's float
  accumulation; the two agree to the printed `0.1°` step, the decision
  (`deskewMin`) is unaffected.
- The LM beam search's final tie-break sorts candidates by score; ties at the
  last float bit are resolved in Go's sort order rather than Python's stable
  order (differences ~1e-6, no settled line observed to flip).


## Unreadable pages are skipped, not surfaced

- `CallPDF` skips a page the engine cannot open; upstream yields a per-page
  `Page` with `source="error"` and a `note`. In the Go port `Page.Source` is
  `"ocr"`, `"text"` or `"hybrid"` and `Page.Note` carries the text-layer verdict
  and diagnostics (upstream `Page.note`), but a page the engine cannot open is
  dropped rather than returned as an error page.
