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

## `mode="text"` and the unreadable-script nuance are not ported

- `PDFOptions.Mode` is `"auto"` or `"ocr"`. Upstream also has `"text"` (use the
  layer even when the trust checks fail) and, in `auto`, uses the layer for
  scripts the recognizer cannot read. Not implemented.

## Quantized models

- `rec.onnx` is mixed-precision int8 and `order.onnx` is dynamic int8; upstream
  ships FP32 (plus an optional int8 `rec`). This moves the last digits of the
  output (rec ~-0.16pp word recall; order reading-order neutral). See
  `OPTIMIZATIONS_LOG.md` for the A/B evidence.

## Unreadable pages are skipped, not surfaced

- `CallPDF` skips a page the engine cannot open; upstream yields a per-page
  `Page` with `source="error"` and a `note`. In the Go port `Page.Source` is
  `"ocr"`, `"text"` or `"hybrid"`.
