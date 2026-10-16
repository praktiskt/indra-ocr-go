# Deviations from upstream indra

The port aims to reproduce the Python reference (`indra_ocr`,
<https://github.com/datalab-to/indra>) on the same inputs. This file records the
**intentional** places where its output differs, so a diff against the reference
is not mistaken for a bug. Everything else is expected to match; the model-level
changes are in [`OPTIMIZATIONS_LOG.md`](OPTIMIZATIONS_LOG.md) and the evaluation
method in [`../.agents/skills/quality-eval/SKILL.md`](../.agents/skills/quality-eval/SKILL.md).

## Text layer: spacing acute is composed

- **What:** in the PDF text layer, a spacing acute `U+00B4` that sits **between
  two letters** is folded into the precomposed acute of the following letter
  (`Cicho´n` -> `Cichoń`, `Fr´echet` -> `Fréchet`, `investigaci´on` ->
  `investigación`, `Mej´ıa` -> `Mejía`). A `U+00B4` that follows a non-letter is
  left as-is (it is a prime/symbol, e.g. `SplExtp´,`).
- **Why:** PDFs draw the accent as a separate glyph positioned before its base,
  and pdfium emits characters in content order, so the raw layer reads
  `Cicho´n`. Upstream passes that through, but the recognizer does not — its
  vocab is NFC (`indra_ocr/vocab.py`), so the OCR path already produces
  `Cichoń`. This makes the two paths agree.
- **Where:** `pdfium.go` (`foldSpacingAcute`), text-layer extraction only; the
  OCR path and image input are unchanged.
- **Impact:** Latin accents in PDF text-layer text are normalized; not applied to
  primes. Low volume (~150 of 389 `U+00B4` in 2149 OPB corpus pages; the rest are
  primes/symbols).
- **Validation:** `TestFoldSpacingAcute` (the containing cases + primes).

## `pdf_order` (PDF content order) not ported

- Upstream keeps the PDF content order when it is coherent (`order=pdf`); the Go
  port orders with the order model. On OPB `realdoc_pdfs` (auto) the Go headline
  is 0.6964 vs upstream 0.7121. Open item; see `OPTIMIZATIONS_LOG.md`.

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
