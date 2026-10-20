# indra-ocr-go

Native Go port of the [Indra OCR pipeline](https://github.com/datalab-to/indra) (Apache-2.0).

~1M-parameter recognizers — a base Latin/multi model plus CJK, RTL (Arabic/Hebrew) and world (Greek/Cyrillic/Indic) models — routed per line by a script classifier, with n-gram language-model beam search. Plus detector, ordering, orientation (including vertical columns), PDF/file input and math (LaTeX) models on ONNX Runtime, CPU only.

The base recognizer ships as a mixed-precision int8 build (raw graph, stem + first two blocks kept FP32) and the reading-order model as dynamic int8; the three script-group recognizers ship FP32, because group-model int8 needs per-script calibration and a per-script quality guard. The math encoder/decoder (`math_enc`/`math_dec_step`, `OCR.Math`) are dynamic int8. See [`tools/`](tools/README.md) for the recipes and [`docs/OPTIMIZATIONS_LOG.md`](docs/OPTIMIZATIONS_LOG.md) for the A/B evidence (FUNSD guard 0.8658 FP32 → 0.8680 int8).

## Requirements

- Go 1.26+
- CGO enabled
- linux/amd64: zero external deps (ONNX Runtime 1.31.0 + libstdc++/libgcc bundled)
- other platforms: install [libonnxruntime](https://github.com/microsoft/onnxruntime) and set `ORT_LIB_PATH`

## Install

```bash
go get github.com/praktiskt/indra-ocr-go
```

## Usage

```go
package main

import (
    "fmt"
    "image"
    _ "image/jpeg"
    "os"

    "github.com/praktiskt/indra-ocr-go"
)

func main() {
    // Create OCR pipeline (models and native libs embedded)
    ocr, err := indra.NewOCR(4)
    if err != nil {
        panic(err)
    }
    defer ocr.Close()

    // Load image
    f, _ := os.Open("page.jpg")
    img, _, _ := image.Decode(f)
    f.Close()

    gray := indra.FromImage(img)

    // Run OCR
    page := ocr.Call(gray)
    fmt.Println(page.Markdown)
}
```

## Batch (pipelined)

`CallAll` OCRs many pages on one handle, overlapping the recognizer (the
bottleneck) with orientation/detection of the next page and assembly of the
previous one. Results keep input order and are identical to calling `Call`
per page, but a single handle sustains ~1.3x the pages/s.

```go
pages := ocr.CallAll(grays) // []*Gray -> []*Page, same order
```

## Pool (concurrent)

```go
// Host-tuned default: half the CPUs as single-threaded workers.
pool, err := indra.NewPoolAuto()
if err != nil {
    panic(err)
}
defer pool.Close()

page := pool.Call(gray)         // one page, blocks for a free worker

pages = pool.CallAll(grays)     // batch: spreads pages over workers,
                                // pipelining stages within each worker
```

Override anything with the builder (all fields optional):

```go
pool, err := indra.NewPoolBuilder().
    Workers(6).                 // 0 = auto (AutoWorkers)
    Threads(1).                 // intra-op threads per model
    MemBudget(2 << 30).         // cap workers to ~2GB (PerHandleBytes each)
    Build()
```

`NewPoolAuto` picks `AutoWorkers()` = `NumCPU/2` workers with one thread
each: throughput plateaus by then, and each extra worker costs a handle
(~120-260MB). `MemBudget` trades workers for a memory ceiling; a few pages of
`Call` or `CallAll` per pool is the intended use.

## Text layer & furniture

Upstream reads a PDF's text layer where it is sound and OCRs the rest
(`mode="auto"`). `CallPDF` does this with the embedded PDFium engine; a caller
that has its own PDF engine (or a text layer from elsewhere) can hand runs to
`CallText` instead:

```go
page := ocr.CallText(gray, &indra.TextLayer{Runs: []indra.TextRun{
    {Text: "Hello", Box: [4]float64{x0, y0, x1, y1}},
    // ... runs in PDF content order. Add per-char Chars (and Quarter/Seq for a
    // real layer) to split a run across detected lines (e.g. table cells).
}})
```

`CallText` trusts the layer only when it is not invisible (render mode 3), has no
broken-font characters, its runs sit on ink, and its right-to-left letters are
stored in logical order (the `indra_ocr/textlayer.py` checks); otherwise it falls
back to plain OCR. `Page.Source` is `"text"` or `"hybrid"` (lines the layer does
not cover are OCR'd); `Page.Note` carries the verdict and diagnostics. Ordering
uses the order model; upstream `pdf_order` is not ported (ported and measured
neutral/negative: it regresses multi-column, see `docs/OPTIMIZATIONS_LOG.md`).

`ocr.Furniture = true` (or `PoolBuilder.Furniture(true)`) keeps page headers and
footers in the markdown, as upstream `load(furniture=True)`; the default drops
them.

## PDF input

`CallPDF` renders PDF bytes with the embedded PDFium engine and, in `"auto"`
mode, reads each page's text layer where it is sound (the `CallText` path above)
— upstream `mode="auto"`. The engine is embedded, nothing is installed.
`CallPDF` is safe to call concurrently — it is fine to run `Pool.CallPDF` from
many goroutines: each call renders on its own PDFium library instance (opened
lazily, up to a small cap).

```go
pages, err := ocr.CallPDF(pdfBytes, nil)                       // auto
pages, err = ocr.CallPDF(pdfBytes, &indra.PDFOptions{Mode: "ocr"})
```

`PDFOptions`: `DPI` (default 200), `Mode` (`"auto"` / `"ocr"` / `"text"`),
`MaxPages`, `NoImageOCR`. `"text"` uses the text layer even when the trust checks
fail; `"auto"` also uses an untrusted layer when the recognizer cannot read its
script, and checks a trusted layer against the page image (legacy fonts).
`NoImageOCR` drops detected lines the layer does not cover instead of OCRing them
(upstream `image_ocr=False`).
Upstream's `pdf_order` (keep the PDF content order) is not ported — Go orders
with the order model. It was ported and measured: neutral on the full OPB
headline and -1.14pp on `olmo_multi_column`, so it is rejected (see
`docs/OPTIMIZATIONS_LOG.md`).

## API

The pipeline is `NewOCR` (one handle) or `NewPool*` (concurrent). Both expose
`Call` (image), `CallTurn` (image, orientation on/off), `CallAll` (batch),
`CallText` (caller-supplied PDF text layer) and `CallPDF` (PDF bytes). `Page`
holds `Lines` (reading order), `Blocks`, `Markdown`, `Source`, `Note`,
`Rotation`, `Skew` and `Vertical` (a page set in vertical columns; its boxes are
mapped back to the page as-is).

```go
ocr, _ := indra.NewOCR(4)       // single handle, 4 intra-op threads
defer ocr.Close()
page := ocr.Call(indra.FromImage(img))

pool, _ := indra.NewPoolAuto()  // NumCPU/2 workers x 1 thread
defer pool.Close()
pages := pool.CallFiles(paths)  // read+decode in parallel, OCR over the pool
```

Options are fields on the handle or methods on the builder:

```go
ocr.Furniture = true            // keep page headers/footers
ocr.Math = true                 // read math lines as LaTeX (opt-in)

pool, _ := indra.NewPoolBuilder().
    Workers(6).Threads(1).MemBudget(2 << 30).Furniture(true).Build()
```

PDF input and the text layer are described above (`CallPDF`, `CallText`,
`PDFOptions`). `Page.Text()` is the reading-order line texts joined by newlines.

`InitLib()` loads ONNX Runtime eagerly (optional; the constructors do it).
Types: `OCR`, `Pool`, `PoolBuilder`, `Page`, `Line`, `Block`, `Gray`,
`TextLayer`/`TextRun`/`TextChar`, `PDFOptions`, `Threads`. `AutoWorkers` and
`PerHandleBytes` back the pool defaults.

To drive the stages directly: `FromImage` / `FromRGBA` / `NewGray` build a
`Gray`; `OCR.Detect`, `OCR.Upright` and `OCR.RecHeight` expose detection and
deskew; `CropLine` / `PadBatch` / `WidthBatches` reproduce the recognizer's line
preprocessing (used by [`tools/calib`](tools/)).

## Environment variables

| Variable | Description |
|----------|-------------|
| `ORT_LIB_PATH` | Path to libonnxruntime shared library (overrides bundled libs) |

### Optional SIMD builds

Building with `GOEXPERIMENT=simd` (Go 1.27+) replaces the per-pixel ink
conversions (`fillInk` and `CropLine`'s ink map) with AVX-512 fast paths.
They are bit-identical to the scalar results on every corpus page; without
the experiment the library compiles and runs the scalar code. The AVX-512
kernels are selected at runtime (via a CPU feature check), so a
`GOEXPERIMENT=simd` binary still runs the scalar code on CPUs without
AVX-512 instead of trapping. `Pool` also
gains `CallFiles(paths)`: parallel file decode feeding the pipelined pool,
the full equivalent of the reference `convert_many`.

## License

Code Apache-2.0. Models [OpenRAIL-M](LICENSE-MODELS). Bundled native libs [MIT / GPLv3+Runtime Exception](LICENSE-LIBS).
