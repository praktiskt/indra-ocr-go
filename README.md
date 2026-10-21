# indra-ocr-go

Native Go port of the [Indra OCR pipeline](https://github.com/datalab-to/indra)
(Apache-2.0), tracking upstream package **indra-ocr 1.1.2** on model set
**`v1.1.0`** (the two 1.1.x code releases after 1.1.0 share those weights:
1.1.1 distrusts a right-to-left text layer stored in visual order, 1.1.2 checks a
layer's script against the page image before trusting it — both ported here).
CPU only, ONNX Runtime, all models and native libraries embedded — nothing is
downloaded at runtime.

Four script-group recognizers (base Latin/Greek/Cyrillic, CJK, RTL, world
Indic/Caucasian) routed per line by a script classifier, with n-gram
language-model beam search; plus detector, reading order, orientation (including
vertical columns) and optional math (LaTeX) models.

Every recognizer ships as a mixed-precision int8 build (group models with
**per-script** calibration) and the order model as dynamic int8. Measured on the
199-page FUNSD guard: **0.9318** word recall, 16.6 pages/s at the pool's default
8 workers x 1 thread. See [`tools/`](tools/README.md) for the recipes and
[`docs/OPTIMIZATIONS_LOG.md`](docs/OPTIMIZATIONS_LOG.md) for the evidence.

## Requirements

- Go 1.26+, CGO enabled
- linux/amd64: no external deps (ONNX Runtime 1.31.0 + libstdc++/libgcc bundled)
- other platforms: install [libonnxruntime](https://github.com/microsoft/onnxruntime)
  and set `ORT_LIB_PATH`

```bash
go get github.com/praktiskt/indra-ocr-go
```

## Usage

```go
ocr, err := indra.NewOCR(4)   // one handle, 4 intra-op threads
defer ocr.Close()

img, _ := os.Open("page.jpg")
decode, _, _ := image.Decode(img)
page := ocr.Call(indra.FromImage(decode))   // *Page
fmt.Println(page.Markdown)                 // markdown, blocks, reading order
```

## API

| | |
|---|---|
| `NewOCR(threads)` | one handle; `Call` (image), `CallTurn` (image, orientation on/off), `CallAll` (batch), `CallText` (caller-supplied text layer), `CallPDF` (PDF bytes) |
| `NewPoolAuto()` / `NewPoolBuilder()` | concurrent workers (`NumCPU/2` x 1 thread by default); same calls, plus `CallFiles(paths)` |
| `Page` | `Lines` (reading order), `Blocks`, `Markdown`, `Text()`, `Source`, `Note`, `Rotation`, `Skew`, `Vertical` |
| options | `ocr.Furniture` (keep headers/footers), `ocr.Math` (read math as LaTeX), `PoolBuilder.Workers/Threads/MemBudget/Furniture` |

`CallAll` pipelines one handle (recognition of one page overlaps detection of
the next) and returns the pages in input order, identical to calling `Call`
per page.

```go
pool, _ := indra.NewPoolBuilder().Workers(6).Threads(1).MemBudget(2 << 30).Build()
defer pool.Close()
pages := pool.CallAll(grays)
```

## PDF and text layer

`CallPDF` renders PDF bytes with the embedded PDFium engine and, in `"auto"`
mode, uses each page's text layer where it is sound and OCRs the rest — the
upstream `mode="auto"` behaviour. It is safe to call concurrently: each call
renders on its own PDFium instance.

```go
pages, err := ocr.CallPDF(pdfBytes, nil)                                  // auto
pages, err = ocr.CallPDF(pdfBytes, &indra.PDFOptions{Mode: "ocr"})
```

`PDFOptions`: `DPI` (200), `Mode` (`"auto"` / `"ocr"` / `"text"`), `MaxPages`,
`NoImageOCR` (drop lines the layer does not cover instead of OCRing them).
`"text"` uses the layer even when the trust checks fail.

A caller with its own PDF engine can hand the runs over instead:

```go
page := ocr.CallText(gray, &indra.TextLayer{Runs: []indra.TextRun{
    {Text: "Hello", Box: [4]float64{x0, y0, x1, y1}},
}})
```

`Page.Source` is `"text"` (all lines from the layer), `"hybrid"` (the rest
OCR'd) or `"ocr"`; `Page.Note` carries the layer verdict. The layer is trusted
only when it is not invisible (an OCR layer over a scan), has no broken-font
characters, sits on ink, and stores right-to-left letters in logical order.

Upstream's `pdf_order` (keep the PDF content order) is not ported: it was
ported and measured neutral on the OPB headline and −1.14pp on
`olmo_multi_column`, so Go orders with the order model.

## Notes

- `ORT_LIB_PATH` — path to a libonnxruntime shared library (overrides the
  bundled one).
- Building with `GOEXPERIMENT=simd` (Go 1.27+) swaps the per-pixel ink
  conversions for AVX-512 fast paths. The kernels are picked at runtime, so
  the binary still runs the scalar code on CPUs without AVX-512.
- `InitLib()` loads ONNX Runtime eagerly (optional; the constructors do it).
- To drive the stages directly: `FromImage`/`FromRGBA`/`NewGray` build a
  `Gray`; `OCR.Detect`, `OCR.Upright`, `OCR.RecHeight` expose detection and
  deskew; `CropLine`/`PadBatch`/`WidthBatches` reproduce the recognizer's
  preprocessing.

## License

Code Apache-2.0. Models [OpenRAIL-M](LICENSE-MODELS). Bundled native libs
[MIT / GPLv3+Runtime Exception](LICENSE-LIBS).
