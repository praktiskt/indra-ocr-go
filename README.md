# indra-ocr-go

Native Go port of the [Indra OCR pipeline](https://github.com/datalab-to/indra) (Apache-2.0).

~1M-parameter recognizer plus detector / ordering / orientation models on ONNX Runtime, CPU only.

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
pool, err := indra.NewPool(4, 4) // 4 workers, 4 threads each
if err != nil {
    panic(err)
}
defer pool.Close()

page := pool.Call(gray)         // one page, blocks for a free worker

pages = pool.CallAll(grays)     // batch: spreads pages over workers,
                                // pipelining stages within each worker
```

## API

| Function | Description |
|----------|-------------|
| `InitLib()` | Initialize ONNX Runtime library |
| `NewOCR(threads)` | Create OCR pipeline |
| `(*OCR).CallAll(pages)` | Batch OCR on one handle, stages pipelined across pages |
| `NewPool(workers, threads)` | Create concurrent pool |
| `(*Pool).CallAll(pages)` | Batch OCR across pool workers, pipelined within each |
| `VocabFromJSON(s)` | Parse vocabulary JSON |
| `NewVocab(chars)` | Create vocabulary |
| `DecodeDet(out, h, w)` | Decode detector output |
| `LineFeatures(boxes, w, h, texts)` | Extract line features |
| `DecodeOrder(a, sweeps)` | Decode ordering matrix |
| `OrderXYCut(boxes)` | XY-cut ordering |
| `PageBlocks(...)` | Extract layout blocks |
| `PageMarkdown(...)` | Generate markdown |
| `TableHTML(boxes, texts)` | Generate table HTML |
| `NewGray(px, w, h)` | Create grayscale image |
| `FromImage(img)` | Convert any image.Image to grayscale |
| `FromRGBA(img)` | Convert *image.RGBA to grayscale |
| `PageToInput(g, longSide)` | Prepare detector input |
| `CropLine(g, box, height)` | Crop line for recognizer |
| `RotatePage(g, degrees)` | Rotate page |
| `EstimateSkew(g)` | Estimate skew angle |

## Environment variables

| Variable | Description |
|----------|-------------|
| `ORT_LIB_PATH` | Path to libonnxruntime shared library (overrides bundled libs) |

## License

Code Apache-2.0. Models [OpenRAIL-M](LICENSE-MODELS). Bundled native libs [MIT / GPLv3+Runtime Exception](LICENSE-LIBS).
