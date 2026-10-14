# indra-ocr-go

Native Go port of the [Indra OCR pipeline](https://github.com/datalab-to/indra) (Apache-2.0).

~1M-parameter recognizer plus detector / ordering / orientation models on ONNX Runtime, CPU only.

## Requirements

- Go 1.26+
- CGO enabled
- libonnxruntime (see [onnxruntime_go](https://github.com/yalue/onnxruntime_go))

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
    // Initialize ONNX Runtime
    if err := indra.InitLib(); err != nil {
        panic(err)
    }

    // Create OCR pipeline (models embedded, no download needed)
    ocr, err := indra.NewOCR(4)
    if err != nil {
        panic(err)
    }
    defer ocr.Close()

    // Load image
    f, _ := os.Open("page.jpg")
    img, _, _ := image.Decode(f)
    f.Close()

    gray := indra.FromRGBA(img.(*image.RGBA))

    // Run OCR
    page := ocr.Call(gray)
    fmt.Println(page.Markdown)
}
```

## Pool (concurrent)

```go
pool, err := indra.NewPool(4, 4) // 4 workers, 4 threads each
if err != nil {
    panic(err)
}
defer pool.Close()

page := pool.Call(gray)
```

## API

| Function | Description |
|----------|-------------|
| `InitLib()` | Initialize ONNX Runtime library |
| `NewOCR(threads)` | Create OCR pipeline |
| `NewPool(workers, threads)` | Create concurrent pool |
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
| `FromRGBA(img)` | Convert RGBA to grayscale |
| `PageToInput(g, longSide)` | Prepare detector input |
| `CropLine(g, box, height)` | Crop line for recognizer |
| `RotatePage(g, degrees)` | Rotate page |
| `EstimateSkew(g)` | Estimate skew angle |

## Environment variables

| Variable | Description |
|----------|-------------|
| `ORT_LIB_PATH` | Path to libonnxruntime shared library (skips standard search) |

## License

Code Apache-2.0. Models [OpenRAIL-M](LICENSE-MODELS). See file for terms.
