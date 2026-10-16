# indra-ocr-go

Native Go port of the [Indra OCR pipeline](https://github.com/datalab-to/indra) (Apache-2.0).

~1M-parameter recognizer plus detector / ordering / orientation models on ONNX Runtime, CPU only.

The recognizer ships as a mixed-precision int8 build (stem + first two blocks kept FP32, and Convs promoted Conv1d→Conv2d so ONNX Runtime applies its NCHWc layout): ~2.4x faster than the FP32 model at the pool's one-thread-per-worker setting for ~0.15pp less word recall. See [`tools/`](tools/README.md) for the recipe.

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

## API

| Function | Description |
|----------|-------------|
| `InitLib()` | Initialize ONNX Runtime library |
| `NewOCR(threads)` | Create OCR pipeline |
| `(*OCR).CallAll(pages)` | Batch OCR on one handle, stages pipelined across pages |
| `NewPool(workers, threads)` | Create concurrent pool (explicit sizing) |
| `NewPoolAuto()` | Pool sized for the host (AutoWorkers, 1 thread each) |
| `NewPoolBuilder()` | Chainable builder: Workers/Threads/ThreadsPerModel/MemBudget |
| `AutoWorkers()` | Default worker count (NumCPU/2) |
| `PerHandleBytes` | Estimated resident memory per worker (budget unit) |
| `(*Pool).CallAll(pages)` | Batch OCR across pool workers, pipelined within each |
| `(*Pool).CallFiles(paths)` | Read+decode files in parallel, OCR over the pool (paths -> pages) |
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
