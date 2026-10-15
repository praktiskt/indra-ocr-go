// Command calib extracts recognizer calibration batches from page images.
//
// It runs the real pipeline (upright -> detect -> crop) to produce exactly the
// line crops the recognizer sees at inference, then packs each width batch as
// the recognizer's two inputs:
//
//	x_NNNN.raw  little-endian float32, shape [B, 1, 32, W]  (padded crops)
//	w_NNNN.raw  little-endian int64,   shape [B]            (per-line widths)
//
// These feed tools/quantize_rec.py. Calibrating on pipeline crops (rather than
// arbitrary text images) is what matches the runtime activation distribution.
//
// Usage:
//
//	INDRA_CORPUS_DIR=/path/to/corpus go run ./tools/calib -out ./calib
//
// The corpus dir needs an img/ subdir of PNG pages; any document images work.
package main

import (
	"bytes"
	"encoding/binary"
	"flag"
	"fmt"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sort"

	indra "github.com/praktiskt/indra-ocr-go"
)

func writeF32(path string, v []float32) {
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		panic(err)
	}
}

func writeI64(path string, v []int64) {
	b := make([]byte, len(v)*8)
	for i, f := range v {
		binary.LittleEndian.PutUint64(b[i*8:], uint64(f))
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		panic(err)
	}
}

func main() {
	out := flag.String("out", "calib", "output directory for x_*.raw / w_*.raw")
	corpus := flag.String("corpus", os.Getenv("INDRA_CORPUS_DIR"), "dir with an img/ subdir of PNG pages")
	batch := flag.Int("batch", 64, "width bucket size when packing crops")
	limit := flag.Int("limit", 1<<30, "max pages to process")
	flag.Parse()

	if *corpus == "" {
		fmt.Fprintln(os.Stderr, "set -corpus or INDRA_CORPUS_DIR")
		os.Exit(2)
	}
	files, _ := filepath.Glob(filepath.Join(*corpus, "img", "*.png"))
	sort.Strings(files)
	if len(files) > *limit {
		files = files[:*limit]
	}
	if len(files) == 0 {
		fmt.Fprintf(os.Stderr, "no PNGs in %s/img\n", *corpus)
		os.Exit(2)
	}
	if err := indra.InitLib(); err != nil {
		panic(err)
	}
	ocr, err := indra.NewOCR(1)
	if err != nil {
		panic(err)
	}
	defer ocr.Close()
	if err := os.MkdirAll(*out, 0o755); err != nil {
		panic(err)
	}

	idx, nlines := 0, 0
	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		img, err := png.Decode(bytes.NewReader(data))
		if err != nil {
			continue
		}
		g := indra.FromImage(img)
		g2, _, _ := ocr.Upright(g)
		boxes := ocr.Detect(g2)
		if len(boxes) == 0 {
			continue
		}
		crops := make([][]float32, len(boxes))
		widths := make([]int, len(boxes))
		for i, b := range boxes {
			fb := [4]float64{float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}
			crops[i] = indra.CropLine(g2, fb, ocr.M.RecHeight)
			widths[i] = len(crops[i]) / ocr.M.RecHeight
		}
		for _, part := range indra.WidthBatches(widths, *batch) {
			lines := make([][]float32, len(part))
			hs := make([]int, len(part))
			ws := make([]int, len(part))
			for k, j := range part {
				lines[k], hs[k], ws[k] = crops[j], ocr.M.RecHeight, widths[j]
			}
			batched, wOut, _ := indra.PadBatch(lines, hs, ws)
			writeF32(fmt.Sprintf("%s/x_%04d.raw", *out, idx), batched)
			writeI64(fmt.Sprintf("%s/w_%04d.raw", *out, idx), wOut)
			idx++
			nlines += len(part)
		}
	}
	fmt.Printf("wrote %d batches, %d lines from %d pages\n", idx, nlines, len(files))
}
