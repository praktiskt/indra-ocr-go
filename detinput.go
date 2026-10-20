// Detector input (pageToInput + ink map) and recognizer batch packing
// (PadBatch/WidthBatches, orientInput).
package indra

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

const inputMultiple = 32

// pageToInput resizes so the long side is longSide and pads to a multiple of
// 32 with white. Returns the float32 ink map (0=paper), padded size and scale.
// Port of page_to_input in det_post.py.
func pageToInput(g *Gray, longSide int) (ink []float32, pw, ph int, scale float64) {
	scale = float64(longSide) / float64(max(g.W, g.H))
	nw := max(1, int(math.Round(float64(g.W)*scale)))
	nh := max(1, int(math.Round(float64(g.H)*scale)))
	var rs *Gray
	switch {
	case nw == g.W && nh == g.H:
		rs = g
	case scale < 1:
		rs = g.resizeArea(nw, nh)
	default:
		rs = g.resizeLinear(nw, nh)
	}
	pw = ((nw + inputMultiple - 1) / inputMultiple) * inputMultiple
	ph = ((nh + inputMultiple - 1) / inputMultiple) * inputMultiple
	ink = make([]float32, pw*ph) // zero = paper
	fillInk(ink, pw, rs.Pix, rs.W, nw, nh)
	return ink, pw, ph, scale
}

// fillInk writes the ink map (1-p/255) for the nw x nh top-left region.
// Rows are independent; sharded across CPUs for page-sized inputs.
func fillInk(ink []float32, pw int, src []uint8, sw, nw, nh int) {
	if nh < 256 {
		fillInkRows(ink, pw, src, sw, nw, 0, nh)
		return
	}
	workers := runtime.NumCPU()
	if workers > nh {
		workers = nh
	}
	var wg sync.WaitGroup
	rowsPer := (nh + workers - 1) / workers
	for wi := 0; wi < workers; wi++ {
		y0, y1 := wi*rowsPer, min(nh, (wi+1)*rowsPer)
		if y0 >= y1 {
			break
		}
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			fillInkRows(ink, pw, src, sw, nw, y0, y1)
		}(y0, y1)
	}
	wg.Wait()
}

// fillInkRowsScalar is the original per-row body (also the row-tail helper of
// the SIMD build, where fillInkRows is a var pointing at the fast path).
func fillInkRowsScalar(ink []float32, pw int, src []uint8, sw, nw, y0, y1 int) {
	for y := y0; y < y1; y++ {
		ou, iu := y*pw, y*sw
		for x := 0; x < nw; x++ {
			ink[ou+x] = 1 - float32(src[iu+x])/255
		}
	}
}

// fillInkRows selects the per-row fill; the SIMD build replaces it with the
// AVX-512 fast path. inkSet1Fn handles the flat CropLine conversion.
var (
	fillInkRows func(ink []float32, pw int, src []uint8, sw, nw, y0, y1 int) = fillInkRowsScalar
	inkSet1Fn   func(ink []float32, src []uint8)                             = inkSet1Scalar
)

// inkSet1Scalar fills a flat sink with 1 - p/255 (scalar reference body; the
// SIMD build swaps it out via inkSet1Fn).
func inkSet1Scalar(ink []float32, src []uint8) {
	for i, p := range src {
		ink[i] = 1 - float32(p)/255
	}
}

var fillInkRowDevice = 0

// PadBatch right-pads ink maps of one height to one width. Returns
// [B*H*Wm] NCHW-flattened (C=1) plus widths.
func PadBatch(lines [][]float32, h int, widths []int) (batched []float32, wOut []int64, wm int) {
	wm = 0
	for _, w := range widths {
		if w > wm {
			wm = w
		}
	}
	wm = ((wm + widthMultiple - 1) / widthMultiple) * widthMultiple
	batched = make([]float32, len(lines)*h*wm)
	wOut = make([]int64, len(lines))
	for i, ln := range lines {
		w := widths[i]
		for r := 0; r < h; r++ {
			copy(batched[(i*h+r)*wm:(i*h+r)*wm+w], ln[r*w:(r+1)*w])
		}
		wOut[i] = int64(w)
	}
	return batched, wOut, wm
}

// WidthBatches groups line indices narrowest-first so little compute goes to
// padding. Port of width_batches in rec_preproc.py.
func WidthBatches(widths []int, maxBatch int) [][]int {
	order := make([]int, len(widths))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return widths[order[a]] < widths[order[b]] })
	const maxRatio = 1.3
	const minWidth = 64
	const maxArea = 64 * 640
	var batches [][]int
	for _, idx := range order {
		if n := len(batches); n > 0 {
			last := batches[n-1]
			first := last[0]
			if len(last) < maxBatch &&
				max(widths[idx], minWidth) <= int(float64(max(widths[first], minWidth))*maxRatio) &&
				(len(last)+1)*max(widths[idx], minWidth) <= maxArea {
				batches[n-1] = append(last, idx)
				continue
			}
		}
		batches = append(batches, []int{idx})
	}
	return batches
}

// orientInput builds the [side,side] ink map for orient.onnx. Port of orient_input.
func orientInput(g *Gray, side int) []float32 {
	s := float64(side) / float64(max(g.W, g.H))
	nw := max(1, int(math.Round(float64(g.W)*s)))
	nh := max(1, int(math.Round(float64(g.H)*s)))
	sm := g.resizeArea(nw, nh)
	out := make([]float32, side*side)
	fillInk(out, side, sm.Pix, sm.W, nw, nh)
	return out
}
