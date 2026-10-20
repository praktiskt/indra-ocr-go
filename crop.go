// Line crop for the recognizer (CropLine). Port of rec_preproc.py.
package indra

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

// Line crop geometry, port of rec_preproc.py.
const (
	lineHeight    = 32
	padY          = 0.2
	padX          = 0.3
	widthMultiple = 8
	maxLineWidth  = 4096
)

// percentileFromHist computes np.percentile(b, 100*q) (linear) of a uint8
// array from its histogram: no sort of the pixels. Port of _percentiles.
func percentileFromHist(hist []int, n int, q float64) float64 {
	x := q * float64(n-1)
	i := math.Floor(x)
	g := x - i
	ii := int(i)
	searchRight := func(v int) int {
		// first index where cum[j] > v
		return sort.Search(len(hist), func(j int) bool { return hist[j] > v })
	}
	lo := float64(searchRight(ii))
	hi := float64(searchRight(min(ii+1, n-1)))
	d := hi - lo
	if g >= 0.5 {
		return hi - d*(1-g)
	}
	return lo + d*g
}

// isDarkLine reports light text on a dark ground (a banner, a dark table
// head, a slide): the ground (the median of the line box) lies nearer the
// dark end than the text does, so the bright tail is the longer one. The
// recognizers read dark text on light paper, so such a line is read
// inverted. Port of is_dark_line in rec_preproc.py.
func isDarkLine(g *Gray, box [4]float64) bool {
	x0, y0, x1, y1 := int(math.Floor(box[0])), int(math.Floor(box[1])), int(math.Ceil(box[2])), int(math.Ceil(box[3]))
	sy0, sy1 := max(0, y0), min(g.H, y1+1)
	sx0, sx1 := max(0, x0), min(g.W, x1+1)
	if sy1 <= sy0 || sx1 <= sx0 {
		return false
	}
	var hist [256]int
	n := 0
	for y := sy0; y < sy1; y++ {
		row := g.Pix[y*g.W : y*g.W+sx1]
		for _, p := range row[sx0:sx1] {
			hist[p]++
		}
		n += sx1 - sx0
	}
	if n == 0 {
		return false
	}
	lo := percentileFromHist(hist[:], n, 0.02)
	med := percentileFromHist(hist[:], n, 0.5)
	hi := percentileFromHist(hist[:], n, 0.98)
	return hi-med > 2*(med-lo)+8
}

// lineDarkness runs isDarkLine over boxes (in parallel for large pages).
func lineDarkness(g *Gray, boxes [][4]float32) []bool {
	out := make([]bool, len(boxes))
	if len(boxes) < 16 {
		for i, b := range boxes {
			out[i] = isDarkLine(g, f64box(b))
		}
		return out
	}
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	for i, b := range boxes {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, b [4]float32) {
			defer wg.Done()
			out[i] = isDarkLine(g, f64box(b))
			<-sem
		}(i, b)
	}
	wg.Wait()
	return out
}

func f64box(b [4]float32) [4]float64 {
	return [4]float64{float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}
}

// CropLine crops a line box (ink box, page px) with PAD_X/PAD_Y margins and
// resizes to height px high, keeping aspect ratio. Returns ink map [height x W].
// window > 1 (a tall window) keeps the text scale and adds the page above and
// below the line; its middle band is the window-1 crop. dark inverts a
// light-on-dark line. Port of crop_line in rec_preproc.py.
func CropLine(g *Gray, box [4]float64, height int, window int, dark bool) []float32 {
	if window < 1 {
		window = 1
	}
	// window > 1 grows the pad on both sides and multiplies the height, so
	// the middle band stays the window-1 crop at the same text scale.
	py := (float64(window)*(1+2*padY) - 1) / 2
	height = height * window

	x0, y0, x1, y1 := box[0], box[1], box[2], box[3]
	h := maxf(1, y1-y0)
	X0, X1 := x0-padX*h, x1+padX*h
	Y0, Y1 := y0-py*h, y1+py*h
	ix0, iy0 := int(math.Floor(X0)), int(math.Floor(Y0))
	ix1, iy1 := int(math.Ceil(X1)), int(math.Ceil(Y1))
	cw := max(1, ix1-ix0)
	ch := max(1, iy1-iy0)
	sx0, sy0 := max(0, ix0), max(0, iy0)
	sx1, sy1 := min(g.W, ix1), min(g.H, iy1)
	fullInside := sx0 == ix0 && sy0 == iy0 && sx1 == ix1 && sy1 == iy1
	var src *Gray
	if fullInside && cw == g.W && !dark {
		// Fast path: crop is full-width slice, no borders, no inversion.
		src = &Gray{Pix: g.Pix[sy0*g.W : sy1*g.W], W: cw, H: ch}
	} else {
		// Build crop; outside the page: paper (255), or black on a dark line.
		fill := uint8(255)
		if dark {
			fill = 0
		}
		crop := make([]uint8, cw*ch)
		for i := range crop {
			crop[i] = fill
		}
		if sx1 > sx0 && sy1 > sy0 {
			w := sx1 - sx0
			for y := sy0; y < sy1; y++ {
				copy(crop[(y-iy0)*cw+(sx0-ix0):(y-iy0)*cw+(sx0-ix0)+w], g.Pix[y*g.W+sx0:y*g.W+sx1])
			}
		}
		src = &Gray{Pix: crop, W: cw, H: ch}
	}
	outW := max(widthMultiple, min(maxLineWidth, int(math.Round(float64(cw)*float64(height)/float64(ch)))))
	var rs *Gray
	if ch > height {
		rs = src.resizeArea(outW, height)
	} else {
		rs = src.resizeLinear(outW, height)
	}
	out := make([]float32, outW*height)
	if dark {
		for i, p := range rs.Pix {
			out[i] = float32(p) / 255
		}
	} else {
		inkSet1Fn(out, rs.Pix)
	}
	return out
}
