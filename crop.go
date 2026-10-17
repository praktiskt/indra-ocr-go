// Line crop for the recognizer (CropLine). Port of rec_preproc.py.
package indra

import (
	"math"
)

// Line crop geometry, port of rec_preproc.py.
const (
	lineHeight    = 32
	padY          = 0.2
	padX          = 0.3
	widthMultiple = 8
	maxLineWidth  = 4096
)

// CropLine crops a line box (ink box, page px) with PAD_X/PAD_Y margins and
// resizes to height px high, keeping aspect ratio. Returns ink map [height x W].
func CropLine(g *Gray, box [4]float64, height int) []float32 {
	x0, y0, x1, y1 := box[0], box[1], box[2], box[3]
	h := maxf(1, y1-y0)
	X0, X1 := x0-padX*h, x1+padX*h
	Y0, Y1 := y0-padY*h, y1+padY*h
	ix0, iy0 := int(math.Floor(X0)), int(math.Floor(Y0))
	ix1, iy1 := int(math.Ceil(X1)), int(math.Ceil(Y1))
	cw := max(1, ix1-ix0)
	ch := max(1, iy1-iy0)
	sx0, sy0 := max(0, ix0), max(0, iy0)
	sx1, sy1 := min(g.W, ix1), min(g.H, iy1)
	fullInside := sx0 == ix0 && sy0 == iy0 && sx1 == ix1 && sy1 == iy1
	var src *Gray
	if fullInside && cw == g.W {
		// Fast path: crop is full-width slice, no white borders.
		src = &Gray{Pix: g.Pix[sy0*g.W : sy1*g.W], W: cw, H: ch}
	} else {
		// Build crop with white outside the page.
		crop := make([]uint8, cw*ch)
		for i := range crop {
			crop[i] = 255
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
	inkSet1Fn(out, rs.Pix)
	return out
}
