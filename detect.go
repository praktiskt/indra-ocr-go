// Line detection (det.onnx + the adaptive second pass) and line cropping.
package indra

import (
	"math"
	"runtime"
	"sort"
	"sync"
)

// Detect runs det.onnx (+ adaptive second pass), returns boxes in page px.
func (o *OCR) Detect(g *Gray) [][4]float32 {
	ink, pw, ph, scale := pageToInput(g, o.m.DetLongSide)
	raw, oh, ow, err := o.s.runDet(ink, ph, pw)
	if err != nil {
		return nil
	}
	boxes := decodeDet(raw, oh, ow)
	if len(boxes) >= 8 {
		hs := make([]float64, len(boxes))
		for i, b := range boxes {
			hs[i] = float64(b.Y1 - b.Y0)
		}
		sort.Float64s(hs)
		med := npMedian(hs)
		if med < adaptiveMinH {
			ls := min(adaptiveMaxSide, int(float64(o.m.DetLongSide)*adaptiveTargetH/math.Max(med, 1))) / 32 * 32
			if ls > o.m.DetLongSide {
				ink2, pw2, ph2, scale2 := pageToInput(g, ls)
				raw2, oh2, ow2, err2 := o.s.runDet(ink2, ph2, pw2)
				if err2 == nil {
					boxes = decodeDet(raw2, oh2, ow2)
					scale = scale2
				}
			}
		}
	}
	out := make([][4]float32, 0, len(boxes))
	for _, b := range boxes {
		x0 := clampf32(b.X0/float32(scale), 0, float32(g.W))
		y0 := clampf32(b.Y0/float32(scale), 0, float32(g.H))
		x1 := clampf32(b.X1/float32(scale), 0, float32(g.W))
		y1 := clampf32(b.Y1/float32(scale), 0, float32(g.H))
		out = append(out, [4]float32{x0, y0, x1, y1})
	}
	return out
}

func clampf32(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// lineCrop is one line's grayscale crop at rec height.
type lineCrop struct {
	ink []float32
	w   int
}

// cropLines crops every box to `height` px (times the window; a window > 1
// reads the page above and below the line) — in parallel for large pages.
// dark (when not nil) inverts light-on-dark lines.
func (o *OCR) cropLines(g *Gray, boxes [][4]float32, height, window int, dark []bool) []lineCrop {
	n := len(boxes)
	crops := make([]lineCrop, n)
	crop1 := func(i int, b [4]float32) {
		fb := f64box(b)
		d := false
		if dark != nil {
			d = dark[i]
		}
		ink := CropLine(g, fb, height, window, d)
		crops[i] = lineCrop{ink, len(ink) / (height * window)}
	}
	if n >= 16 {
		var wg sync.WaitGroup
		sem := make(chan struct{}, runtime.NumCPU())
		for i, b := range boxes {
			wg.Add(1)
			sem <- struct{}{}
			go func(i int, b [4]float32) {
				defer wg.Done()
				crop1(i, b)
				<-sem
			}(i, b)
		}
		wg.Wait()
	} else {
		for i, b := range boxes {
			crop1(i, b)
		}
	}
	return crops
}
