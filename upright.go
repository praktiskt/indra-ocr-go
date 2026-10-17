// Orientation (orient.onnx) and page uprighting (orient + deskew).
package indra

import (
	"math"
)

// Orientation returns quarter turns CCW from orient.onnx (0 without model).
func (o *OCR) orientation(g *Gray) int {
	if o.s.orient == nil {
		return 0
	}
	ink := orientInput(g, 768)
	p, err := o.s.runOrient(ink)
	if err != nil {
		return 0
	}
	k := 0
	for i := 1; i < 4; i++ {
		if p[i] > p[k] {
			k = i
		}
	}
	if p[k] > OrientMin {
		return k
	}
	return 0
}

// Upright turns the page upright and straightens tilt.
// Orientation (model) and skew search (CPU) run concurrently on the
// common upright path; rotated pages re-estimate skew after rotation.
func (o *OCR) Upright(g *Gray) (*Gray, int, float64) {
	kCh := make(chan int, 1)
	aCh := make(chan float64, 1)
	go func() { kCh <- o.orientation(g) }()
	go func() { aCh <- estimateSkewPy(g) }()
	k := <-kCh
	var a float64
	if k != 0 {
		<-aCh // discard pre-rotation estimate
		g = rot90CCW(g, -k)
		a = estimateSkewPy(g)
	} else {
		a = <-aCh
	}
	if math.Abs(a) >= deskewMin {
		g = rotatePage(g, a)
	} else {
		a = 0
	}
	return g, k, a
}

// Call OCRs one page image.
