// Orientation (orient.onnx) and page uprighting (orient + deskew, up to 45
// degrees, plus vertical-column pages).
package indra

import (
	"math"
	"runtime"
	"sync"
)

const (
	// deskewMax: degrees of tilt searched (quarter turns cover the rest).
	deskewMax = 45.0
	// deskewMin: smaller tilts are left alone (the search step is 0.1).
	deskewMin = 0.3
	// previewSide: the skew search and orient.onnx read thumbnails of at
	// most 800 pixels, taken from this one shrink of the page.
	previewSide = 1600
	// orientVertical: orient.onnx class (5-class models): the page is set in
	// vertical columns.
	orientVertical = 4
)

// preview is the page shrunk once to at most side pixels a side (area
// averaging): the skew search and the orientation model read it instead of
// the full scan. Port of preview.
func (o *OCR) preview(g *Gray, side int) *Gray {
	s := float64(side) / float64(max(g.W, g.H))
	if s >= 1 {
		return g
	}
	return g.resizeArea(max(1, int(math.Round(float64(g.W)*s))), max(1, int(math.Round(float64(g.H)*s))))
}

// inkThumb is the page shrunk to side pixels as gray (the sharpness sums read
// 255-p, so the ink map stays implicit).
func inkThumb(g *Gray, side int) *Gray {
	s := float64(side) / float64(max(g.W, g.H))
	return g.resizeArea(max(1, int(math.Round(float64(g.W)*s))), max(1, int(math.Round(float64(g.H)*s))))
}

// rowSharpness: how sharply the horizontal ink profile changes row to row,
// the page turned by a degrees: highest when the text rows are level. Black
// border (cv2's warpAffine default) fused; sampling stride keeps the argmax.
func rowSharpness(src *Gray, a float64, stride int) float64 {
	rad := a * math.Pi / 180
	cos, sin := math.Cos(rad), math.Sin(rad)
	w, h := src.W, src.H
	cx, cy := float64(w)/2, float64(h)/2
	sums := make([]float64, h)
	inv := 1.0 / 255.0
	for y := 0; y < h; y++ {
		dy := float64(y) - cy
		s := 0.0
		sxf := cos*(-cx) - sin*dy + cx
		syf := sin*(-cx) + cos*dy + cy
		for x := 0; x < w; x += stride {
			x0 := math.Floor(sxf)
			y0i := math.Floor(syf)
			fx := sxf - x0
			fy := syf - y0i
			xa, xb := int(x0), int(x0)+1
			ya, yb := int(y0i), int(y0i)+1
			// Sample(x, y) = src pixel if in bounds else 0 (black).
			sample := func(xi, yi int) float64 {
				if xi < 0 || xi >= w || yi < 0 || yi >= h {
					return 0
				}
				return float64(src.Pix[yi*w+xi])
			}
			v := sample(xa, ya)*(1-fx)*(1-fy) + sample(xb, ya)*fx*(1-fy) +
				sample(xa, yb)*(1-fx)*fy + sample(xb, yb)*fx*fy
			s += 255 - v
			sxf += cos * float64(stride)
			syf += sin * float64(stride)
		}
		sums[y] = s * float64(stride) * inv
	}
	tot := 0.0
	for y := 1; y < h; y++ {
		d := sums[y] - sums[y-1]
		tot += d * d
	}
	return tot
}

func evalAngles(angles []float64, src *Gray, stride int) []float64 {
	vals := make([]float64, len(angles))
	var wg sync.WaitGroup
	sem := make(chan struct{}, runtime.NumCPU())
	for i, a := range angles {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, a float64) {
			defer wg.Done()
			vals[i] = rowSharpness(src, a, stride)
			<-sem
		}(i, a)
	}
	wg.Wait()
	return vals
}

func arangeF(start, step float64, n int) []float64 {
	out := make([]float64, n)
	for k := range out {
		out[k] = start + float64(k)*step
	}
	return out
}

// estimateSkewPy returns degrees (counter-clockwise) that straighten the
// text rows of a page: the angle whose horizontal ink profile changes most
// sharply row to row. Searched over +-45 degrees in 2 degree steps on a
// half-size thumbnail; the two best angles and 0 are then refined in 0.5 and
// 0.1 degree steps on the full thumbnail (800 pixels), where small text rows
// are still resolved. Port of estimate_skew.
func estimateSkewPy(g *Gray) float64 {
	const side = 800
	fine := inkThumb(g, side)
	mean := 0.0
	for _, p := range fine.Pix {
		mean += float64(255 - p)
	}
	if mean/float64(len(fine.Pix))/255 < 0.003 { // a blank page has no rows
		return 0
	}
	coarse := inkThumb(g, side/2)
	grid := arangeF(-deskewMax, 2.0, 46) // -45..45 step 2
	scores := evalAngles(grid, coarse, 4)
	// the two best grid angles and 0 start the refinement
	i0, i1 := 0, 1
	if scores[1] > scores[0] {
		i0, i1 = 1, 0
	}
	for i := 2; i < len(grid); i++ {
		if scores[i] > scores[i0] {
			i1, i0 = i0, i
		} else if scores[i] > scores[i1] {
			i1 = i
		}
	}
	starts := []float64{0, grid[i0], grid[i1]}
	// The 0.5 degree ranges of nearby starts overlap: each angle scored once.
	seen := map[float64]bool{}
	var near []float64
	for _, a0 := range starts {
		for k := 0; k < 9; k++ {
			a := a0 - 2.0 + float64(k)*0.5
			if !seen[a] {
				seen[a] = true
				near = append(near, a)
			}
		}
	}
	nv := evalAngles(near, fine, 2)
	a1 := near[0]
	best := nv[0]
	for i := 1; i < len(near); i++ {
		if nv[i] > best {
			best, a1 = nv[i], near[i]
		}
	}
	last := arangeF(a1-0.5, 0.1, 11)
	lv := evalAngles(last, fine, 1)
	fineB := a1
	fineV := best
	for i := range last {
		if lv[i] > fineV {
			fineV, fineB = lv[i], last[i]
		}
	}
	return fineB
}

// orientation returns quarter turns (counter-clockwise) the page content is
// turned by, from orient.onnx: 0 without the model or when no turn is more
// likely than OrientMin; orientVertical for a page set in vertical columns.
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
	for i := 1; i < 5; i++ {
		if p[i] > p[k] {
			k = i
		}
	}
	if p[k] > OrientMin {
		return k
	}
	return 0
}

// turn applies the quarter turns and the deskew: the page (and the preview,
// for measuring the angle again after a turn that swaps rows and columns).
// Port of _turn.
func (o *OCR) turn(g *Gray, k int, a float64, small *Gray) (*Gray, int, float64) {
	// A half turn keeps the rows, so the angle stands; after a quarter turn
	// the rows were columns, so the angle is measured again on the turned
	// page.
	q := -k
	if k == orientVertical {
		q = 1
		// A quarter turn counter-clockwise: columns become lines, the first
		// (right) column on top, each read left to right.
		g = rot90CCW(g, 1)
	} else if k != 0 {
		g = rot90CCW(g, -k)
	}
	if k == 1 || k == 3 || k == orientVertical {
		s := rot90CCW(small, q)
		a = estimateSkewPy(s)
	}
	if math.Abs(a) >= deskewMin {
		g = rotatePage(g, a)
	} else {
		a = 0
	}
	return g, k, a
}

// Upright turns the page upright and straightens tilt: the preview is
// deskewed, the orientation model reads the straightened preview (a tilted
// page could read as turned), the page is turned and the tilt re-measured
// after a turn that swapped rows and columns.
func (o *OCR) Upright(g *Gray) (*Gray, int, float64) {
	small := o.preview(g, previewSide)
	a := estimateSkewPy(small)
	k := 0
	if math.Abs(a) >= deskewMin {
		k = o.orientation(rotatePage(small, a))
	} else {
		k = o.orientation(small)
	}
	return o.turn(g, k, a, small)
}
