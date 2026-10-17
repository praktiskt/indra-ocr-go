// Skew estimation by projection sharpness (estimate_skew, the 600px port).
package indra

import (
	"math"
	"runtime"
	"sync"
)

func sharpRows(src []uint8, w, h int, cos, sin, cx, cy float64) float64 {
	sums := make([]float64, h)
	inv := 1.0 / 255.0
	lastX, lastY := w-1, h-1
	for y := 0; y < h; y++ {
		dy := float64(y) - cy
		s := 0.0
		sxf := cos*(-cx) - sin*dy + cx
		syf := sin*(-cx) + cos*dy + cy
		for x := 0; x < w; x++ {
			x0 := int(math.Floor(sxf))
			y0i := int(math.Floor(syf))
			fx, fy := sxf-float64(x0), syf-float64(y0i)
			var xa, xb int
			var wx0, wx1 float64
			if x0 < 0 {
				xa, xb, wx0, wx1 = 0, 0, 1, 0
			} else if x0 >= lastX {
				xa, xb, wx0, wx1 = lastX, lastX, 1, 0
			} else {
				xa, xb, wx0, wx1 = x0, x0+1, 1-fx, fx
			}
			var ya, yb int
			var wy0, wy1 float64
			if y0i < 0 {
				ya, yb, wy0, wy1 = 0, 0, 1, 0
			} else if y0i >= lastY {
				ya, yb, wy0, wy1 = lastY, lastY, 1, 0
			} else {
				ya, yb, wy0, wy1 = y0i, y0i+1, 1-fy, fy
			}
			a := float64(src[ya*w+xa])
			b := float64(src[ya*w+xb])
			c := float64(src[yb*w+xa])
			d := float64(src[yb*w+xb])
			v := a*wx0*wy0 + b*wx1*wy0 + c*wx0*wy1 + d*wx1*wy1
			s += 255 - v
			sxf += cos
			syf += sin
		}
		sums[y] = s * inv
	}
	tot := 0.0
	for y := 1; y < h; y++ {
		d := sums[y] - sums[y-1]
		tot += d * d
	}
	return tot
}

// sharpRowsStride is sharpRows sampling every stride-th source pixel along
// the scanline: the row profile is scaled by the identical factor for every
// angle, so argmax over angles is unchanged while the cost drops by stride.
func sharpRowsStride(src []uint8, w, h int, cos, sin, cx, cy float64, stride int) float64 {
	sums := make([]float64, h)
	inv := 1.0 / 255.0
	lastX, lastY := w-1, h-1
	for y := 0; y < h; y++ {
		dy := float64(y) - cy
		s := 0.0
		sxf := cos*(-cx) - sin*dy + cx
		syf := sin*(-cx) + cos*dy + cy
		for x := 0; x < w; x += stride {
			x0 := int(math.Floor(sxf))
			y0i := int(math.Floor(syf))
			fx, fy := sxf-float64(x0), syf-float64(y0i)
			var xa, xb int
			var wx0, wx1 float64
			if x0 < 0 {
				xa, xb, wx0, wx1 = 0, 0, 1, 0
			} else if x0 >= lastX {
				xa, xb, wx0, wx1 = lastX, lastX, 1, 0
			} else {
				xa, xb, wx0, wx1 = x0, x0+1, 1-fx, fx
			}
			var ya, yb int
			var wy0, wy1 float64
			if y0i < 0 {
				ya, yb, wy0, wy1 = 0, 0, 1, 0
			} else if y0i >= lastY {
				ya, yb, wy0, wy1 = lastY, lastY, 1, 0
			} else {
				ya, yb, wy0, wy1 = y0i, y0i+1, 1-fy, fy
			}
			a := float64(src[ya*w+xa])
			b := float64(src[ya*w+xb])
			c := float64(src[yb*w+xa])
			d := float64(src[yb*w+xb])
			v := a*wx0*wy0 + b*wx1*wy0 + c*wx0*wy1 + d*wx1*wy1
			s += 255 - v
			sxf += cos * float64(stride)
			syf += sin * float64(stride)
		}
		sums[y] = s * inv
	}
	tot := 0.0
	for y := 1; y < h; y++ {
		d := sums[y] - sums[y-1]
		tot += d * d
	}
	return tot
}

// estimateSkew returns degrees CCW that straighten text rows. Port of estimate_skew.
func estimateSkew(g *Gray) float64 {
	const side = 600
	s := float64(side) / float64(max(g.W, g.H))
	sm := g.resizeArea(max(1, int(math.Round(float64(g.W)*s))), max(1, int(math.Round(float64(g.H)*s))))
	sum := 0.0
	for _, p := range sm.Pix {
		sum += float64(255 - p)
	}
	if sum/float64(len(sm.Pix))/255 < 0.003 {
		return 0
	}
	// Coarse pass on a half-size thumbnail with stride-2 sampling (~1/8 the
	// pixels); mid+fine passes stay on the full 600px thumbnail for precision.
	sm2 := sm.resizeArea(max(1, sm.W/2), max(1, sm.H/2))
	cx2, cy2 := float64(sm2.W)/2, float64(sm2.H)/2
	cx, cy := float64(sm.W)/2, float64(sm.H)/2
	sharpCoarse := func(a float64) float64 {
		rad := a * math.Pi / 180
		return sharpRowsStride(sm2.Pix, sm2.W, sm2.H, math.Cos(rad), math.Sin(rad), cx2, cy2, 2)
	}
	sharp := func(a float64) float64 {
		rad := a * math.Pi / 180
		return sharpRows(sm.Pix, sm.W, sm.H, math.Cos(rad), math.Sin(rad), cx, cy)
	}
	sharpMid := func(a float64) float64 {
		rad := a * math.Pi / 180
		return sharpRowsStride(sm.Pix, sm.W, sm.H, math.Cos(rad), math.Sin(rad), cx, cy, 2)
	}
	evalAllCoarse := func(angles []float64) []float64 {
		vals := make([]float64, len(angles))
		var wg sync.WaitGroup
		for i, a := range angles {
			wg.Add(1)
			go func(i int, a float64) {
				defer wg.Done()
				vals[i] = sharpCoarse(a)
			}(i, a)
		}
		wg.Wait()
		return vals
	}
	evalAllCoarse2 := func(angles []float64) []float64 {
		vals := make([]float64, len(angles))
		var wg sync.WaitGroup
		for i, a := range angles {
			wg.Add(1)
			go func(i int, a float64) {
				defer wg.Done()
				vals[i] = sharpMid(a)
			}(i, a)
		}
		wg.Wait()
		return vals
	}
	evalAll := func(angles []float64) []float64 {
		vals := make([]float64, len(angles))
		if len(angles) < 4 {
			for i, a := range angles {
				vals[i] = sharp(a)
			}
			return vals
		}
		var wg sync.WaitGroup
		sem := make(chan struct{}, runtime.NumCPU())
		for i, a := range angles {
			wg.Add(1)
			sem <- struct{}{}
			go func(i int, a float64) {
				defer wg.Done()
				vals[i] = sharp(a)
				<-sem
			}(i, a)
		}
		wg.Wait()
		return vals
	}
	var coarse []float64
	for a := -5.0; a <= 5.000001; a += 1.0 {
		coarse = append(coarse, a)
	}
	cv := evalAllCoarse(coarse)
	best, bestV := coarse[0], cv[0]
	for i := range coarse {
		if cv[i] > bestV {
			bestV, best = cv[i], coarse[i]
		}
	}
	var mid []float64
	for a := best - 1.0; a <= best+1.000001; a += 0.5 {
		mid = append(mid, a)
	}
	mv := evalAllCoarse2(mid)
	midB, midV := best, bestV
	for i := range mid {
		if mv[i] > midV {
			midV, midB = mv[i], mid[i]
		}
	}
	var fine []float64
	for a := midB - 0.5; a <= midB+0.500001; a += 0.1 {
		fine = append(fine, a)
	}
	fv := evalAll(fine)
	fineB, fineV := midB, midV
	for i := range fine {
		if fv[i] > fineV {
			fineV, fineB = fv[i], fine[i]
		}
	}
	return fineB
}
