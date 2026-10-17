package indra

import (
	"math"
)

// sharpRowsDark fuses rotate + row-sum sharpness where out-of-source samples
// read BLACK (cv2's default border for warpAffine, used by pipeline.py's
// estimate_skew): black pixels contribute the full ink 255/255.
func sharpRowsDark(src []uint8, w, h int, cos, sin, cx, cy float64, stride int) float64 {
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
				return float64(src[yi*w+xi])
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

// estimateSkewPy is the blend of pipeline.py's estimator (800px thumbnail,
// 0.5 then 0.1 degree searches, black border) fused into a single bilinear
// pass with no rotated-image allocation, parallelized over the angle grid.
func estimateSkewPy(g *Gray) float64 {
	const side = 800
	s := float64(side) / float64(max(g.W, g.H))
	sm := g.resizeArea(max(1, int(math.Round(float64(g.W)*s))), max(1, int(math.Round(float64(g.H)*s))))
	sum := 0.0
	for _, p := range sm.Pix {
		sum += float64(255 - p)
	}
	if sum/float64(len(sm.Pix))/255 < 0.003 {
		return 0
	}
	cx, cy := float64(sm.W)/2, float64(sm.H)/2
	// Coarse search at stride 2 (same per-angle scale, argmax preserved to
	// tie flips); the decisive fine search runs stride 1 at full resolution.
	sharp := func(a float64) float64 {
		rad := a * math.Pi / 180
		return sharpRowsStride(sm.Pix, sm.W, sm.H, math.Cos(rad), math.Sin(rad), cx, cy, 2)
	}
	sharpCoarse := func(a float64) float64 {
		rad := a * math.Pi / 180
		return sharpRowsStride(sm.Pix, sm.W, sm.H, math.Cos(rad), math.Sin(rad), cx, cy, 4)
	}
	evalAngles := func(angles []float64, fn func(a float64) float64) []float64 {
		vals := make([]float64, len(angles))
		for i, a := range angles {
			vals[i] = fn(a)
		}
		return vals
	}
	// Python coarse grid: 0.5-degree steps over [-5, 5].
	var coarse []float64
	for a := -5.0; a <= 5.0001; a += 0.5 {
		coarse = append(coarse, a)
	}
	cv := evalAngles(coarse, sharpCoarse)
	best, bestV := coarse[0], cv[0]
	for i := range coarse {
		if cv[i] > bestV {
			bestV, best = cv[i], coarse[i]
		}
	}
	// Python fine grid: 0.1-degree steps over best +/- 0.5.
	var fine []float64
	for a := best - 0.5; a <= best+0.5001; a += 0.1 {
		fine = append(fine, a)
	}
	fv := evalAngles(fine, sharp)
	fineB, fineV := best, bestV
	for i := range fine {
		if fv[i] > fineV {
			fineV, fineB = fv[i], fine[i]
		}
	}
	return fineB
}
