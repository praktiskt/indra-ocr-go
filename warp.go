// Bilinear page rotation for deskew and the inverse box mapping.
package indra

import (
	"math"
	"runtime"
	"sync"
)

// rotatePage rotates counter-clockwise by degrees about the centre, same size,
// bilinear, border replicate. Port of rotate_page.
func rotatePage(g *Gray, degrees float64) *Gray {
	if degrees == 0 {
		out := make([]uint8, len(g.Pix))
		copy(out, g.Pix)
		return &Gray{Pix: out, W: g.W, H: g.H}
	}
	rad := degrees * math.Pi / 180
	cos, sin := math.Cos(rad), math.Sin(rad)
	cx, cy := float64(g.W)/2, float64(g.H)/2
	out := make([]uint8, len(g.Pix))
	workers := runtime.NumCPU()
	if workers > g.H {
		workers = g.H
	}
	if workers <= 1 || g.H < 256 {
		rotateRows(g.Pix, g.W, g.H, out, 0, g.H, cos, sin, cx, cy)
		return &Gray{Pix: out, W: g.W, H: g.H}
	}
	var wg sync.WaitGroup
	rowsPer := (g.H + workers - 1) / workers
	for wi := 0; wi < workers; wi++ {
		y0, y1 := wi*rowsPer, min(g.H, (wi+1)*rowsPer)
		if y0 >= y1 {
			break
		}
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			rotateRows(g.Pix, g.W, g.H, out, y0, y1, cos, sin, cx, cy)
		}(y0, y1)
	}
	wg.Wait()
	return &Gray{Pix: out, W: g.W, H: g.H}
}

func rotateRows(src []uint8, w, h int, out []uint8, y0, y1 int, cos, sin, cx, cy float64) {
	lastX, lastY := w-1, h-1
	for y := y0; y < y1; y++ {
		dy := float64(y) - cy
		ou := y * w
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
				if w == 1 {
					xa, xb = 0, 0
				}
			} else {
				xa, xb, wx0, wx1 = x0, x0+1, 1-fx, fx
			}
			var ya, yb int
			var wy0, wy1 float64
			if y0i < 0 {
				ya, yb, wy0, wy1 = 0, 0, 1, 0
			} else if y0i >= lastY {
				ya, yb, wy0, wy1 = lastY, lastY, 1, 0
				if h == 1 {
					ya, yb = 0, 0
				}
			} else {
				ya, yb, wy0, wy1 = y0i, y0i+1, 1-fy, fy
			}
			a := float64(src[ya*w+xa])
			b := float64(src[ya*w+xb])
			c := float64(src[yb*w+xa])
			d := float64(src[yb*w+xb])
			out[ou+x] = uint8(a*wx0*wy0 + b*wx1*wy0 + c*wx0*wy1 + d*wx1*wy1 + 0.5)
			sxf += cos
			syf += sin
		}
	}
}

// sharpRows fuses rotatePage + row-sum sharpness: no output image alloc.
// tot = sum_y (rowsum[y]-rowsum[y-1])^2 where rowsum uses ink=(255-v)/255.
// Warp coordinates advance incrementally along x (sxf += cos, syf += sin):
// 2 adds/px instead of 4 muls+4 adds. Starts exact per row; drift over a
// row is ~1e-11px, far below the Floor pixel boundary.

// unrotateBox maps a box of the deskewed page back to the page before it.
func unrotateBox(box [4]float64, degrees float64, w, h int) [4]float64 {
	rad := -degrees * math.Pi / 180 // inverse rotation
	cos, sin := math.Cos(rad), math.Sin(rad)
	cx, cy := float64(w)/2, float64(h)/2
	mapPt := func(x, y float64) (float64, float64) {
		dx, dy := x-cx, y-cy
		return cos*dx - sin*dy + cx, sin*dx + cos*dy + cy
	}
	x0, y0, x1, y1 := box[0], box[1], box[2], box[3]
	pts := [4][2]float64{}
	xs := []float64{x0, x1, x0, x1}
	ys := []float64{y0, y0, y1, y1}
	for i := range pts {
		pts[i][0], pts[i][1] = mapPt(xs[i], ys[i])
	}
	nx0, ny0 := pts[0][0], pts[0][1]
	nx1, ny1 := nx0, ny0
	for _, p := range pts[1:] {
		nx0, ny0 = minf(nx0, p[0]), minf(ny0, p[1])
		nx1, ny1 = maxf(nx1, p[0]), maxf(ny1, p[1])
	}
	return [4]float64{
		math.Round(clampf(nx0, 0, float64(w))*10) / 10,
		math.Round(clampf(ny0, 0, float64(h))*10) / 10,
		math.Round(clampf(nx1, 0, float64(w))*10) / 10,
		math.Round(clampf(ny1, 0, float64(h))*10) / 10,
	}
}

func rot90CCW(g *Gray, k int) *Gray {
	k = ((k % 4) + 4) % 4
	out := g
	for i := 0; i < k; i++ {
		nw, nh := out.H, out.W
		px := make([]uint8, nw*nh)
		for y := 0; y < nh; y++ {
			for x := 0; x < nw; x++ {
				// CCW: dst(x,y) = src(W-1-y, x)
				px[y*nw+x] = out.Pix[x*out.W+(out.W-1-y)]
			}
		}
		out = &Gray{Pix: px, W: nw, H: nh}
	}
	return out
}
