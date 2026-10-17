// Resampling: area-average downscale, bilinear upscale (cv2.INTER_AREA/LINEAR).
package indra

import (
	"math"
	"runtime"
	"sync"
)

func (g *Gray) resize(nw, nh int) *Gray {
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	if nw == g.W && nh == g.H {
		out := make([]uint8, len(g.Pix))
		copy(out, g.Pix)
		return &Gray{Pix: out, W: nw, H: nh}
	}
	if nw < g.W || nh < g.H {
		return g.resizeArea(nw, nh)
	}
	return g.resizeLinear(nw, nh)
}

// areaTable holds precomputed per-destination-column source weights so each
// row only multiplies and adds. Products and accumulation order match the
// original per-term float ops exactly (w = wx*wy, interior columns weigh
// wxL=xx+1-xx=1.0 by construction), so outputs are bit-identical.
type areaTable struct {
	x0, x1           []int
	wxL, wxT, single []float64
}

func buildAreaTable(sw int, sx0s, sx1s []float64, xx0s, xx1s []int, wxLs, wxTs, wxSs []float64) *areaTable {
	return &areaTable{x0: xx0s, x1: xx1s, wxL: wxLs, wxT: wxTs, single: wxSs}
}

// resizeArea is exact box-area resampling (cv2 INTER_AREA): each dst pixel is
// the overlap-weighted mean of the source rect it covers.
func (g *Gray) resizeArea(nw, nh int) *Gray {
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	out := make([]uint8, nw*nh)
	sx := float64(g.W) / float64(nw)
	sy := float64(g.H) / float64(nh)
	src := g.Pix
	sw := g.W
	sh := g.H
	// X source spans depend only on the column: precompute once so the
	// per-row worker does not redo them for every y (identical values).
	xx0s := make([]int, nw)
	xx1s := make([]int, nw)
	wxLs := make([]float64, nw)
	wxTs := make([]float64, nw)
	wxSs := make([]float64, nw)
	sx0s := make([]float64, nw)
	sx1s := make([]float64, nw)
	for x := 0; x < nw; x++ {
		sx0 := float64(x) * sx
		sx1 := float64(x+1) * sx
		xx0 := int(sx0)
		xx1 := int(sx1)
		if float64(xx1) < sx1 {
			xx1++
		}
		if xx0 < 0 {
			xx0 = 0
		}
		if xx1 > sw {
			xx1 = sw
		}
		xx0s[x], xx1s[x] = xx0, xx1
		sx0s[x], sx1s[x] = sx0, sx1
		if xx1-xx0 <= 1 {
			wxSs[x] = min(min(sx1, float64(xx0)+1), float64(sw)) - max(sx0, float64(xx0))
			wxLs[x], wxTs[x] = 0, 0
			continue
		}
		wxLs[x] = min(sx1, float64(xx0)+1) - max(sx0, float64(xx0))
		wxTs[x] = min(sx1, float64(xx1)) - float64(xx1-1)
	}
	t := buildAreaTable(sw, sx0s, sx1s, xx0s, xx1s, wxLs, wxTs, wxSs)
	// Shard rows across CPUs for page-sized inputs.
	// Small heights: sequential to avoid goroutine overhead.
	if nh >= 256 {
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
				resizeAreaRows(src, sw, sh, out, nw, y0, y1, sy, t)
			}(y0, y1)
		}
		wg.Wait()
		return &Gray{Pix: out, W: nw, H: nh}
	}
	resizeAreaRows(src, sw, sh, out, nw, 0, nh, sy, t)
	return &Gray{Pix: out, W: nw, H: nh}
}

// resizeAreaRows replaces the per-pixel weight recomputation with precomputed
// per-column weights. Products and accumulation order match the original
// per-term float ops exactly (wx*wy, interior cols weigh wxL=xx+1-xx=1.0),
// so outputs are bit-identical for the same input.
func resizeAreaRows(src []uint8, sw, sh int, out []uint8, nw, y0, y1 int, sy float64, t *areaTable) {
	sums := make([]float64, nw)
	wsums := make([]float64, nw)
	for y := y0; y < y1; y++ {
		sy0 := float64(y) * sy
		sy1 := float64(y+1) * sy
		yy0 := int(sy0)
		yy1 := int(sy1)
		if float64(yy1) < sy1 {
			yy1++
		}
		if yy0 < 0 {
			yy0 = 0
		}
		if yy1 > sh {
			yy1 = sh
		}
		ou := y * nw
		for yy := yy0; yy < yy1; yy++ {
			// wy matches the original per-yy computation bit-for-bit.
			oy0 := sy0
			if float64(yy) > oy0 {
				oy0 = float64(yy)
			}
			oy1 := sy1
			if float64(yy+1) < oy1 {
				oy1 = float64(yy + 1)
			}
			wy := oy1 - oy0
			if wy <= 0 {
				continue
			}
			base := yy * sw
			for x := 0; x < nw; x++ {
				if w := t.single[x]; w != 0 {
					wsums[x] += w * wy
					sums[x] += float64(src[base+t.x0[x]]) * (w * wy)
					continue
				}
				wL := t.wxL[x] * wy
				wT := t.wxT[x] * wy
				wsums[x] += wL
				sums[x] += float64(src[base+t.x0[x]]) * wL
				for xx := t.x0[x] + 1; xx < t.x1[x]-1; xx++ {
					wsums[x] += wy
					sums[x] += float64(src[base+xx]) * wy
				}
				wsums[x] += wT
				sums[x] += float64(src[base+t.x1[x]-1]) * wT
			}
		}
		for x := 0; x < nw; x++ {
			out[ou+x] = uint8(sums[x]/wsums[x] + 0.5)
			sums[x] = 0
			wsums[x] = 0
		}
	}
}

// resizeLinear is half-pixel bilinear with edge replicate (matches cv2).
func (g *Gray) resizeLinear(nw, nh int) *Gray {
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	out := make([]uint8, nw*nh)
	sx := float64(g.W) / float64(nw)
	sy := float64(g.H) / float64(nh)
	src := g.Pix
	sw, sh := g.W, g.H
	if nh >= 256 {
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
				resizeLinearRows(src, sw, sh, out, nw, y0, y1, sx, sy)
			}(y0, y1)
		}
		wg.Wait()
		return &Gray{Pix: out, W: nw, H: nh}
	}
	resizeLinearRows(src, sw, sh, out, nw, 0, nh, sx, sy)
	return &Gray{Pix: out, W: nw, H: nh}
}

func resizeLinearRows(src []uint8, sw, sh int, out []uint8, nw, y0, y1 int, sx, sy float64) {
	lastX, lastY := sw-1, sh-1
	for y := y0; y < y1; y++ {
		fy := (float64(y)+0.5)*sy - 0.5
		y0i := int(math.Floor(fy))
		dy := fy - float64(y0i)
		if y0i < 0 {
			y0i, dy = 0, 0
		} else if y0i >= lastY {
			y0i, dy = lastY, 1
			if sh == 1 {
				y0i, dy = 0, 0
			}
		}
		r0 := y0i * sw
		r1 := r0
		if y0i+1 < sh {
			r1 = (y0i + 1) * sw
		}
		ou := y * nw
		wy0, wy1 := 1-dy, dy
		// fx advances by sx per x step: 1 add/px instead of mult+adds.
		// Starts exact; drift ~1e-12px over a line width, far below Floor.
		fx := 0.5*sx - 0.5
		for x := 0; x < nw; x++ {
			x0i := int(math.Floor(fx))
			dx := fx - float64(x0i)
			if x0i < 0 {
				x0i, dx = 0, 0
			} else if x0i >= lastX {
				x0i, dx = lastX, 1
				if sw == 1 {
					x0i, dx = 0, 0
				}
			}
			a := float64(src[r0+x0i])
			b := float64(src[r0+x0i+boolToInt(x0i+1 < sw)])
			c := float64(src[r1+x0i])
			d := float64(src[r1+x0i+boolToInt(x0i+1 < sw)])
			wx0, wx1 := 1-dx, dx
			out[ou+x] = uint8(a*wx0*wy0 + b*wx1*wy0 + c*wx0*wy1 + d*wx1*wy1 + 0.5)
			fx += sx
		}
	}
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
