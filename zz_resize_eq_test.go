package indra

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// Temp equivalence check: new tabled ResizeArea must be bit-identical to the
// original per-pixel implementation on real corpus pages, for the sizes the
// pipeline uses. Dumps bit-equality counts to stderr.
func TestZZResizeEquivalence(t *testing.T) {
	if os.Getenv("RESIZE_EQ") == "" {
		t.Skip("RESIZE_EQ not set")
	}
	dir := os.Getenv("PAGES")
	files, _ := filepath.Glob(filepath.Join(dir, "*.png"))
	sort.Strings(files)
	resizeOld := func(g *Gray, nw, nh int) *Gray {
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
		xx0s := make([]int, nw)
		xx1s := make([]int, nw)
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
		}
		for y := 0; y < nh; y++ {
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
			for x := 0; x < nw; x++ {
				sx0, sx1 := sx0s[x], sx1s[x]
				xx0, xx1 := xx0s[x], xx1s[x]
				sum, wsum := 0.0, 0.0
				for yy := yy0; yy < yy1; yy++ {
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
					for xx := xx0; xx < xx1; xx++ {
						ox0 := sx0
						if float64(xx) > ox0 {
							ox0 = float64(xx)
						}
						ox1 := sx1
						if float64(xx+1) < ox1 {
							ox1 = float64(xx + 1)
						}
						wx := ox1 - ox0
						if wx <= 0 {
							continue
						}
						w := wx * wy
						sum += float64(src[base+xx]) * w
						wsum += w
					}
				}
				out[ou+x] = uint8(sum/wsum + 0.5)
			}
		}
		return &Gray{Pix: out, W: nw, H: nh}
	}
	tot := map[string]int{}
	for _, f := range files {
		data, _ := os.ReadFile(f)
		img, _ := png.Decode(bytes.NewReader(data))
		g := FromImage(img)
		for _, spec := range [][2]int{
			{max(1, g.W/2), max(1, g.H/2)},
			{int(float64(g.W) * 600.0 / float64(max(g.W, g.H))), int(float64(g.H) * 600.0 / float64(max(g.W, g.H)))},
			{1024, 1024}, // det-like; PageToInput rounds: nw,nh from scale
		} {
			nw, nh := spec[0], spec[1]
			if nw < 1 || nh < 1 || nw > 4096 || nh > 4096 {
				continue
			}
			// det-like exact shadow of PageToInput rounding
			scale := float64(1024) / float64(max(g.W, g.H))
			dw := max(1, int(float64(g.W)*scale+0.5))
			dh := max(1, int(float64(g.H)*scale+0.5))
			a := resizeOld(g, dw, dh)
			b := g.ResizeArea(dw, dh)
			eq := true
			for i := range a.Pix {
				if a.Pix[i] != b.Pix[i] {
					eq = false
					break
				}
			}
			tot["det"] += boolInt(eq)
			_ = nw
			_ = nh
			// skew thumb
			s := 600.0 / float64(max(g.W, g.H))
			sw := max(1, int(float64(g.W)*s+0.5))
			sh := max(1, int(float64(g.H)*s+0.5))
			ao := resizeOld(g, sw, sh)
			bo := g.ResizeArea(sw, sh)
			eq2 := true
			for i := range ao.Pix {
				if ao.Pix[i] != bo.Pix[i] {
					eq2 = false
					break
				}
			}
			tot["skewthumb"] += boolInt(eq2)
			// half
			hw, hh := max(1, g.W/2), max(1, g.H/2)
			ah := resizeOld(g, hw, hh)
			bh := g.ResizeArea(hw, hh)
			eq3 := true
			for i := range ah.Pix {
				if ah.Pix[i] != bh.Pix[i] {
					eq3 = false
					break
				}
			}
			tot["half"] += boolInt(eq3)
		}
	}
	enc, _ := json.Marshal(tot)
	fmt.Println("resize-eq:", string(enc))
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
