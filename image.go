// Grayscale image ops backing the pipeline. Port of the cv2/numpy bits in
// det_post.py, rec_preproc.py, pipeline.py and line_mask.py. Pure Go:
// bilinear upscale, area-average downscale, bilinear warp for deskew.
package indra

import (
	"image"
	"math"
	"runtime"
	"sort"
	"sync"
)

// Gray is a uint8 grayscale page, row-major.
type Gray struct {
	Pix []uint8
	W   int
	H   int
}

// NewGray wraps px (len W*H).
func NewGray(px []uint8, w, h int) *Gray { return &Gray{Pix: px, W: w, H: h} }

// FromImage converts any image.Image to gray (ITU-R 601 luma).
// Use this for decoded files (JPEG decodes to YCbCr, not RGBA).
func FromImage(img image.Image) *Gray {
	switch m := img.(type) {
	case *image.Gray:
		b := m.Bounds()
		w, h := b.Dx(), b.Dy()
		out := make([]uint8, w*h)
		if b.Min.X == 0 && b.Min.Y == 0 && m.Stride == w {
			copy(out, m.Pix)
			return &Gray{Pix: out, W: w, H: h}
		}
		for y := 0; y < h; y++ {
			copy(out[y*w:(y+1)*w], m.Pix[(b.Min.Y+y)*m.Stride+b.Min.X:(b.Min.Y+y)*m.Stride+b.Min.X+w])
		}
		return &Gray{Pix: out, W: w, H: h}
	case *image.RGBA:
		return FromRGBA(m)
	case *image.NRGBA:
		b := m.Bounds()
		w, h := b.Dx(), b.Dy()
		out := make([]uint8, w*h)
		for y := 0; y < h; y++ {
			ro := (b.Min.Y+y)*m.Stride + b.Min.X*4
			wo := y * w
			for x := 0; x < w; x++ {
				r, g, bl := m.Pix[ro], m.Pix[ro+1], m.Pix[ro+2]
				out[wo+x] = uint8((299*int(r) + 587*int(g) + 114*int(bl) + 500) / 1000)
				ro += 4
			}
		}
		return &Gray{Pix: out, W: w, H: h}
	case *image.YCbCr:
		return fromYCbCr(m)
	case *image.Gray16:
		b := m.Bounds()
		w, h := b.Dx(), b.Dy()
		out := make([]uint8, w*h)
		for y := 0; y < h; y++ {
			ro := (b.Min.Y+y)*m.Stride + b.Min.X*2
			wo := y * w
			for x := 0; x < w; x++ {
				out[wo+x] = m.Pix[ro] // high byte = >>8
				ro += 2
			}
		}
		return &Gray{Pix: out, W: w, H: h}
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			out[y*w+x] = uint8((299*(r>>8) + 587*(g>>8) + 114*(bl>>8) + 500) / 1000)
		}
	}
	return &Gray{Pix: out, W: w, H: h}
}

func fromYCbCr(m *image.YCbCr) *Gray {
	b := m.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]uint8, w*h)
	switch m.SubsampleRatio {
	case image.YCbCrSubsampleRatio444:
		for y := 0; y < h; y++ {
			yo := (b.Min.Y+y)*m.YStride + b.Min.X
			co := (b.Min.Y+y)*m.CStride + b.Min.X
			wo := y * w
			for x := 0; x < w; x++ {
				out[wo+x] = m.Y[yo+x] // luma direct, no color math
				_ = m.Cb[co+x]
			}
		}
	default:
		// 422 / 420 / 411: luma plane subsampled in chroma only.
		for y := 0; y < h; y++ {
			yo := (b.Min.Y+y)*m.YStride + b.Min.X
			copy(out[y*w:(y+1)*w], m.Y[yo:yo+w])
		}
	}
	return &Gray{Pix: out, W: w, H: h}
}

// FromRGBA converts an image.RGBA to gray (ITU-R 601 luma).
func FromRGBA(img *image.RGBA) *Gray {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			o := img.PixOffset(b.Min.X+x, b.Min.Y+y)
			r, g, bl := img.Pix[o], img.Pix[o+1], img.Pix[o+2]
			out[y*w+x] = uint8((299*int(r) + 587*int(g) + 114*int(bl) + 500) / 1000)
		}
	}
	return &Gray{Pix: out, W: w, H: h}
}

// At returns the pixel with edge clamping.
func (g *Gray) At(x, y int) uint8 {
	if x < 0 {
		x = 0
	}
	if x >= g.W {
		x = g.W - 1
	}
	if y < 0 {
		y = 0
	}
	if y >= g.H {
		y = g.H - 1
	}
	return g.Pix[y*g.W+x]
}

// Resize scales to nw x nh, choosing the filter like cv2: area-average when
// shrinking, bilinear otherwise. (Callers that must mirror an explicit cv2
// choice use ResizeArea/ResizeLinear directly.)
func (g *Gray) Resize(nw, nh int) *Gray {
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
		return g.ResizeArea(nw, nh)
	}
	return g.ResizeLinear(nw, nh)
}

// ResizeArea is exact box-area resampling (cv2 INTER_AREA): each dst pixel is
// the overlap-weighted mean of the source rect it covers.
func (g *Gray) ResizeArea(nw, nh int) *Gray {
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
				resizeAreaRows(src, sw, sh, out, nw, y0, y1, sy, sx0s, sx1s, xx0s, xx1s)
			}(y0, y1)
		}
		wg.Wait()
		return &Gray{Pix: out, W: nw, H: nh}
	}
	resizeAreaRows(src, sw, sh, out, nw, 0, nh, sy, sx0s, sx1s, xx0s, xx1s)
	return &Gray{Pix: out, W: nw, H: nh}
}

func resizeAreaRows(src []uint8, sw, sh int, out []uint8, nw, y0, y1 int, sy float64, sx0s, sx1s []float64, xx0s, xx1s []int) {
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
}

// ResizeLinear is half-pixel bilinear with edge replicate (matches cv2).
func (g *Gray) ResizeLinear(nw, nh int) *Gray {
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

const inputMultiple = 32

// PageToInput resizes so the long side is longSide and pads to a multiple of
// 32 with white. Returns the float32 ink map (0=paper), padded size and scale.
// Port of page_to_input in det_post.py.
func PageToInput(g *Gray, longSide int) (ink []float32, pw, ph int, scale float64) {
	scale = float64(longSide) / float64(max(g.W, g.H))
	nw := max(1, int(math.Round(float64(g.W)*scale)))
	nh := max(1, int(math.Round(float64(g.H)*scale)))
	var rs *Gray
	switch {
	case nw == g.W && nh == g.H:
		rs = g
	case scale < 1:
		rs = g.ResizeArea(nw, nh)
	default:
		rs = g.ResizeLinear(nw, nh)
	}
	pw = ((nw + inputMultiple - 1) / inputMultiple) * inputMultiple
	ph = ((nh + inputMultiple - 1) / inputMultiple) * inputMultiple
	ink = make([]float32, pw*ph) // zero = paper
	fillInk(ink, pw, rs.Pix, rs.W, nw, nh)
	return ink, pw, ph, scale
}

// fillInk writes the ink map (1-p/255) for the nw x nh top-left region.
// Rows are independent; sharded across CPUs for page-sized inputs.
func fillInk(ink []float32, pw int, src []uint8, sw, nw, nh int) {
	if nh < 256 {
		for y := 0; y < nh; y++ {
			ou, iu := y*pw, y*sw
			for x := 0; x < nw; x++ {
				ink[ou+x] = 1 - float32(src[iu+x])/255
			}
		}
		return
	}
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
			for y := y0; y < y1; y++ {
				ou, iu := y*pw, y*sw
				for x := 0; x < nw; x++ {
					ink[ou+x] = 1 - float32(src[iu+x])/255
				}
			}
		}(y0, y1)
	}
	wg.Wait()
}

// Line crop geometry, port of rec_preproc.py.
const (
	LineHeight    = 32
	PadY          = 0.2
	PadX          = 0.3
	WidthMultiple = 8
	MaxLineWidth  = 4096
)

// CropLine crops a line box (ink box, page px) with PAD_X/PAD_Y margins and
// resizes to height px high, keeping aspect ratio. Returns ink map [height x W].
func CropLine(g *Gray, box [4]float64, height int) []float32 {
	x0, y0, x1, y1 := box[0], box[1], box[2], box[3]
	h := maxf(1, y1-y0)
	X0, X1 := x0-PadX*h, x1+PadX*h
	Y0, Y1 := y0-PadY*h, y1+PadY*h
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
	outW := max(WidthMultiple, min(MaxLineWidth, int(math.Round(float64(cw)*float64(height)/float64(ch)))))
	var rs *Gray
	if ch > height {
		rs = src.ResizeArea(outW, height)
	} else {
		rs = src.ResizeLinear(outW, height)
	}
	out := make([]float32, outW*height)
	for i, p := range rs.Pix {
		out[i] = 1 - float32(p)/255
	}
	return out
}

// PadBatch right-pads ink maps to one width. Returns [B*H*Wm] NCHW-flattened
// (C=1) plus widths.
func PadBatch(lines [][]float32, heights, widths []int) (batched []float32, wOut []int64, wm int) {
	wm = 0
	for _, w := range widths {
		if w > wm {
			wm = w
		}
	}
	wm = ((wm + WidthMultiple - 1) / WidthMultiple) * WidthMultiple
	h := heights[0]
	batched = make([]float32, len(lines)*h*wm)
	wOut = make([]int64, len(lines))
	for i, ln := range lines {
		w := widths[i]
		for r := 0; r < h; r++ {
			copy(batched[(i*h+r)*wm:(i*h+r)*wm+w], ln[r*w:(r+1)*w])
		}
		wOut[i] = int64(w)
	}
	return batched, wOut, wm
}

// WidthBatches groups line indices narrowest-first so little compute goes to
// padding. Port of width_batches in rec_preproc.py.
func WidthBatches(widths []int, maxBatch int) [][]int {
	order := make([]int, len(widths))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return widths[order[a]] < widths[order[b]] })
	const maxRatio = 1.3
	const minWidth = 64
	const maxArea = 64 * 640
	var batches [][]int
	for _, idx := range order {
		if n := len(batches); n > 0 {
			last := batches[n-1]
			first := last[0]
			if len(last) < maxBatch &&
				max(widths[idx], minWidth) <= int(float64(max(widths[first], minWidth))*maxRatio) &&
				(len(last)+1)*max(widths[idx], minWidth) <= maxArea {
				batches[n-1] = append(last, idx)
				continue
			}
		}
		batches = append(batches, []int{idx})
	}
	return batches
}

// OrientInput builds the [side,side] ink map for orient.onnx. Port of orient_input.
func OrientInput(g *Gray, side int) []float32 {
	s := float64(side) / float64(max(g.W, g.H))
	nw := max(1, int(math.Round(float64(g.W)*s)))
	nh := max(1, int(math.Round(float64(g.H)*s)))
	sm := g.ResizeArea(nw, nh)
	out := make([]float32, side*side)
	fillInk(out, side, sm.Pix, sm.W, nw, nh)
	return out
}

// RotatePage rotates counter-clockwise by degrees about the centre, same size,
// bilinear, border replicate. Port of rotate_page.
func RotatePage(g *Gray, degrees float64) *Gray {
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

// sharpRows fuses RotatePage + row-sum sharpness: no output image alloc.
// tot = sum_y (rowsum[y]-rowsum[y-1])^2 where rowsum uses ink=(255-v)/255.
// Warp coordinates advance incrementally along x (sxf += cos, syf += sin):
// 2 adds/px instead of 4 muls+4 adds. Starts exact per row; drift over a
// row is ~1e-11px, far below the Floor pixel boundary.
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

// EstimateSkew returns degrees CCW that straighten text rows. Port of estimate_skew.
func EstimateSkew(g *Gray) float64 {
	const side = 600
	s := float64(side) / float64(max(g.W, g.H))
	sm := g.ResizeArea(max(1, int(math.Round(float64(g.W)*s))), max(1, int(math.Round(float64(g.H)*s))))
	sum := 0.0
	for _, p := range sm.Pix {
		sum += float64(255 - p)
	}
	if sum/float64(len(sm.Pix))/255 < 0.003 {
		return 0
	}
	// Coarse pass on a half-size thumbnail with stride-2 sampling (~1/8 the
	// pixels); mid+fine passes stay on the full 600px thumbnail for precision.
	sm2 := sm.ResizeArea(max(1, sm.W/2), max(1, sm.H/2))
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

// UnrotateBox maps a box of the deskewed page back to the page before it.
func UnrotateBox(box [4]float64, degrees float64, w, h int) [4]float64 {
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

func Rot90CCW(g *Gray, k int) *Gray {
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

func maxf(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func minf(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func clampf(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
