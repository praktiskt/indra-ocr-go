// Grayscale image ops backing the pipeline. Port of the cv2/numpy bits in
// det_post.py, rec_preproc.py, pipeline.py and line_mask.py. Pure Go:
// bilinear upscale, area-average downscale, bilinear warp for deskew.
package indra

import (
	"image"
	"math"
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
	for y := 0; y < nh; y++ {
		y0, y1 := float64(y)*sy, float64(y+1)*sy
		for x := 0; x < nw; x++ {
			x0, x1 := float64(x)*sx, float64(x+1)*sx
			sum, wsum := 0.0, 0.0
			for yy := int(y0); float64(yy) < y1; yy++ {
				oy0 := math.Max(y0, float64(yy))
				oy1 := math.Min(y1, float64(yy+1))
				if oy1 <= oy0 {
					continue
				}
				for xx := int(x0); float64(xx) < x1; xx++ {
					ox0 := math.Max(x0, float64(xx))
					ox1 := math.Min(x1, float64(xx+1))
					if ox1 <= ox0 {
						continue
					}
					w := (ox1 - ox0) * (oy1 - oy0)
					sum += float64(g.At(xx, yy)) * w
					wsum += w
				}
			}
			out[y*nw+x] = uint8(math.Round(sum / wsum))
		}
	}
	return &Gray{Pix: out, W: nw, H: nh}
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
	for y := 0; y < nh; y++ {
		fy := (float64(y)+0.5)*sy - 0.5
		y0 := int(math.Floor(fy))
		dy := fy - float64(y0)
		for x := 0; x < nw; x++ {
			fx := (float64(x)+0.5)*sx - 0.5
			x0 := int(math.Floor(fx))
			dx := fx - float64(x0)
			a := float64(g.At(x0, y0))
			b := float64(g.At(x0+1, y0))
			c := float64(g.At(x0, y0+1))
			d := float64(g.At(x0+1, y0+1))
			v := a*(1-dx)*(1-dy) + b*dx*(1-dy) + c*(1-dx)*dy + d*dx*dy
			out[y*nw+x] = uint8(math.Round(v))
		}
	}
	return &Gray{Pix: out, W: nw, H: nh}
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
	for y := 0; y < nh; y++ {
		for x := 0; x < nw; x++ {
			ink[y*pw+x] = 1 - float32(rs.Pix[y*rs.W+x])/255
		}
	}
	return ink, pw, ph, scale
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
	// Build crop with white outside the page.
	crop := make([]uint8, cw*ch)
	for i := range crop {
		crop[i] = 255
	}
	sx0, sy0 := max(0, ix0), max(0, iy0)
	sx1, sy1 := min(g.W, ix1), min(g.H, iy1)
	for y := sy0; y < sy1; y++ {
		copy(crop[(y-iy0)*cw+(sx0-ix0):(y-iy0)*cw+(sx1-ix0)], g.Pix[y*g.W+sx0:y*g.W+sx1])
	}
	src := &Gray{Pix: crop, W: cw, H: ch}
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
	// Insertion sort is fine for page-sized N; use simple sort.
	for i := 1; i < len(order); i++ {
		for j := i; j > 0 && widths[order[j]] < widths[order[j-1]]; j-- {
			order[j], order[j-1] = order[j-1], order[j]
		}
	}
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
	for y := 0; y < nh; y++ {
		for x := 0; x < nw; x++ {
			out[y*side+x] = 1 - float32(sm.Pix[y*sm.W+x])/255
		}
	}
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
	for y := 0; y < g.H; y++ {
		for x := 0; x < g.W; x++ {
			// Inverse map: source point for dest (x,y).
			dx, dy := float64(x)-cx, float64(y)-cy
			sx := cos*dx - sin*dy + cx
			sy := sin*dx + cos*dy + cy
			x0 := int(math.Floor(sx))
			y0 := int(math.Floor(sy))
			fx, fy := sx-float64(x0), sy-float64(y0)
			a := float64(g.At(x0, y0))
			b := float64(g.At(x0+1, y0))
			c := float64(g.At(x0, y0+1))
			d := float64(g.At(x0+1, y0+1))
			out[y*g.W+x] = uint8(math.Round(a*(1-fx)*(1-fy) + b*fx*(1-fy) + c*(1-fx)*fy + d*fx*fy))
		}
	}
	return &Gray{Pix: out, W: g.W, H: g.H}
}

// EstimateSkew returns degrees CCW that straighten text rows. Port of estimate_skew.
func EstimateSkew(g *Gray) float64 {
	const side = 800
	s := float64(side) / float64(max(g.W, g.H))
	sm := g.ResizeArea(max(1, int(math.Round(float64(g.W)*s))), max(1, int(math.Round(float64(g.H)*s))))
	ink := make([]float32, len(sm.Pix))
	sum := 0.0
	for i, p := range sm.Pix {
		ink[i] = (255 - float32(p)) / 255
		sum += float64(ink[i])
	}
	if sum/float64(len(ink)) < 0.003 {
		return 0
	}
	sharp := func(a float64) float64 {
		r := RotatePage(sm, a)
		// Row sums, squared diffs.
		var tot float64
		prev := 0.0
		for y := 0; y < r.H; y++ {
			s := 0.0
			for x := 0; x < r.W; x++ {
				s += float64(255-r.Pix[y*r.W+x]) / 255
			}
			if y > 0 {
				d := s - prev
				tot += d * d
			}
			prev = s
		}
		return tot
	}
	best, bestV := 0.0, -1.0
	for a := -5.0; a <= 5.000001; a += 0.5 {
		if v := sharp(a); v > bestV {
			bestV, best = v, a
		}
	}
	fine, fineV := best, bestV
	for a := best - 0.5; a <= best+0.500001; a += 0.1 {
		if v := sharp(a); v > fineV {
			fineV, fine = v, a
		}
	}
	return fine
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
