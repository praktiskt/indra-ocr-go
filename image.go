// Grayscale page type and image conversions. Port of the cv2/numpy bits in
// det_post.py, rec_preproc.py, pipeline.py and line_mask.py.
package indra

import (
	"image"
)

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

// resize scales to nw x nh, choosing the filter like cv2: area-average when
// shrinking, bilinear otherwise. (Callers that must mirror an explicit cv2
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
