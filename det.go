// Line detection decoding. Port of decode() in det_post.py.
//
// Input: det.onnx output [3,h,w] (core logit, top dist, bottom dist) at
// OUT_STRIDE=2. Connected components on sigmoid(core)>0.5 (4-connectivity);
// box y from mean (centre-top, centre+bottom); x from component span.
package indra

import "math"

const (
	outStride   = 2
	coreXShrink = 0.15
)

// DetBox is x0,y0,x1,y1,score in input pixels.
type DetBox struct {
	X0, Y0, X1, Y1 float32
	Score          float32
}

// DecodeDet decodes one page. out is [3*h*w] in [c,h,w] order.
func DecodeDet(out []float32, h, w int) []DetBox {
	if len(out) < 3*h*w {
		return nil
	}
	core := out[:h*w]
	top := out[h*w : 2*h*w]
	bot := out[2*h*w : 3*h*w]
	// Sigmoid + threshold mask.
	mask := make([]bool, h*w)
	for i, v := range core {
		p := float32(1 / (1 + math.Exp(float64(-v))))
		if p > 0.5 {
			mask[i] = true
		}
	}
	lab, n := connectedComponents(mask, h, w)
	if n == 0 {
		return nil
	}
	// Stats per label (1-based).
	area := make([]int, n+1)
	left := make([]int, n+1)
	width := make([]int, n+1)
	for i := 1; i <= n; i++ {
		left[i] = w
	}
	right := make([]int, n+1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			l := lab[y*w+x]
			if l == 0 {
				continue
			}
			area[l]++
			if x < left[l] {
				left[l] = x
			}
			if x > right[l] {
				right[l] = x
			}
		}
	}
	for i := 1; i <= n; i++ {
		width[i] = right[i] - left[i] + 1
	}
	// Bincount means of top/bot/score over component pixels.
	cnt := make([]float64, n+1)
	sumTop := make([]float64, n+1)
	sumBot := make([]float64, n+1)
	sumScore := make([]float64, n+1)
	for idx := 0; idx < h*w; idx++ {
		l := lab[idx]
		if l == 0 {
			continue
		}
		y := idx / w
		yc := (float64(y) + 0.5) * outStride
		p := 1 / (1 + math.Exp(float64(-core[idx])))
		cnt[l]++
		sumTop[l] += yc - float64(top[idx])
		sumBot[l] += yc + float64(bot[idx])
		sumScore[l] += p
	}
	var boxes []DetBox
	for i := 1; i <= n; i++ {
		if area[i] < 2 {
			continue
		}
		y0 := sumTop[i] / cnt[i]
		y1 := sumBot[i] / cnt[i]
		score := sumScore[i] / cnt[i]
		lh := y1 - y0
		if lh < 1 {
			lh = 1
		}
		ex := math.Min(coreXShrink*lh, 0.5*lh)
		x0 := float64(left[i])*outStride - ex
		x1 := float64(left[i]+width[i])*outStride + ex
		boxes = append(boxes, DetBox{
			X0: float32(x0), Y0: float32(y0),
			X1: float32(x1), Y1: float32(y1),
			Score: float32(score),
		})
	}
	return boxes
}

// connectedComponents labels 4-connected true pixels, returns labels + count.
func connectedComponents(mask []bool, h, w int) ([]int, int) {
	lab := make([]int, h*w)
	count := 0
	stack := make([]int, 0, 64)
	for i := 0; i < h*w; i++ {
		if !mask[i] || lab[i] != 0 {
			continue
		}
		count++
		lab[i] = count
		stack = stack[:0]
		stack = append(stack, i)
		for len(stack) > 0 {
			p := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			px, py := p%w, p/w
			if px > 0 {
				q := p - 1
				if mask[q] && lab[q] == 0 {
					lab[q] = count
					stack = append(stack, q)
				}
			}
			if px+1 < w {
				q := p + 1
				if mask[q] && lab[q] == 0 {
					lab[q] = count
					stack = append(stack, q)
				}
			}
			if py > 0 {
				q := p - w
				if mask[q] && lab[q] == 0 {
					lab[q] = count
					stack = append(stack, q)
				}
			}
			if py+1 < h {
				q := p + w
				if mask[q] && lab[q] == 0 {
					lab[q] = count
					stack = append(stack, q)
				}
			}
		}
	}
	return lab, count
}
