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

// detBox is x0,y0,x1,y1,score in input pixels.
type detBox struct {
	X0, Y0, X1, Y1 float32
	Score          float32
}

// decodeDet decodes one page. out is [3*h*w] in [c,h,w] order.
func decodeDet(out []float32, h, w int) []detBox {
	if len(out) < 3*h*w {
		return nil
	}
	core := out[:h*w]
	top := out[h*w : 2*h*w]
	bot := out[2*h*w : 3*h*w]
	// Foreground <=> sigmoid(core) > 0.5 <=> core > 0. Labeled in one
	// union-find pass over core directly: no mask array, no Exp here.
	lab, n := labelForeground(core, h, w)
	if n == 0 {
		return nil
	}
	// Single fused pass: bounds + means. Exp only on foreground pixels.
	area := make([]int, n+1)
	left := make([]int, n+1)
	for i := 1; i <= n; i++ {
		left[i] = w
	}
	right := make([]int, n+1)
	cnt := make([]float64, n+1)
	sumTop := make([]float64, n+1)
	sumBot := make([]float64, n+1)
	sumScore := make([]float64, n+1)
	for idx := 0; idx < h*w; idx++ {
		l := lab[idx]
		if l == 0 {
			continue
		}
		x := idx % w
		y := idx / w
		area[l]++
		if x < left[l] {
			left[l] = x
		}
		if x > right[l] {
			right[l] = x
		}
		cv := core[idx]
		yc := (float64(y) + 0.5) * outStride
		p := 1 / (1 + math.Exp(float64(-cv)))
		cnt[l]++
		sumTop[l] += yc - float64(top[idx])
		sumBot[l] += yc + float64(bot[idx])
		sumScore[l] += p
	}
	var boxes []detBox
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
		x1 := float64(left[i]+right[i]-left[i]+1)*outStride + ex
		boxes = append(boxes, detBox{
			X0: float32(x0), Y0: float32(y0),
			X1: float32(x1), Y1: float32(y1),
			Score: float32(score),
		})
	}
	return boxes
}

// labelForeground labels 4-connected core>0 pixels with two-pass union-find.
// Streaming raster order: no mask array, no stack. Returns int32 labels + count.
func labelForeground(core []float32, h, w int) ([]int32, int) {
	lab := make([]int32, h*w)
	parent := make([]int32, 1, 256) // parent[0] unused; index = provisional label
	find := func(x int32) int32 {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	var cnt int32
	for y := 0; y < h; y++ {
		row := y * w
		for x := 0; x < w; x++ {
			idx := row + x
			if core[idx] <= 0 {
				continue
			}
			var left, up int32
			if x > 0 {
				left = lab[idx-1]
			}
			if y > 0 {
				up = lab[idx-w]
			}
			switch {
			case left == 0 && up == 0:
				cnt++
				parent = append(parent, cnt)
				lab[idx] = cnt
			case up == 0:
				lab[idx] = left
			case left == 0:
				lab[idx] = up
			default:
				rl, ru := find(left), find(up)
				if rl != ru {
					if rl > ru {
						rl, ru = ru, rl
					}
					parent[ru] = rl
				}
				lab[idx] = rl
			}
		}
	}
	if cnt == 0 {
		return lab, 0
	}
	// Flatten roots to compact 1..n ids.
	remap := make([]int32, cnt+1)
	var n int32
	for i := range lab {
		if lab[i] == 0 {
			continue
		}
		r := find(lab[i])
		id := remap[r]
		if id == 0 {
			n++
			id = n
			remap[r] = id
		}
		lab[i] = id
	}
	return lab, int(n)
}
