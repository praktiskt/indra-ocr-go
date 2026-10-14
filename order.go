// Reading order + block clustering. Port of order_post.py (torch-free parts).
package indra

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// BlockClasses are the layout head classes per line.
var BlockClasses = []string{
	"Text", "Section-Header", "Page-Header", "Page-Footer", "List-Group",
	"Table", "Caption", "Footnote", "Table-Of-Contents", "Form",
}

// GridLabels are blocks ordered row-by-row.
var gridLabel = map[string]bool{"Table": true, "Form": true, "Table-Of-Contents": true}

func blockIsGrid(id int) bool {
	if id < 0 || id >= len(BlockClasses) {
		return false
	}
	return gridLabel[BlockClasses[id]]
}

const (
	nGeom  = 10
	nText  = 6
	NFeats = nGeom + nText
)

// LineFeatures builds [N x NFeats] float32 features. Port of line_features.
func LineFeatures(boxes [][4]float64, pageW, pageH float64, texts []string) []float32 {
	n := len(boxes)
	out := make([]float32, n*NFeats)
	if n == 0 {
		return out
	}
	hs := make([]float64, n)
	for i, b := range boxes {
		hs[i] = math.Max(b[3]-b[1], 1e-3)
	}
	sorted := append([]float64(nil), hs...)
	sort.Float64s(sorted)
	medH := npMedian(sorted)
	for i, b := range boxes {
		w := math.Max(b[2]-b[0], 1e-3)
		h := hs[i]
		g := []float64{
			b[0] / pageW, b[1] / pageH, b[2] / pageW, b[3] / pageH,
			(b[0] + b[2]) / (2 * pageW), (b[1] + b[3]) / (2 * pageH),
			w / pageW, math.Log(h / medH), math.Log(w/h) / 4,
			medH / pageH * 20,
		}
		copy(out[i*NFeats:i*NFeats+nGeom], float64s(g))
		if texts != nil && i < len(texts) {
			t := trimSpace(texts[i])
			if t != "" {
				r := []rune(t)
				nd := 0
				for _, c := range r {
					if c >= '0' && c <= '9' {
						nd++
					}
				}
				row := out[i*NFeats+nGeom : (i+1)*NFeats]
				row[0] = float32(math.Log1p(float64(len(t))) / 5)
				row[1] = float32(float64(nd) / float64(len(t)))
				if isListStart(t) {
					row[2] = 1
				}
				if endsWithHyphen(t) {
					row[3] = 1
				}
				last := r[len(r)-1]
				if last == '.' || last == '!' || last == '?' || last == ':' || last == ';' {
					row[4] = 1
				}
				if unicode.IsUpper(r[0]) {
					row[5] = 1
				}
			}
		}
	}
	return out
}

func float64s(v []float64) []float32 {
	o := make([]float32, len(v))
	for i, x := range v {
		o[i] = float32(x)
	}
	return o
}

func trimSpace(s string) string {
	// ASCII fast path is enough for feature flags.
	start := 0
	for start < len(s) && (s[start] == ' ' || s[start] == '\t' || s[start] == '\n') {
		start++
	}
	end := len(s)
	for end > start && (s[end-1] == ' ' || s[end-1] == '\t' || s[end-1] == '\n') {
		end--
	}
	return s[start:end]
}

func isListStart(t string) bool {
	r := []rune(t)
	if len(r) > 0 {
		switch r[0] {
		case '•', '-', '–', '▪', '◦', '*', '·', '→':
			return true
		}
	}
	// (t[:3].rstrip(".)")).isdigit() and len(t) > 3
	prefix := t
	if len(prefix) > 3 {
		prefix = prefix[:3]
	}
	prefix = strings.TrimRight(prefix, ".)")
	if len(t) > 3 && len(prefix) > 0 && isAllDigits(prefix) {
		return true
	}
	return false
}

func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

func endsWithHyphen(t string) bool { return len(t) > 0 && t[len(t)-1] == '-' }

// DecodeOrder orders from a pairwise logit matrix [N x N]. Port of decode_order.
func DecodeOrder(a [][]float64, sweeps int) []int {
	n := len(a)
	if n <= 1 {
		o := make([]int, n)
		for i := range o {
			o[i] = i
		}
		return o
	}
	// lp = log P(i before j) = -softplus(-a)
	lp := make([][]float64, n)
	for i := range lp {
		lp[i] = make([]float64, n)
		for j := range lp[i] {
			lp[i][j] = -softplus(-a[i][j])
		}
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(p, q int) bool {
		si, sj := 0.0, 0.0
		for k := 0; k < n; k++ {
			si += -lp[k][order[p]] + lp[order[p]][k]
			sj += -lp[k][order[q]] + lp[order[q]][k]
		}
		// sort by rowsum desc: key = sum_j lp[i,j] - sum_k lp[k,i]
		ki := 0.0
		kj := 0.0
		for k := 0; k < n; k++ {
			ki += lp[order[p]][k] - lp[k][order[p]]
			kj += lp[order[q]][k] - lp[k][order[q]]
		}
		_ = si
		_ = sj
		return ki > kj
	})
	pos := make([]int, n)
	for sweep := 0; sweep < sweeps; sweep++ {
		improved := false
		cur := append([]int(nil), order...)
		for _, x := range cur {
			// rest = order without x
			rest := make([]int, 0, n-1)
			curPos := -1
			for k, v := range order {
				if v == x {
					curPos = k
					continue
				}
				rest = append(rest, v)
			}
			before := make([]float64, n-1) // lp[rest,x]
			after := make([]float64, n-1)  // lp[x,rest]
			for k, v := range rest {
				before[k] = lp[v][x]
				after[k] = lp[x][v]
			}
			gains := make([]float64, n)
			cum := 0.0
			gains[0] = 0
			for k := 0; k < n-1; k++ {
				cum += before[k]
				gains[k+1] = cum
			}
			// add reversed cumsum of after
			cum = 0
			suf := make([]float64, n)
			for k := n - 2; k >= 0; k-- {
				cum += after[k]
				suf[k] = cum
			}
			for k := 0; k < n; k++ {
				gains[k] += suf[k]
			}
			best := 0
			for k := 1; k < n; k++ {
				if gains[k] > gains[best] {
					best = k
				}
			}
			if gains[best] > gains[curPos]+1e-6 {
				order = append(append([]int(nil), rest[:best]...), append([]int{x}, rest[best:]...)...)
				improved = true
			}
		}
		if !improved {
			break
		}
	}
	_ = pos
	return order
}

func softplus(x float64) float64 {
	if x > 20 {
		return x
	}
	return math.Log1p(math.Exp(x))
}

// OrderTopDown sorts rows top-to-bottom, left-to-right.
func OrderTopDown(boxes [][4]float64) []int {
	n := len(boxes)
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	if n == 0 {
		return idx
	}
	cy := make([]float64, n)
	hs := make([]float64, n)
	for i, b := range boxes {
		cy[i] = (b[1] + b[3]) / 2
		hs[i] = b[3] - b[1]
	}
	s := append([]float64(nil), hs...)
	sort.Float64s(s)
	h := npMedian(s)
	if h*0.5 < 1e-3 {
		h = 2e-3
	}
	sort.Slice(idx, func(p, q int) bool {
		rp := pyRound(cy[idx[p]] / (h * 0.5))
		rq := pyRound(cy[idx[q]] / (h * 0.5))
		if rp != rq {
			return rp < rq
		}
		return boxes[idx[p]][0] < boxes[idx[q]][0]
	})
	return idx
}

// pyRound matches Python's round(x) (banker's rounding, half to even) for
// non-negative values, which order_top_down relies on for row grouping.
func pyRound(x float64) int {
	f := math.Floor(x)
	d := x - f
	switch {
	case d < 0.5:
		return int(f)
	case d > 0.5:
		return int(f) + 1
	default:
		if int(f)%2 == 0 {
			return int(f)
		}
		return int(f) + 1
	}
}

// OrderXYCut is the recursive XY-cut baseline.
func OrderXYCut(boxes [][4]float64) []int {
	ids := make([]int, len(boxes))
	for i := range ids {
		ids[i] = i
	}
	return xyRec(boxes, ids)
}

func bestCut(boxes [][4]float64, ids []int, axis int) (gap float64, k int, o []int) {
	o = append([]int(nil), ids...)
	sort.Slice(o, func(p, q int) bool {
		if axis == 1 {
			return boxes[o[p]][1] < boxes[o[q]][1]
		}
		return boxes[o[p]][0] < boxes[o[q]][0]
	})
	lo := func(i int) float64 {
		if axis == 1 {
			return boxes[i][1]
		}
		return boxes[i][0]
	}
	hi := func(i int) float64 {
		if axis == 1 {
			return boxes[i][3]
		}
		return boxes[i][2]
	}
	end := hi(o[0])
	gap, k = 0, -1
	for j := 1; j < len(o); j++ {
		g := lo(o[j]) - end
		if g > gap {
			gap, k = g, j
		}
		if hi(o[j]) > end {
			end = hi(o[j])
		}
	}
	return gap, k, o
}

func xyRec(boxes [][4]float64, ids []int) []int {
	if len(ids) <= 1 {
		return append([]int(nil), ids...)
	}
	gy, ky, oy := bestCut(boxes, ids, 1)
	gx, kx, ox := bestCut(boxes, ids, 0)
	if ky < 0 && kx < 0 {
		sub := make([][4]float64, len(ids))
		for i, id := range ids {
			sub[i] = boxes[id]
		}
		local := OrderTopDown(sub)
		out := make([]int, len(ids))
		for i, l := range local {
			out[i] = ids[l]
		}
		return out
	}
	var k int
	var o []int
	if gy >= gx {
		k, o = ky, oy
	} else {
		k, o = kx, ox
	}
	return append(xyRec(boxes, o[:k]), xyRec(boxes, o[k:])...)
}

// cluster is average-linkage clustering of same-group logits. Port of _cluster.
func cluster(logits [][]float64) [][]int {
	n := len(logits)
	if n == 0 {
		return nil
	}
	// Mean matrix M, -Inf on diagonal/dead.
	negInf := math.Inf(-1)
	S := make([][]float64, n)
	M := make([][]float64, n)
	for i := range S {
		S[i] = append([]float64(nil), logits[i]...)
		S[i][i] = 0
		M[i] = append([]float64(nil), S[i]...)
		M[i][i] = negInf
	}
	size := make([]float64, n)
	alive := make([]bool, n)
	members := make([][]int, n)
	for i := range size {
		size[i] = 1
		alive[i] = true
		members[i] = []int{i}
	}
	best := make([]int, n)
	bestV := make([]float64, n)
	for i := 0; i < n; i++ {
		b := 0
		for j := 1; j < n; j++ {
			if M[i][j] > M[i][b] {
				b = j
			}
		}
		best[i], bestV[i] = b, M[i][b]
	}
	for iter := 0; iter < n-1; iter++ {
		a := 0
		for i := 1; i < n; i++ {
			if alive[i] && bestV[i] > bestV[a] || !alive[a] {
				a = i
			}
		}
		if !alive[a] || bestV[a] <= 0 {
			break
		}
		b := best[a]
		if a > b {
			a, b = b, a
		}
		for j := 0; j < n; j++ {
			S[a][j] += S[b][j]
			S[j][a] += S[j][b]
		}
		S[a][a] = 0
		size[a] += size[b]
		alive[b] = false
		members[a] = append(members[a], members[b]...)
		for j := 0; j < n; j++ {
			v := S[a][j] / (size[a] * size[j])
			if !alive[j] || j == a {
				v = negInf
			}
			M[a][j], M[j][a] = v, v
		}
		for j := 0; j < n; j++ {
			M[b][j], M[j][b] = negInf, negInf
		}
		bestV[b] = negInf
		// refresh best[a]
		bb := 0
		for j := 1; j < n; j++ {
			if M[a][j] > M[a][bb] {
				bb = j
			}
		}
		best[a], bestV[a] = bb, M[a][bb]
		for j := 0; j < n; j++ {
			if !alive[j] || j == a {
				continue
			}
			if best[j] == a || best[j] == b {
				bb := 0
				for k := 1; k < n; k++ {
					if M[j][k] > M[j][bb] {
						bb = k
					}
				}
				best[j], bestV[j] = bb, M[j][bb]
			} else if M[j][a] > bestV[j] {
				best[j], bestV[j] = a, M[j][a]
			}
		}
	}
	var out [][]int
	for i := 0; i < n; i++ {
		if alive[i] {
			s := append([]int(nil), members[i]...)
			sort.Ints(s)
			out = append(out, s)
		}
	}
	return out
}

// DecodeBlocksOrder orders with layout heads. Port of decode_blocks_order.
func DecodeBlocksOrder(a, same [][]float64, boxes [][4]float64, cls [][]float64) []int {
	n := len(a)
	if n <= 1 {
		o := make([]int, n)
		for i := range o {
			o[i] = i
		}
		return o
	}
	blocks := cluster(same)
	inner := make([][]int, len(blocks))
	for bi, b := range blocks {
		bx := make([][4]float64, len(b))
		for i, id := range b {
			bx[i] = boxes[id]
		}
		var local []int
		if cls != nil && blockIsGrid(argmaxRowSum(cls, b)) {
			local = OrderTopDown(bx)
		} else {
			local = OrderXYCut(bx)
		}
		inner[bi] = make([]int, len(b))
		for i, l := range local {
			inner[bi][i] = b[l]
		}
	}
	// Block order by mean pairwise logit.
	m := len(blocks)
	ab := make([][]float64, m)
	for i := range ab {
		ab[i] = make([]float64, m)
	}
	for p := 0; p < m; p++ {
		for q := 0; q < m; q++ {
			if p == q {
				continue
			}
			s := 0.0
			for _, i := range blocks[p] {
				for _, j := range blocks[q] {
					s += a[i][j]
				}
			}
			ab[p][q] = s / float64(len(blocks[p])*len(blocks[q]))
		}
	}
	border := DecodeOrder(ab, 4)
	var out []int
	for _, k := range border {
		out = append(out, inner[k]...)
	}
	return out
}

func argmaxRowSum(cls [][]float64, rows []int) int {
	if len(cls) == 0 {
		return 0
	}
	c := len(cls[0])
	sum := make([]float64, c)
	for _, r := range rows {
		for j := 0; j < c && j < len(cls[r]); j++ {
			sum[j] += cls[r][j]
		}
	}
	b := 0
	for j := 1; j < c; j++ {
		if sum[j] > sum[b] {
			b = j
		}
	}
	return b
}

// SplitBlocks cuts the reading order where same-block logit <= 0.
func SplitBlocks(order []int, same [][]float64) [][]int {
	var blocks [][]int
	for _, i := range order {
		if len(blocks) > 0 && same[blocks[len(blocks)-1][len(blocks[len(blocks)-1])-1]][i] > 0 {
			blocks[len(blocks)-1] = append(blocks[len(blocks)-1], i)
		} else {
			blocks = append(blocks, []int{i})
		}
	}
	return blocks
}
