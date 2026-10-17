// Assembly: gutter/wide-gap split, reading order, lines and markdown.
package indra

import (
	"math"
	"sort"
	"strings"
)

func f64boxes(bs [][4]float32) [][4]float64 {
	out := make([][4]float64, len(bs))
	for i, b := range bs {
		out[i] = [4]float64{float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}
	}
	return out
}

// finish assembles results and undoes deskew on the output geometry.
func (o *OCR) finish(g *Gray, k int, skew float64, boxes [][4]float32, rec []recLine) *Page {
	wh := [2]float64{float64(g.W), float64(g.H)}
	p := o.assemble(wh, boxes, rec)
	p.Rotation, p.Skew = 90*k, math.Round(skew*100)/100
	if skew != 0 {
		for i := range p.Lines {
			p.Lines[i].Box = unrotateBox(p.Lines[i].Box, skew, g.W, g.H)
		}
		for i := range p.Blocks {
			p.Blocks[i].BBox = unrotateBox(p.Blocks[i].BBox, skew, g.W, g.H)
		}
	}
	return p
}

// CallAll OCRs pages in order, overlapping the per-page stages so the
// recognizer (the bottleneck) runs concurrently with orientation+detection of
// the next page and assembly of the previous one. Results keep input order.
//
// ONNX Runtime sessions are not safe for concurrent Run, so each stage is a
// single goroutine that owns its stage's sessions (orient+det / rec / order):

// assemble splits wide-gap lines, orders, and renders markdown.
func (o *OCR) assemble(wh [2]float64, boxes [][4]float32, rec []recLine) *Page {
	// Gutter cuts + SPLIT_GAP split.
	fb := make([][4]float64, len(boxes))
	for i, b := range boxes {
		fb[i] = [4]float64{float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}
	}
	cuts := gutterCuts(o.v, fb, rec)
	var nb [][4]float64
	var nr []recLine
	for i := range fb {
		var at []float64
		if i < len(cuts) {
			at = cuts[i]
		}
		for _, part := range splitLine(o.v, fb[i], rec[i], SplitGap, at) {
			nb = append(nb, part.box)
			nr = append(nr, part.rec)
		}
	}
	fb, rec = nb, nr
	texts := make([]string, len(rec))
	for i, r := range rec {
		texts[i] = r.text
	}
	order, clsM, sameM := o.readingOrder(fb, texts, wh)
	// NOTE: no table re-split + re-order here. The reference pipeline only
	// re-runs the order model when table *heads* are enabled; the Go port
	// renders tables geometrically, so the line set stays as detected
	// (cells below only affect table rendering).
	lines := make([]Line, len(order))
	for k, i := range order {
		lines[k] = Line{
			Box:  [4]float64{r1(fb[i][0]), r1(fb[i][1]), r1(fb[i][2]), r1(fb[i][3])},
			Text: rec[i].text, Conf: math.Round(rec[i].conf*1e4) / 1e4,
		}
	}
	p := &Page{Lines: lines, Source: "ocr"}
	if clsM != nil {
		p.Blocks = pageBlocks(fb, texts, order, clsM, sameM)
		cells := map[int][][2]string{}
		cellBoxes := map[int][][4]float64{}
		for i := range fb {
			parts := splitLine(o.v, fb[i], rec[i], CellGap, nil)
			if len(parts) > 1 {
				for _, pt := range parts {
					cells[i] = append(cells[i], [2]string{"", pt.rec.text})
					cellBoxes[i] = append(cellBoxes[i], pt.box)
				}
			}
		}
		p.Markdown = pageMarkdown(fb, texts, order, clsM, sameM, o.Furniture, cells, cellBoxes)
	} else if len(fb) > 0 {
		var ts []string
		for _, i := range order {
			ts = append(ts, texts[i])
		}
		p.Markdown = joinNonEmpty(ts, "\n")
	}
	return p
}

func joinNonEmpty(ts []string, sep string) string {
	out := ""
	for _, t := range ts {
		if out == "" {
			out = t
		} else {
			out += sep + t
		}
	}
	return out
}

func (o *OCR) readingOrder(boxes [][4]float64, texts []string, wh [2]float64) ([]int, [][]float64, [][]float64) {
	n := len(boxes)
	if n < 2 || o.s.order == nil {
		return orderXYCut(boxes), nil, nil
	}
	feats := lineFeatures(boxes, wh[0], wh[1], texts)
	out, err := o.s.runOrder(feats, n)
	if err != nil {
		return orderXYCut(boxes), nil, nil
	}
	a := reshape2(out.A, n, n)
	same := reshape2(out.Same, n, n)
	cls := reshape2(out.Cls, n, len(blockClasses))
	return decodeBlocksOrder(a, same, boxes, cls), cls, same
}

func reshape2(f []float32, r, c int) [][]float64 {
	flat := make([]float64, r*c)
	m := make([][]float64, r)
	for i := range m {
		row := flat[i*c : (i+1)*c]
		base := i * c
		for j := 0; j < c && base+j < len(f); j++ {
			row[j] = float64(f[base+j])
		}
		m[i] = row
	}
	return m
}

type splitPart struct {
	box [4]float64
	rec recLine
}

func splitLine(v *vocab, box [4]float64, r recLine, gap float64, at []float64) []splitPart {
	h := math.Max(1, box[3]-box[1])
	spaceID := v.SpaceID()
	var segs [][]token
	cur := []token{}
	prev := math.NaN()
	var segToks [][]token
	_ = segToks
	for _, tk := range r.toks {
		if !(tk.ID == 0 || tk.ID == spaceID) {
			cut := false
			if !math.IsNaN(prev) {
				if tk.X-prev > gap*h {
					cut = true
				} else {
					for _, c := range at {
						if prev < c && c < tk.X {
							cut = true
							break
						}
					}
				}
			}
			if cut {
				segs = append(segs, cur)
				cur = []token{}
			}
			prev = tk.X
		}
		cur = append(cur, tk)
	}
	segs = append(segs, cur)
	if len(segs) == 1 {
		return []splitPart{{box: box, rec: r}}
	}
	var out []splitPart
	for _, sg := range segs {
		var ids []int
		var xs []float64
		for _, tk := range sg {
			// Keep space tokens in the text (pipeline.py _decode uses all
			// tokens); only the box geometry ignores them.
			ids = append(ids, tk.ID)
			if tk.ID == 0 || tk.ID == spaceID {
				continue
			}
			xs = append(xs, tk.X)
		}
		text := strings.TrimSpace(v.Decode(ids))
		if text != "" && len(xs) > 0 {
			mn, mx := xs[0], xs[0]
			for _, x := range xs[1:] {
				mn, mx = math.Min(mn, x), math.Max(mx, x)
			}
			bx := [4]float64{
				math.Max(box[0], mn-0.35*h), box[1],
				math.Min(box[2], mx+0.35*h), box[3],
			}
			out = append(out, splitPart{box: bx, rec: recLine{text: text, conf: r.conf, toks: sg}})
		}
	}
	if len(out) == 0 {
		return []splitPart{{box: box, rec: r}}
	}
	return out
}

func gutterCuts(v *vocab, boxes [][4]float64, rec []recLine) [][]float64 {
	n := len(boxes)
	cuts := make([][]float64, n)
	if n == 0 {
		return cuts
	}
	spaceID := v.SpaceID()
	lefts := make([]float64, n)
	rights := make([]float64, n)
	for i, b := range boxes {
		lefts[i], rights[i] = b[0], b[2]
	}
	sL, sR := lefts, rights // sorted lazily on first wide gap
	sorted := false
	countNear := func(vals []float64, x, tol float64) int {
		lo := sort.SearchFloat64s(vals, x-tol)
		hi := sort.SearchFloat64s(vals, x+tol)
		// open interval (x-tol, x+tol): trim exact boundaries
		for lo < hi && vals[lo] <= x-tol {
			lo++
		}
		for hi > lo && vals[hi-1] >= x+tol {
			hi--
		}
		return hi - lo
	}
	for i := range boxes {
		h := math.Max(1, boxes[i][3]-boxes[i][1])
		var xs []float64
		for _, tk := range rec[i].toks {
			if tk.ID == 0 || tk.ID == spaceID {
				continue
			}
			xs = append(xs, tk.X)
		}
		tol := 0.5 * h
		wide := false
		for k := 0; k+1 < len(xs); k++ {
			if xs[k+1]-xs[k] > 1.6*h {
				wide = true
				break
			}
		}
		if !wide {
			continue
		}
		if !sorted {
			sL = append([]float64(nil), lefts...)
			sR = append([]float64(nil), rights...)
			sort.Float64s(sL)
			sort.Float64s(sR)
			sorted = true
		}
		for k := 0; k+1 < len(xs); k++ {
			a, c := xs[k], xs[k+1]
			if c-a <= 1.6*h {
				continue
			}
			gl, gr := a+0.35*h, c-0.35*h
			near := countNear(sR, gl, tol) + countNear(sL, gr, tol)
			// countNear includes box i itself; original skips j==i and
			// counts a box once even if both edges match. Adjust.
			if math.Abs(rights[i]-gl) < tol {
				near--
			}
			if math.Abs(lefts[i]-gr) < tol {
				near--
			}
			if math.Abs(rights[i]-gl) < tol && math.Abs(lefts[i]-gr) < tol {
				near++ // was counted twice, should be removed once
			}
			if near >= 3 {
				cuts[i] = append(cuts[i], (a+c)/2)
			}
		}
	}
	return cuts
}
