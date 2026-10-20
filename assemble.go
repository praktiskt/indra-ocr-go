// Assembly: marker/leader tidy, gutter/wide-gap split, reading order (with
// right-to-left and vertical handling), lines and markdown.
package indra

import (
	"math"
	"regexp"
	"strings"
	"unicode"
)

var (
	// leaderChars: dot leaders, fill-in rules.
	leaderChars = map[rune]bool{'.': true, '·': true, '…': true, '‥': true, '_': true, '-': true, '–': true, '—': true, '=': true}
	// dashes: a lone short dash can be a table cell; other lone marks are not.
	dashes = map[rune]bool{'-': true, '–': true, '—': true}
	// bullets: list markers (not dashes: a "-" can be a table cell).
	bullets = "•◦▪▫●○■□‣⁃∙*"
	// a numbered list marker: "1.", "12)", "a.", "B)", "(3)", "(b)", "(iv)".
	enumerator = regexp.MustCompile(`^(\d{1,3}|[A-Za-z])[.)]$|^\((\d{1,3}|[A-Za-z]|[ivxlcIVXLC]{1,5})\)$`)
	// a list marker and its item's text are at most this many line heights
	// apart: the label rule (training labels cut a text-layer line only at
	// wider gaps).
	markerGap = 2.5
)

func f64boxes(bs [][4]float32) [][4]float64 {
	out := make([][4]float64, len(bs))
	for i, b := range bs {
		out[i] = [4]float64{float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}
	}
	return out
}

// tidyLines drops the two kinds of lines that carry no text of their own:
//
//   - a line of leader characters only (form fill-in dots and rules, specks)
//     is dropped: 3 or more of them, a box 3 or more times wider than tall
//     (CTC reads a dotted rule as one "."), or any dot or underscore alone. A
//     short lone dash stays: it can be an empty table cell;
//   - a lone list marker (a bullet, "1.", "(a)") joins the line that starts
//     within markerGap line heights after it on the same row (on its left on
//     a right-to-left page): the item keeps its marker. On a left-to-right
//     page the marker's tokens join too (space: the space token), so a table
//     line still splits into cells at its gaps.
//
// Other lines without letters or digits ("-", "/ /", "( )") stay: they can be
// table cells or form content. Port of tidy_lines.
func (o *OCR) tidyLines(boxes [][4]float64, rec []recLine, rtl bool) ([][4]float64, []recLine) {
	keep := make([]bool, len(rec))
	for i := range keep {
		keep[i] = true
	}
	rec = append([]recLine(nil), rec...)
	h := make([]float64, len(boxes))
	w := make([]float64, len(boxes))
	for i, b := range boxes {
		h[i] = b[3] - b[1]
		w[i] = b[2] - b[0]
	}
	for i, r := range rec {
		core := strings.ReplaceAll(r.text, " ", "")
		allLeader := core != ""
		for _, c := range core {
			if !leaderChars[c] {
				allLeader = false
				break
			}
		}
		if !allLeader {
			continue
		}
		wide := w[i] >= 3*h[i]
		anyDash := false
		for _, c := range core {
			if dashes[c] {
				anyDash = true
				break
			}
		}
		if len([]rune(core)) >= 3 || wide || !anyDash {
			keep[i] = false
		}
	}
	for i, r := range rec {
		t := strings.TrimSpace(r.text)
		if !keep[i] || !(strings.ContainsRune(bullets, firstRune(t)) && len(t) == 1 || enumerator.MatchString(t)) {
			continue
		}
		best, bestGap := -1, 0.0
		for j := range rec {
			if j == i || !keep[j] || strings.TrimSpace(rec[j].text) == "" {
				continue
			}
			ov := min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1])
			if ov < 0.5*math.Min(h[i], h[j]) {
				continue
			}
			gap := boxes[j][0] - boxes[i][2]
			if rtl {
				gap = boxes[i][0] - boxes[j][2]
			}
			if -(0.3 * h[i]) <= gap && gap <= markerGap*math.Max(h[i], h[j]) &&
				(bestGap < 0 || gap < bestGap) {
				best, bestGap = j, gap
			}
		}
		if best < 0 {
			continue
		}
		tj, cj, kj := rec[best].text, rec[best].conf, rec[best].toks
		chars := len(kj) > 0 && kj[0].Ch != ""
		// same kind of tokens (text-layer chars or recognizer ids) only
		if !rtl && len(r.toks) > 0 && len(kj) > 0 &&
			(r.toks[0].Ch != "") == chars && (chars || o.v.SpaceID() >= 0) {
			x := (boxes[i][2] + boxes[best][0]) / 2
			sp := token{ID: o.v.SpaceID(), Ch: "", X: x}
			if chars {
				sp = token{Ch: " ", X: x, Seq: kj[len(kj)-1].Seq, hasSeq: kj[len(kj)-1].hasSeq}
			}
			kj = append(append(append([]token{}, r.toks...), sp), kj...)
		}
		rec[best] = recLine{text: t + " " + strings.TrimSpace(tj), conf: cj, toks: kj}
		boxes[best] = [4]float64{
			math.Min(boxes[i][0], boxes[best][0]),
			math.Min(boxes[i][1], boxes[best][1]),
			math.Max(boxes[i][2], boxes[best][2]),
			math.Max(boxes[i][3], boxes[best][3]),
		}
		keep[i] = false
	}
	idx := []int{}
	for i, k := range keep {
		if k {
			idx = append(idx, i)
		}
	}
	nb := make([][4]float64, len(idx))
	nr := make([]recLine, len(idx))
	for q, i := range idx {
		nb[q], nr[q] = boxes[i], rec[i]
	}
	return nb, nr
}

func firstRune(s string) rune {
	for _, c := range s {
		return c
	}
	return 0
}

// finish assembles results and undoes deskew on the output geometry (and the
// vertical quarter turn, mapping the boxes back to the page as it is).
func (o *OCR) finish(g *Gray, k int, skew float64, boxes [][4]float32, rec []recLine) *Page {
	wh := [2]float64{float64(g.W), float64(g.H)}
	vertical := k == orientVertical
	p := o.assemble(wh, boxes, rec, vertical)
	if vertical {
		p.Rotation = 0
	} else {
		p.Rotation = 90 * k
	}
	p.Skew = math.Round(skew*100) / 100
	p.Vertical = vertical
	if skew != 0 {
		for i := range p.Lines {
			p.Lines[i].Box = unrotateBox(p.Lines[i].Box, skew, g.W, g.H)
		}
		for i := range p.Blocks {
			p.Blocks[i].BBox = unrotateBox(p.Blocks[i].BBox, skew, g.W, g.H)
		}
	}
	if vertical { // back on the page as it is: (x, y) -> (W - y, x)
		W := float64(g.H)
		for i := range p.Lines {
			p.Lines[i].Box = verticalBack(p.Lines[i].Box, W)
		}
		for i := range p.Blocks {
			p.Blocks[i].BBox = verticalBack(p.Blocks[i].BBox, W)
		}
	}
	return p
}

func verticalBack(b [4]float64, W float64) [4]float64 {
	return [4]float64{W - b[3], b[0], W - b[1], b[2]}
}

// assemble splits wide-gap lines, tidies markers and leaders, orders, and
// renders markdown. Vertical pages (read turned a quarter turn) fall back to
// XY-cut blocks: the order model saw no vertical pages.
func (o *OCR) assemble(wh [2]float64, boxes [][4]float32, rec []recLine, vertical bool) *Page {
	// A right-to-left page: lines read right to left (_decode), and reading
	// order and markdown work on the page mirrored left to right, so the
	// first column is the right one. Letter counts do not depend on order.
	total, nRTL := 0, 0
	for _, r := range rec {
		for _, c := range r.text {
			if unicode.IsLetter(c) {
				total++
				if isRTLChar(c) {
					nRTL++
				}
			}
		}
	}
	pageRTL := nRTL > total-nRTL
	width := wh[0]

	mirror := func(b [4]float64) [4]float64 {
		if !pageRTL {
			return b
		}
		return [4]float64{width - b[2], b[1], width - b[0], b[3]}
	}

	// Text-layer labels split a line at wide gaps; do the same (SplitGap).
	boxes64 := f64boxes(boxes)
	cuts := gutterCuts(o, boxes64, rec)
	var nb2 [][4]float64
	var nr []recLine
	for i := range boxes64 {
		var at []float64
		if i < len(cuts) {
			at = cuts[i]
		}
		for _, part := range splitLine(o, boxes64[i], rec[i], SplitGap, at, pageRTL) {
			nb2 = append(nb2, part.box)
			nr = append(nr, part.rec)
		}
	}
	boxes64, rec = o.tidyLines(boxes64, nr, pageRTL)
	texts := make([]string, len(rec))
	for i, r := range rec {
		texts[i] = r.text
	}
	gb := make([][4]float64, len(boxes64))
	for i, b := range boxes64 {
		gb[i] = mirror(b)
	}
	order, clsM, sameM := o.readingOrder(gb, texts, wh)
	if vertical && clsM != nil {
		// The order model saw no vertical pages: turned, their columns of
		// short lines look like one table read row by row across tiers.
		// Blocks and order come from XY-cut instead, without grid classes.
		blocks := xyCutBlocks(boxes64, 1.0)
		order = []int{}
		for _, b := range blocks {
			order = append(order, b...)
		}
		same := make([][]float64, len(boxes64))
		for i := range same {
			same[i] = make([]float64, len(boxes64))
			for j := range same[i] {
				same[i][j] = -1
			}
		}
		for _, b := range blocks {
			for _, i := range b {
				for _, j := range b {
					same[i][j] = 1
				}
			}
		}
		for i := range clsM {
			for _, gl := range []string{"Table", "Form", "Table-Of-Contents"} {
				clsM[i][blockIndex(gl)] = -1e4
			}
		}
	}
	lines := make([]Line, len(order))
	for k, i := range order {
		lines[k] = Line{
			Box:  [4]float64{r1(boxes64[i][0]), r1(boxes64[i][1]), r1(boxes64[i][2]), r1(boxes64[i][3])},
			Text: texts[i], Conf: math.Round(rec[i].conf*1e4) / 1e4,
		}
	}
	p := &Page{Lines: lines, Source: "ocr"}
	if clsM != nil {
		p.Blocks = pageBlocks(gb, texts, order, clsM, sameM)
		for i := range p.Blocks {
			p.Blocks[i].BBox = mirror(p.Blocks[i].BBox)
		}
		cells := map[int][][2]string{}
		cellBoxes := map[int][][4]float64{}
		for i := range boxes64 {
			parts := splitLine(o, boxes64[i], rec[i], CellGap, nil, pageRTL)
			if len(parts) > 1 {
				for _, pt := range parts {
					cells[i] = append(cells[i], [2]string{"", pt.rec.text})
					cellBoxes[i] = append(cellBoxes[i], pt.box)
				}
			}
		}
		p.Markdown = pageMarkdown(gb, texts, order, clsM, sameM, o.Furniture, cells, cellBoxes)
	} else if len(boxes64) > 0 {
		var ts []string
		for _, i := range order {
			ts = append(ts, texts[i])
		}
		p.Markdown = joinNonEmpty(ts, "\n")
	}
	return p
}

// countAlpha is unused leftover; removed.

func blockIndex(name string) int {
	for i, c := range blockClasses {
		if c == name {
			return i
		}
	}
	return -1
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

// splitLine cuts a line where two neighbouring characters are more than gap
// line heights apart (the detector merged two cells or columns). Returns
// [(box, text, tokens)]; one item when nothing is cut. Port of split_line;
// segment texts decode through _decode.
func splitLine(o *OCR, box [4]float64, r recLine, gap float64, at []float64, pageRTL bool) []splitPart {
	h := math.Max(1, box[3]-box[1])
	var segs [][]token
	cur := []token{}
	prev := math.NaN()
	for _, tk := range r.toks {
		if !o.isSpaceTok(tk) {
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
		var xs []float64
		for _, tk := range sg {
			if !o.isSpaceTok(tk) {
				xs = append(xs, tk.X)
			}
		}
		text := strings.TrimSpace(o.decodeToks(sg, &pageRTL))
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

// gutterCuts: per line, the x positions of gaps that sit on a column gutter:
// a gap over GutterGap line heights where at least GutterEdges other lines
// end at its left side or start at its right side.
func gutterCuts(o *OCR, boxes [][4]float64, rec []recLine) [][]float64 {
	const gutterGap = 1.6
	const gutterEdges = 3
	n := len(boxes)
	cuts := make([][]float64, n)
	if n == 0 {
		return cuts
	}
	lefts := make([]float64, n)
	rights := make([]float64, n)
	for i, b := range boxes {
		lefts[i], rights[i] = b[0], b[2]
	}
	for i := range boxes {
		h := math.Max(1, boxes[i][3]-boxes[i][1])
		var xs []float64
		for _, tk := range rec[i].toks {
			if o.isSpaceTok(tk) {
				continue
			}
			xs = append(xs, tk.X)
		}
		for k := 0; k+1 < len(xs); k++ {
			a, c := xs[k], xs[k+1]
			if c-a <= gutterGap*h {
				continue
			}
			gl, gr := a+0.35*h, c-0.35*h // gap edges
			near := 0
			for j := 0; j < n; j++ {
				if j == i {
					continue
				}
				if math.Abs(rights[j]-gl) < 0.5*h || math.Abs(lefts[j]-gr) < 0.5*h {
					near++
				}
			}
			if near >= gutterEdges {
				cuts[i] = append(cuts[i], (a+c)/2)
			}
		}
	}
	return cuts
}
