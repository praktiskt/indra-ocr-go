package indra

// PDF text-layer support: the caller extracts a page's text runs with any PDF
// library and hands them to CallText, which uses them where they are sound and
// OCRs the rest (the upstream "auto" path). Faithful to indra_ocr/textlayer.py
// and the text-page half of document.py; the PDF read and math path stay with
// the caller (Go carries no PDF engine).

import (
	"sort"
	"strings"
	"unicode"
)

// TextChar is one character of a text-layer run and its box in page pixels.
type TextChar struct {
	Ch  rune
	Box [4]float64
}

// TextRun is one text-layer run: the PDF segment between line breaks, its text
// and box in page pixels, in PDF content order. Chars, when present, give
// per-character boxes so a run can split between detected lines (e.g. across
// table cells); when empty, the whole Text is placed by the run box.
type TextRun struct {
	Text  string
	Box   [4]float64
	Chars []TextChar
}

// TextLayer is a page's PDF text layer, extracted by the caller. Runs are in
// content order; coordinates are page pixels matching the Gray passed to
// CallText.
type TextLayer struct {
	Runs []TextRun
	// Invisible marks a layer drawn invisibly (render mode 3): an OCR layer over
	// a scan, not trusted.
	Invisible bool
	// Rotated is the number of non-space characters not upright on the page.
	Rotated int
	// Chars is the non-space character count; 0 counts it from Runs.
	Chars int
}

// Thresholds from textlayer.py.
const (
	maxInvisibleFrac = 0.05
	maxBadCharFrac   = 0.005
	maxNoInkFrac     = 0.03
	inkDark          = 160 // gray level below which a pixel counts as ink
	readableMin      = 0.9 // share of the layer's letters the recognizer must know
)

// badCharCount is textlayer.bad_char_count: replacement, private-use, surrogate,
// unassigned or control characters (bar tab/newline) mark a broken font map.
func badCharCount(s string) int {
	n := 0
	for _, c := range s {
		if c == '\uFFFD' || unicode.Is(unicode.Co, c) || unicode.Is(unicode.Cs, c) ||
			unicode.Is(unicode.Cn, c) ||
			(unicode.IsControl(c) && c != '\t' && c != '\n' && c != '\r') {
			n++
		}
	}
	return n
}

func nonSpaceCount(s string) int {
	n := 0
	for _, c := range s {
		if !unicode.IsSpace(c) {
			n++
		}
	}
	return n
}

func layerChars(tl *TextLayer) int {
	if tl.Chars > 0 {
		return tl.Chars
	}
	n := 0
	for _, r := range tl.Runs {
		n += nonSpaceCount(r.Text)
	}
	return n
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func boxHasInk(g *Gray, b [4]float64) bool {
	x0 := clampInt(int(b[0]), 0, g.W)
	x1 := clampInt(int(b[2])+1, 0, g.W)
	y0 := clampInt(int(b[1]), 0, g.H)
	y1 := clampInt(int(b[3])+1, 0, g.H)
	for y := y0; y < y1; y++ {
		row := g.Pix[y*g.W : y*g.W+g.W]
		for x := x0; x < x1; x++ {
			if row[x] < inkDark {
				return true
			}
		}
	}
	return false
}

// noInkFrac is textlayer.no_ink_frac: share of runs whose box holds no dark pixel.
func noInkFrac(g *Gray, runs []TextRun) float64 {
	if len(runs) == 0 {
		return 0
	}
	empty := 0
	for _, r := range runs {
		if !boxHasInk(g, r.Box) {
			empty++
		}
	}
	return float64(empty) / float64(len(runs))
}

// readTextLayer applies the textlayer.read_text_layer checks. trusted=false
// means the caller should OCR the page instead.
func (o *OCR) readTextLayer(g *Gray, tl *TextLayer) (bool, string) {
	if tl.Invisible {
		return false, "invisible text (OCR layer)"
	}
	if len(tl.Runs) == 0 {
		return false, "no text layer"
	}
	if chars := layerChars(tl); chars > 0 {
		bad := 0
		for _, r := range tl.Runs {
			bad += badCharCount(r.Text)
		}
		if float64(bad) > maxBadCharFrac*float64(chars) {
			return false, "broken font map"
		}
	}
	if f := noInkFrac(g, tl.Runs); f > maxNoInkFrac {
		return false, "text runs with no ink"
	}
	return true, ""
}

// readableText is document.readable: whether the recognizer knows the layer's
// script, so OCR can fill lines the layer does not cover.
func (o *OCR) readableText(runs []TextRun) bool {
	letters, known := 0, 0
	for _, r := range runs {
		for _, c := range r.Text {
			if !unicode.IsLetter(c) {
				continue
			}
			letters++
			s := string(c)
			_, a := o.M.Vocab.Index[s]
			b := false
			if o.M.VocabCJK != nil {
				_, b = o.M.VocabCJK.Index[s]
			}
			if a || b {
				known++
			}
		}
	}
	return letters == 0 || float64(known) >= readableMin*float64(letters)
}

// strongRTL reports whether r is a strong right-to-left character (a subset of
// unicodedata.bidirectional in ("R", "AL")).
func strongRTL(r rune) bool {
	return unicode.In(r, unicode.Hebrew, unicode.Arabic, unicode.Syriac,
		unicode.Thaana, unicode.Nko, unicode.Samaritan, unicode.Mandaic)
}

func smallestContaining(boxes [][4]float64, area []float64, x, y float64) int {
	best, bestA := -1, 0.0
	for i, b := range boxes {
		if b[0] <= x && x <= b[2] && b[1] <= y && y <= b[3] {
			if best < 0 || area[i] < bestA {
				best, bestA = i, area[i]
			}
		}
	}
	return best
}

// assignRuns is document.assign_runs: the text of each detected box, from the
// runs whose characters fall inside it (whole-run when no per-char boxes);
// inside a box, runs are ordered by x, right to left in a right-to-left line.
// Returns the text per box and the count of characters outside every box.
func assignRuns(runs []TextRun, boxes [][4]float64) ([]string, int) {
	n := len(boxes)
	out := make([]string, n)
	if n == 0 {
		lost := 0
		for _, r := range runs {
			lost += nonSpaceCount(r.Text)
		}
		return out, lost
	}
	area := make([]float64, n)
	for i, b := range boxes {
		area[i] = (b[2] - b[0]) * (b[3] - b[1])
	}
	type part struct {
		text   string
		x0, x1 float64
	}
	parts := make([][]part, n)
	lost := 0
	for _, run := range runs {
		assign := map[int][]rune{}
		var order []int
		if len(run.Chars) > 0 {
			for _, tc := range run.Chars {
				if unicode.IsSpace(tc.Ch) {
					continue
				}
				cx := (tc.Box[0] + tc.Box[2]) / 2
				cy := (tc.Box[1] + tc.Box[3]) / 2
				j := smallestContaining(boxes, area, cx, cy)
				if j < 0 {
					lost++
					continue
				}
				if _, ok := assign[j]; !ok {
					order = append(order, j)
				}
				assign[j] = append(assign[j], tc.Ch)
			}
		} else {
			cx := (run.Box[0] + run.Box[2]) / 2
			cy := (run.Box[1] + run.Box[3]) / 2
			j := smallestContaining(boxes, area, cx, cy)
			if j < 0 {
				lost += nonSpaceCount(run.Text)
				continue
			}
			order = append(order, j)
			assign[j] = []rune(run.Text)
		}
		for _, j := range order {
			parts[j] = append(parts[j], part{string(assign[j]), run.Box[0], run.Box[2]})
		}
	}
	for i, ps := range parts {
		if len(ps) == 0 {
			continue
		}
		rtl, ltr := 0, 0
		for _, p := range ps {
			for _, c := range p.text {
				if strongRTL(c) {
					rtl++
				} else if unicode.IsLetter(c) {
					ltr++
				}
			}
		}
		if rtl > ltr {
			sort.SliceStable(ps, func(a, b int) bool { return ps[a].x1 > ps[b].x1 })
		} else {
			sort.SliceStable(ps, func(a, b int) bool { return ps[a].x0 < ps[b].x0 })
		}
		var sb strings.Builder
		for k, p := range ps {
			if k > 0 {
				sb.WriteByte(' ')
			}
			sb.WriteString(p.text)
		}
		out[i] = sb.String()
	}
	return out, lost
}
