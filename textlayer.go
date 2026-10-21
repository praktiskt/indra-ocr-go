package indra

// PDF text-layer support: the caller extracts a page's text runs with any PDF
// library and hands them to CallText, which uses them where they are sound and
// OCRs the rest (the upstream "auto" path). Faithful to indra_ocr/textlayer.py
// and the text-page half of document.py; the PDF read and math path stay with
// the caller (Go carries no PDF engine) — but CallPDF drives the embedded
// engine through the same code.

import (
	"math"
	"sort"
	"unicode"

	"golang.org/x/text/unicode/bidi"
	"golang.org/x/text/unicode/norm"
)

// TextChar is one character of a text-layer run and its box in page pixels.
type TextChar struct {
	Ch  rune
	Box [4]float64
}

// TextRun is one text-layer run: the PDF segment between line breaks, its text
// and box in page pixels, in PDF content order. Chars, when present, give
// per-character boxes so a run can split between detected lines (e.g. across
// table cells); when empty, the whole Text is placed by the run box. Quarter is
// the quarter turns from upright in the (turned) frame (0 = upright); Seq is the
// run's position in the page's content order (all runs, upright and sideways).
type TextRun struct {
	Text    string
	Box     [4]float64
	Chars   []TextChar
	Quarter int
	Seq     int
}

// TextStats are the extraction statistics that drive the trust checks and the
// page-turn / vertical-layout decisions (upstream read_text_layer's stats dict).
type TextStats struct {
	Chars    int // non-space characters
	Rotated  int // non-space characters not upright on the page
	Bad      int // bad (broken-font-map) characters among them
	Quarters [4]int
	Steps    [2]int // neighbouring upright characters: down (vertical), right
	OffPage  int
	Turn     int
	Frame    int
	Vertical bool
}

// verticalLayout is textlayer.vertical_layout: most steps between neighbouring
// upright characters go down, not right (vertical writing).
func verticalLayout(st TextStats) bool {
	down, right := st.Steps[0], st.Steps[1]
	return down >= verticalDownMin && down > 2*right
}

// pageTurn is textlayer.page_turn: the quarter turn that holds most of the
// page's characters (a table set sideways on a portrait page), or 0.
func pageTurn(st TextStats) int {
	k := 1
	for i := 2; i < 4; i++ {
		if st.Quarters[i] > st.Quarters[k] {
			k = i
		}
	}
	if float64(st.Quarters[k]) > 0.5*float64(max(st.Chars, 1)) {
		return k
	}
	return 0
}

// TextLayer is a page's PDF text layer. Runs are the upright runs in content
// order; Rotated holds the sideways runs (a rotated table header, an axis
// label). Coordinates are page pixels matching the Gray passed to CallText.
type TextLayer struct {
	Runs    []TextRun
	Rotated []TextRun
	// Invisible marks a layer drawn invisibly (render mode 3): an OCR layer over
	// a scan, not trusted.
	Invisible bool
	// Chars is the non-space character count; 0 counts it from Runs.
	Chars int
	Stats TextStats
	// Trusted and Reason are read_text_layer's verdict on the layer.
	Trusted bool
	Reason  string
}

// Thresholds from textlayer.py.
const (
	maxInvisibleFrac = 0.05
	maxBadCharFrac   = 0.005
	maxNoInkFrac     = 0.03
	maxVisualRTLFrac = 0.25
	minRTLPairs      = 20
	inkDark          = 160 // gray level below which a pixel counts as ink
	readableMin      = 0.9 // share of the layer's letters the recognizer must know
	agreeLines       = 6   // text-layer lines whose script is checked against the image
	agreeShare       = 0.25
	latinCodes       = 0.9
	verticalDownMin  = 30
	// brokenFontMap is textlayer.BROKEN_FONT_MAP: the layer's codes are not text.
	// A broken map still places glyphs, so it is treated specially (no "unreadable"
	// override, and the page is known upright when the glyphs are).
	brokenFontMap = "broken font map"
	// visualRTL is textlayer.VISUAL_RTL: right-to-left text stored in visual order.
	visualRTL  = "right-to-left text in visual order"
	notThePage = "text layer does not match the page"
)

// badCharCount is textlayer.bad_char_count: replacement, private-use, surrogate,
// unassigned or control characters (bar tab) mark a broken font map.
func badCharCount(s string) int {
	n := 0
	for _, c := range s {
		if c == '\uFFFD' || unicode.Is(unicode.Co, c) || unicode.Is(unicode.Cs, c) ||
			unicode.Is(unicode.Cn, c) ||
			(unicode.IsControl(c) && c != '\t') {
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

// visualRTLFrac is textlayer.visual_rtl_frac: neighbouring right-to-left letters
// inside a run, and the share of them that step left to right on the page.
func visualRTLFrac(runs []TextRun) (int, float64) {
	pairs, right := 0, 0
	for _, r := range runs {
		for i := 1; i < len(r.Chars); i++ {
			a, b := r.Chars[i-1], r.Chars[i]
			if isRTLChar(a.Ch) && isRTLChar(b.Ch) {
				pairs++
				if b.Box[0]+b.Box[2] > a.Box[0]+a.Box[2] {
					right++
				}
			}
		}
	}
	return pairs, float64(right) / float64(max(pairs, 1))
}

// layerVerdict is textlayer.read_text_layer's checks on an extracted layer.
// trusted=false means the caller should OCR the page instead.
func (o *OCR) layerVerdict(g *Gray, tl *TextLayer) (bool, string) {
	if tl.Invisible {
		return false, "invisible text (OCR layer)"
	}
	if len(tl.Runs) == 0 {
		return false, "no text layer"
	}
	chars := tl.Chars
	if chars <= 0 {
		chars = layerChars(tl)
	}
	bad := tl.Stats.Bad
	if tl.Stats.Chars == 0 {
		// a caller-built layer carries no stats: count from the runs
		bad = 0
		for _, r := range tl.Runs {
			bad += badCharCount(r.Text)
		}
	}
	if chars > 0 && float64(bad) > maxBadCharFrac*float64(chars) {
		return false, brokenFontMap
	}
	if tl.Stats.Frame == tl.Stats.Turn {
		// Only upright text has a reading direction on the page.
		if pairs, visual := visualRTLFrac(tl.Runs); pairs >= minRTLPairs && visual > maxVisualRTLFrac {
			return false, visualRTL
		}
	}
	if f := noInkFrac(g, tl.Runs); f > maxNoInkFrac {
		return false, "text runs with no ink"
	}
	return true, ""
}

// readTextLayer applies the textlayer.read_text_layer checks. trusted=false
// means the caller should OCR the page instead.
func (o *OCR) readTextLayer(g *Gray, tl *TextLayer) (bool, string) {
	return o.layerVerdict(g, tl)
}

// readableText is document.readable: whether the recognizer knows the layer's
// script, so OCR can fill lines the layer does not cover. A recognizer that
// reads Hangul as jamo knows a syllable by its jamo.
func (o *OCR) readableText(runs []TextRun) bool {
	var vocabs []*vocab
	if o.m != nil && o.m.vocab != nil {
		vocabs = append(vocabs, o.m.vocab)
	}
	if o.m != nil {
		for _, g := range o.m.scriptGroups {
			if g == "base" {
				continue
			}
			if v := o.m.groupVocab(g); v != nil {
				vocabs = append(vocabs, v)
			}
		}
	}
	letters, known := 0, 0
	for _, r := range runs {
		for _, c := range r.Text {
			if !unicode.IsLetter(c) {
				continue
			}
			letters++
			if knownInAny(string(c), vocabs) {
				known++
			}
		}
	}
	return letters == 0 || float64(known) >= readableMin*float64(letters)
}

func knownInAny(c string, vocabs []*vocab) bool {
	for _, v := range vocabs {
		if _, ok := v.Index[c]; ok {
			return true
		}
	}
	all := true
	for _, j := range norm.NFD.String(c) {
		if _, ok := vIndexAll(j, vocabs); !ok {
			all = false
			break
		}
	}
	return all
}

func vIndexAll(r rune, vocabs []*vocab) (int, bool) {
	s := string(r)
	for _, v := range vocabs {
		if id, ok := v.Index[s]; ok {
			return id, true
		}
	}
	return 0, false
}

// strongLTR reports the bidi class L (left-to-right letters). document's
// right_to_left counts only L, not digits.
func strongLTR(c rune) bool {
	p, _ := bidi.LookupRune(c)
	return p.Class() == bidi.L
}

// rightToLeft is document.right_to_left over tokens: the Unicode bidi base
// direction (most strong characters are right-to-left).
func rightToLeft(toks []token) bool {
	rtl, ltr := 0, 0
	for _, t := range toks {
		r := []rune(t.Ch)
		if len(r) != 1 {
			continue
		}
		if isRTLChar(r[0]) {
			rtl++
		} else if strongLTR(r[0]) {
			ltr++
		}
	}
	return rtl > ltr
}

// latinShare is document.latin_share: share of letters that are basic Latin.
func latinShare(text string) float64 {
	letters, ascii := 0, 0
	for _, c := range text {
		if !unicode.IsLetter(c) {
			continue
		}
		letters++
		if c < 128 {
			ascii++
		}
	}
	return float64(ascii) / float64(max(letters, 1))
}

// layerAgrees is document.layer_agrees: at least agreeShare of the checked lines
// agree (a legacy font stores Latin codes for another script, so almost every
// line disagrees).
func layerAgrees(agree []bool) bool {
	if len(agree) == 0 {
		return true
	}
	n := 0
	for _, a := range agree {
		if a {
			n++
		}
	}
	return float64(n) >= agreeShare*float64(len(agree))
}

// boxCovered is document._covered: whether more than frac of box lies inside one
// of boxes.
func boxCovered(box [4]float64, boxes [][4]float64, frac float64) bool {
	area := math.Max((box[2]-box[0])*(box[3]-box[1]), 1e-6)
	for _, o := range boxes {
		ix := math.Min(box[2], o[2]) - math.Max(box[0], o[0])
		iy := math.Min(box[3], o[3]) - math.Max(box[1], o[1])
		if ix < 0 {
			ix = 0
		}
		if iy < 0 {
			iy = 0
		}
		if ix*iy/area > frac {
			return true
		}
	}
	return false
}

// smallestContaining returns the smallest box containing (x, y), or -1.
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

// runTokens builds (char, x, content index) tokens for a run placed by whole-run
// box (no per-character boxes): every character sits at the run box's center.
func runTokens(run TextRun, idx int) []token {
	cx := (run.Box[0] + run.Box[2]) / 2
	var out []token
	for _, c := range run.Text {
		out = append(out, token{Ch: string(c), X: cx, Seq: idx, hasSeq: true})
	}
	return out
}

// assignRuns is document.assign_runs: tokens (char, x, content index) per
// detected line. Each character goes to the smallest detected box that contains
// its center (a space follows the character before it), so a run that spans two
// table cells splits between them. Inside a detected line, each run's part keeps
// the PDF character order and the parts are ordered by x, right to left in a
// right-to-left line. Returns (tokens per detected line, characters outside
// every box).
func assignRuns(runs []TextRun, boxes [][4]float64) ([][]token, int) {
	n := len(boxes)
	if n == 0 {
		lost := 0
		for _, r := range runs {
			if len(r.Chars) > 0 {
				for _, c := range r.Chars {
					if !unicode.IsSpace(c.Ch) {
						lost++
					}
				}
			} else {
				lost += nonSpaceCount(r.Text)
			}
		}
		return nil, lost
	}
	area := make([]float64, n)
	for i, b := range boxes {
		area[i] = (b[2] - b[0]) * (b[3] - b[1])
	}
	parts := make([]map[int][]token, n)
	for i := range parts {
		parts[i] = map[int][]token{}
	}
	lost := 0
	for pos, run := range runs {
		idx := run.Seq
		if len(run.Chars) == 0 {
			idx = pos
		}
		j := -1
		if len(run.Chars) == 0 {
			cx := (run.Box[0] + run.Box[2]) / 2
			cy := (run.Box[1] + run.Box[3]) / 2
			j = smallestContaining(boxes, area, cx, cy)
			if j < 0 {
				lost += nonSpaceCount(run.Text)
			} else {
				parts[j][idx] = runTokens(run, idx)
			}
			continue
		}
		for _, tc := range run.Chars {
			if unicode.IsSpace(tc.Ch) {
				if j >= 0 {
					if prev := parts[j][idx]; len(prev) > 0 {
						parts[j][idx] = append(prev, token{Ch: string(tc.Ch), X: prev[len(prev)-1].X, Seq: idx, hasSeq: true})
					}
				}
				continue
			}
			cx := (tc.Box[0] + tc.Box[2]) / 2
			cy := (tc.Box[1] + tc.Box[3]) / 2
			j = smallestContaining(boxes, area, cx, cy)
			if j < 0 {
				lost++
				j = -1
				continue
			}
			parts[j][idx] = append(parts[j][idx], token{Ch: string(tc.Ch), X: cx, Seq: idx, hasSeq: true})
		}
	}
	toks := make([][]token, n)
	for i, line := range parts {
		if len(line) == 0 {
			continue
		}
		var all []token
		for _, part := range line {
			all = append(all, part...)
		}
		var idxs []int
		for idx := range line {
			idxs = append(idxs, idx)
		}
		if rightToLeft(all) {
			sort.SliceStable(idxs, func(a, b int) bool { return maxTokX(line[idxs[a]]) > maxTokX(line[idxs[b]]) })
		} else {
			sort.SliceStable(idxs, func(a, b int) bool { return minTokX(line[idxs[a]]) < minTokX(line[idxs[b]]) })
		}
		var tk []token
		for _, idx := range idxs {
			if len(tk) > 0 {
				tk = append(tk, token{Ch: " ", X: tk[len(tk)-1].X, Seq: idx, hasSeq: true})
			}
			tk = append(tk, line[idx]...)
		}
		toks[i] = tk
	}
	return toks, lost
}

func maxTokX(ts []token) float64 {
	m := ts[0].X
	for _, t := range ts[1:] {
		if t.X > m {
			m = t.X
		}
	}
	return m
}

func minTokX(ts []token) float64 {
	m := ts[0].X
	for _, t := range ts[1:] {
		if t.X < m {
			m = t.X
		}
	}
	return m
}
