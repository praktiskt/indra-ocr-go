// OCR pipeline over Sessions. Port of OrtOCR in pipeline.py (sans mask.onnx,
// script/rec_cjk routing and math: Latin/multi path only).
package indra

import (
	"math"
	"sort"
	"strings"
)

const (
	// MinConf drops line text under this mean CTC peak probability (box stays).
	MinConf = 0.5
	// OrientMin applies a turn only above this probability.
	OrientMin = 0.95
	// SplitGap cuts a detected line at wider gaps; CellGap splits table cells.
	SplitGap = 3.0
	CellGap  = 1.8
	// Adaptive second detection pass for small text.
	adaptiveMinH    = 9.0
	adaptiveTargetH = 12.0
	adaptiveMaxSide = 2048
	deskewMin       = 0.3
)

// Line is a recognized line in reading order.
type Line struct {
	Box  [4]float64
	Text string
	Conf float64
}

// Page is the OCR result of one page image.
type Page struct {
	Lines    []Line
	Markdown string
	Blocks   []Block
	Source   string // "ocr"
	Rotation int
	Skew     float64
}

// OCR binds sessions + meta + vocab.
type OCR struct {
	S *Sessions
	M *Meta
	V *Vocab
}

// NewOCR opens sessions from embedded models.
func NewOCR(threads int) (*OCR, error) {
	raw, err := modelFS.ReadFile("models/meta.json")
	if err != nil {
		return nil, err
	}
	m, err := ParseMeta(raw)
	if err != nil {
		return nil, err
	}
	s, err := NewSessions(threads)
	if err != nil {
		return nil, err
	}
	return &OCR{S: s, M: m, V: m.Vocab}, nil
}

// Close releases sessions.
func (o *OCR) Close() { o.S.Close() }

// Token is (id, page-x) from a CTC peak.
type Token struct {
	ID int
	X  float64
}

// recLine is (text, conf, tokens) per detected line.
type recLine struct {
	text string
	conf float64
	toks []Token
}

// Detect runs det.onnx (+ adaptive second pass), returns boxes in page px.
func (o *OCR) Detect(g *Gray) [][4]float32 {
	ink, pw, ph, scale := PageToInput(g, o.M.DetLongSide)
	raw, oh, ow, err := o.S.RunDet(ink, ph, pw)
	if err != nil {
		return nil
	}
	boxes := DecodeDet(raw, oh, ow)
	if len(boxes) >= 8 {
		hs := make([]float64, len(boxes))
		for i, b := range boxes {
			hs[i] = float64(b.Y1 - b.Y0)
		}
		sort.Float64s(hs)
		med := npMedian(hs)
		if med < adaptiveMinH {
			ls := min(adaptiveMaxSide, int(float64(o.M.DetLongSide)*adaptiveTargetH/math.Max(med, 1))) / 32 * 32
			if ls > o.M.DetLongSide {
				ink2, pw2, ph2, scale2 := PageToInput(g, ls)
				raw2, oh2, ow2, err2 := o.S.RunDet(ink2, ph2, pw2)
				if err2 == nil {
					boxes = DecodeDet(raw2, oh2, ow2)
					scale = scale2
				}
			}
		}
	}
	out := make([][4]float32, 0, len(boxes))
	for _, b := range boxes {
		x0 := clampf32(b.X0/float32(scale), 0, float32(g.W))
		y0 := clampf32(b.Y0/float32(scale), 0, float32(g.H))
		x1 := clampf32(b.X1/float32(scale), 0, float32(g.W))
		y1 := clampf32(b.Y1/float32(scale), 0, float32(g.H))
		out = append(out, [4]float32{x0, y0, x1, y1})
	}
	return out
}

func clampf32(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// Recognize runs rec.onnx over line boxes, returns (text, conf, tokens).
func (o *OCR) Recognize(g *Gray, boxes [][4]float32) []recLine {
	n := len(boxes)
	out := make([]recLine, n)
	if n == 0 {
		return out
	}
	// Crop all lines.
	type crop struct {
		ink []float32
		w   int
	}
	crops := make([]crop, n)
	widths := make([]int, n)
	for i, b := range boxes {
		fb := [4]float64{float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}
		ink := CropLine(g, fb, o.M.RecHeight)
		w := len(ink) / o.M.RecHeight
		crops[i] = crop{ink, w}
		widths[i] = w
	}
	for _, batch := range WidthBatches(widths, 64) {
		idx := batch
		lines := make([][]float32, len(idx))
		hs := make([]int, len(idx))
		ws := make([]int, len(idx))
		for k, j := range idx {
			lines[k] = crops[j].ink
			hs[k] = o.M.RecHeight
			ws[k] = crops[j].w
		}
		batched, wOut, wm := PadBatch(lines, hs, ws)
		ro, err := o.S.RunRec(batched, len(idx), o.M.RecHeight, wm, wOut)
		if err != nil {
			continue
		}
		stride := float64(wm) / math.Max(float64(ro.T), 1)
		for k, j := range idx {
			ln := int(ro.Lens[k])
			var seq, pm []float64
			var ids []int
			for t := 0; t < ln && t < ro.T; t++ {
				ids = append(ids, int(ro.IDs[k*ro.T+t]))
				pm = append(pm, float64(ro.Conf[k*ro.T+t]))
			}
			// CTC collapse: drop blanks; collapse repeats unless separated
			// by a blank (mirrors pipeline.py: keep = (seq != 0) &
			// concatenate([True], seq[1:] != seq[:-1])).
			var kept []int
			var keptP []float64
			var keptT []int
			prev := -1
			for t, id := range ids {
				if id != 0 && id != prev {
					kept = append(kept, id)
					keptP = append(keptP, pm[t])
					keptT = append(keptT, t)
				}
				prev = id
			}
			_ = seq
			b := boxes[j]
			h := math.Max(1, float64(b[3]-b[1]))
			ix0 := math.Floor(float64(b[0]) - PadX*h)
			cw := math.Ceil(float64(b[2])+PadX*h) - ix0
			f := cw / float64(crops[j].w)
			var toks []Token
			var xs []float64
			for q, id := range kept {
				xs = append(xs, ix0+(float64(keptT[q])+0.5)*stride*f)
				toks = append(toks, Token{ID: id, X: xs[len(xs)-1]})
			}
			conf := 0.0
			if len(keptP) > 0 {
				for _, p := range keptP {
					conf += p
				}
				conf /= float64(len(keptP))
			}
			if conf >= MinConf {
				out[j] = recLine{text: o.V.Decode(kept), conf: conf, toks: toks}
			} else {
				out[j] = recLine{text: "", conf: conf}
			}
		}
	}
	return out
}

// Orientation returns quarter turns CCW from orient.onnx (0 without model).
func (o *OCR) Orientation(g *Gray) int {
	if o.S.Orient == nil {
		return 0
	}
	ink := OrientInput(g, 768)
	p, err := o.S.RunOrient(ink)
	if err != nil {
		return 0
	}
	k := 0
	for i := 1; i < 4; i++ {
		if p[i] > p[k] {
			k = i
		}
	}
	if p[k] > OrientMin {
		return k
	}
	return 0
}

// Upright turns the page upright and straightens tilt.
func (o *OCR) Upright(g *Gray) (*Gray, int, float64) {
	k := o.Orientation(g)
	if k != 0 {
		g = Rot90CCW(g, -k)
	}
	a := EstimateSkew(g)
	if math.Abs(a) >= deskewMin {
		g = RotatePage(g, a)
	} else {
		a = 0
	}
	return g, k, a
}

// Call OCRs one page image.
func (o *OCR) Call(g *Gray) *Page {
	g, k, skew := o.Upright(g)
	boxes := o.Detect(g)
	rec := o.Recognize(g, boxes)
	wh := [2]float64{float64(g.W), float64(g.H)}
	p := o.Assemble(wh, boxes, rec)
	p.Rotation, p.Skew = 90*k, math.Round(skew*100)/100
	if skew != 0 {
		for i := range p.Lines {
			p.Lines[i].Box = UnrotateBox(p.Lines[i].Box, skew, g.W, g.H)
		}
		for i := range p.Blocks {
			p.Blocks[i].BBox = UnrotateBox(p.Blocks[i].BBox, skew, g.W, g.H)
		}
	}
	return p
}

// Assemble splits wide-gap lines, orders, and renders markdown.
func (o *OCR) Assemble(wh [2]float64, boxes [][4]float32, rec []recLine) *Page {
	// Gutter cuts + SPLIT_GAP split.
	fb := make([][4]float64, len(boxes))
	for i, b := range boxes {
		fb[i] = [4]float64{float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}
	}
	cuts := gutterCuts(o.V, fb, rec)
	var nb [][4]float64
	var nr []recLine
	for i := range fb {
		var at []float64
		if i < len(cuts) {
			at = cuts[i]
		}
		for _, part := range splitLine(o.V, fb[i], rec[i], SplitGap, at) {
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
		p.Blocks = PageBlocks(fb, texts, order, clsM, sameM)
		cells := map[int][][2]string{}
		cellBoxes := map[int][][4]float64{}
		for i := range fb {
			parts := splitLine(o.V, fb[i], rec[i], CellGap, nil)
			if len(parts) > 1 {
				for _, pt := range parts {
					cells[i] = append(cells[i], [2]string{"", pt.rec.text})
					cellBoxes[i] = append(cellBoxes[i], pt.box)
				}
			}
		}
		p.Markdown = PageMarkdown(fb, texts, order, clsM, sameM, false, cells, cellBoxes)
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
	if n < 2 || o.S.Order == nil {
		return OrderXYCut(boxes), nil, nil
	}
	feats := LineFeatures(boxes, wh[0], wh[1], texts)
	out, err := o.S.RunOrder(feats, n)
	if err != nil {
		return OrderXYCut(boxes), nil, nil
	}
	a := reshape2(out.A, n, n)
	same := reshape2(out.Same, n, n)
	cls := reshape2(out.Cls, n, len(BlockClasses))
	return DecodeBlocksOrder(a, same, boxes, cls), cls, same
}

func reshape2(f []float32, r, c int) [][]float64 {
	m := make([][]float64, r)
	for i := range m {
		m[i] = make([]float64, c)
		for j := 0; j < c && i*c+j < len(f); j++ {
			m[i][j] = float64(f[i*c+j])
		}
	}
	return m
}

type splitPart struct {
	box [4]float64
	rec recLine
}

func splitLine(v *Vocab, box [4]float64, r recLine, gap float64, at []float64) []splitPart {
	h := math.Max(1, box[3]-box[1])
	spaceID := v.SpaceID()
	var segs [][]Token
	cur := []Token{}
	prev := math.NaN()
	var segToks [][]Token
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
				cur = []Token{}
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
			if tk.ID == 0 || tk.ID == spaceID {
				continue
			}
			ids = append(ids, tk.ID)
			xs = append(xs, tk.X)
		}
		text := v.Decode(ids)
		if strings.TrimSpace(text) != "" && len(xs) > 0 {
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

func gutterCuts(v *Vocab, boxes [][4]float64, rec []recLine) [][]float64 {
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
	for i := range boxes {
		h := math.Max(1, boxes[i][3]-boxes[i][1])
		var xs []float64
		for _, tk := range rec[i].toks {
			if tk.ID == 0 || tk.ID == spaceID {
				continue
			}
			xs = append(xs, tk.X)
		}
		for k := 0; k+1 < len(xs); k++ {
			a, c := xs[k], xs[k+1]
			if c-a <= 1.6*h {
				continue
			}
			gl, gr := a+0.35*h, c-0.35*h
			near := 0
			for j := range boxes {
				if j == i {
					continue
				}
				if math.Abs(rights[j]-gl) < 0.5*h || math.Abs(lefts[j]-gr) < 0.5*h {
					near++
				}
			}
			if near >= 3 {
				cuts[i] = append(cuts[i], (a+c)/2)
			}
		}
	}
	return cuts
}
