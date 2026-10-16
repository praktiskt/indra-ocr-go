// OCR pipeline over Sessions. Port of OrtOCR in pipeline.py (sans mask.onnx and
// math: Latin/multi + CJK routing).
package indra

import (
	"math"
	"runtime"
	"sort"
	"strings"
	"sync"
)

const (
	// MinConf drops line text under this mean CTC peak probability (box stays).
	MinConf = 0.5
	// OrientMin applies a turn only above this probability. Photo receipts
	// with dark borders confuse orient.onnx into mid-confidence votes
	// (e.g. 0.954 for 180deg on an upright photo); true rotations score
	// >=0.998, so the bar sits between them.
	OrientMin = 0.98
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

// Text is the reading-order line texts joined by newlines (upstream Page.text).
func (p *Page) Text() string {
	out := make([]string, len(p.Lines))
	for i, l := range p.Lines {
		out[i] = l.Text
	}
	return strings.Join(out, "\n")
}

// OCR binds sessions + meta + vocab.
type OCR struct {
	S *Sessions
	M *Meta
	V *Vocab
	// Furniture keeps page headers and footers in Markdown (upstream
	// load(furniture=True)). The default, false, drops them. Set before Call.
	Furniture bool
}

// NewOCR opens sessions from embedded models.
func NewOCR(threads int) (*OCR, error) {
	return NewOCRWithThreads(DefaultThreads(threads))
}

// NewOCRWithThreads opens sessions with per-model thread counts.
func NewOCRWithThreads(tc Threads) (*OCR, error) {
	raw, err := modelFS.ReadFile("models/meta.json")
	if err != nil {
		return nil, err
	}
	m, err := ParseMeta(raw)
	if err != nil {
		return nil, err
	}
	s, err := NewSessionsWithThreads(tc)
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

// lineCrop is one line's grayscale crop at rec height.
type lineCrop struct {
	ink []float32
	w   int
}

// cropLines crops every box to rec height (in parallel for large pages).
func (o *OCR) cropLines(g *Gray, boxes [][4]float32) []lineCrop {
	n := len(boxes)
	crops := make([]lineCrop, n)
	if n >= 16 {
		var wg sync.WaitGroup
		sem := make(chan struct{}, runtime.NumCPU())
		for i, b := range boxes {
			wg.Add(1)
			sem <- struct{}{}
			go func(i int, b [4]float32) {
				defer wg.Done()
				fb := [4]float64{float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}
				ink := CropLine(g, fb, o.M.RecHeight)
				crops[i] = lineCrop{ink, len(ink) / o.M.RecHeight}
				<-sem
			}(i, b)
		}
		wg.Wait()
	} else {
		for i, b := range boxes {
			fb := [4]float64{float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}
			ink := CropLine(g, fb, o.M.RecHeight)
			crops[i] = lineCrop{ink, len(ink) / o.M.RecHeight}
		}
	}
	return crops
}

// hanziLines runs script.onnx over the crops: true where the line holds Chinese
// characters (upstream hanzi_lines). All false without script.onnx.
func (o *OCR) hanziLines(crops []lineCrop) []bool {
	out := make([]bool, len(crops))
	if len(crops) == 0 || o.S.Script == nil || o.M.VocabCJK == nil {
		return out
	}
	ws := make([]int, len(crops))
	for i := range crops {
		ws[i] = crops[i].w
	}
	for _, part := range WidthBatches(ws, 512) {
		lines := make([][]float32, len(part))
		hs := make([]int, len(part))
		bw := make([]int, len(part))
		for k, j := range part {
			lines[k] = crops[j].ink
			hs[k] = o.M.RecHeight
			bw[k] = crops[j].w
		}
		batched, wOut, wm := PadBatch(lines, hs, bw)
		p, err := o.S.RunScript(batched, len(part), o.M.RecHeight, wm, wOut)
		if err != nil {
			continue
		}
		for k, j := range part {
			out[j] = p[k] > 0.5
		}
	}
	return out
}

// recognizeSel runs one recognizer session over crops[sel], writing out[sel].
// Fine-grained width batches create many distinct (B,Wm) session shapes; ORT
// retains per-shape allocations (~the whole leak), so batch on a coarse 512px
// grid: bit-identical output, equal speed, less RSS.
func (o *OCR) recognizeSel(out []recLine, boxes [][4]float32, crops []lineCrop, sel []int,
	run func([]float32, int, int, int, []int64) (*RecOut, error), vocab *Vocab) {
	if len(sel) == 0 {
		return
	}
	ws := make([]int, len(sel))
	for k, j := range sel {
		ws[k] = crops[j].w
	}
	for _, part := range WidthBatches(ws, 512) {
		lines := make([][]float32, len(part))
		hs := make([]int, len(part))
		bw := make([]int, len(part))
		for k, p := range part {
			j := sel[p]
			lines[k] = crops[j].ink
			hs[k] = o.M.RecHeight
			bw[k] = crops[j].w
		}
		batched, wOut, wm := PadBatch(lines, hs, bw)
		ro, err := run(batched, len(part), o.M.RecHeight, wm, wOut)
		if err != nil {
			continue
		}
		stride := float64(wm) / math.Max(float64(ro.T), 1)
		for k, p := range part {
			j := sel[p]
			ln := int(ro.Lens[k])
			if ln > ro.T {
				ln = ro.T
			}
			ids := make([]int, 0, ln)
			pm := make([]float64, 0, ln)
			base := k * ro.T
			for t := 0; t < ln; t++ {
				ids = append(ids, int(ro.IDs[base+t]))
				pm = append(pm, float64(ro.Conf[base+t]))
			}
			// CTC collapse: drop blanks; collapse repeats unless separated
			// by a blank (mirrors pipeline.py: keep = (seq != 0) &
			// concatenate([True], seq[1:] != seq[:-1])).
			kept := make([]int, 0, len(ids))
			keptP := make([]float64, 0, len(ids))
			keptT := make([]int, 0, len(ids))
			prev := -1
			for t, id := range ids {
				if id != 0 && id != prev {
					kept = append(kept, id)
					keptP = append(keptP, pm[t])
					keptT = append(keptT, t)
				}
				prev = id
			}
			b := boxes[j]
			h := math.Max(1, float64(b[3]-b[1]))
			ix0 := math.Floor(float64(b[0]) - PadX*h)
			cw := math.Ceil(float64(b[2])+PadX*h) - ix0
			f := cw / float64(crops[j].w)
			toks := make([]Token, 0, len(kept))
			for q, id := range kept {
				x := ix0 + (float64(keptT[q])+0.5)*stride*f
				toks = append(toks, Token{ID: id, X: x})
			}
			conf := 0.0
			if len(keptP) > 0 {
				for _, pp := range keptP {
					conf += pp
				}
				conf /= float64(len(keptP))
			}
			if conf >= MinConf {
				out[j] = recLine{text: vocab.Decode(kept), conf: conf, toks: toks}
			} else {
				out[j] = recLine{text: "", conf: conf}
			}
		}
	}
}

// Recognize runs the recognizer over line boxes, routing lines the script
// classifier (script.onnx) calls Chinese to rec_cjk.onnx (upstream recognize),
// returns (text, conf, tokens).
func (o *OCR) Recognize(g *Gray, boxes [][4]float32) []recLine {
	n := len(boxes)
	out := make([]recLine, n)
	if n == 0 {
		return out
	}
	crops := o.cropLines(g, boxes)
	hanzi := o.hanziLines(crops)
	var latin, cjk []int
	for j, h := range hanzi {
		if h {
			cjk = append(cjk, j)
		} else {
			latin = append(latin, j)
		}
	}
	o.recognizeSel(out, boxes, crops, latin, o.S.RunRec, o.V)
	if len(cjk) > 0 {
		o.recognizeSel(out, boxes, crops, cjk, o.S.RunRecCJK, o.M.VocabCJK)
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
// Orientation (model) and skew search (CPU) run concurrently on the
// common upright path; rotated pages re-estimate skew after rotation.
func (o *OCR) Upright(g *Gray) (*Gray, int, float64) {
	kCh := make(chan int, 1)
	aCh := make(chan float64, 1)
	go func() { kCh <- o.Orientation(g) }()
	go func() { aCh <- EstimateSkewPy(g) }()
	k := <-kCh
	var a float64
	if k != 0 {
		<-aCh // discard pre-rotation estimate
		g = Rot90CCW(g, -k)
		a = EstimateSkewPy(g)
	} else {
		a = <-aCh
	}
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
	return o.finish(g, k, skew, boxes, rec)
}

func f64boxes(bs [][4]float32) [][4]float64 {
	out := make([][4]float64, len(bs))
	for i, b := range bs {
		out[i] = [4]float64{float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}
	}
	return out
}

// CallText OCRs a page image but uses the caller's PDF text layer where it is
// sound (upstream mode="auto"): trusted runs supply each detected line's text,
// and lines the layer does not cover are OCR'd (hybrid). Falls back to plain
// OCR when the layer is not trusted (scan, invisible layer, broken font map,
// runs with no ink). Ordering uses the order model; upstream's pdf_order
// (content order) is not ported. Source is "text" or "hybrid".
func (o *OCR) CallText(g *Gray, tl *TextLayer) *Page {
	if trusted, _ := o.readTextLayer(g, tl); !trusted {
		return o.Call(g)
	}
	g2, k, skew := o.Upright(g)
	boxes := o.Detect(g2)
	txt, _ := assignRuns(tl.Runs, f64boxes(boxes))
	rec := make([]recLine, len(boxes))
	var empty [][4]float32
	var emptyAt []int
	for i := range boxes {
		if strings.TrimSpace(txt[i]) != "" {
			rec[i] = recLine{text: txt[i], conf: 1.0}
		} else {
			empty = append(empty, boxes[i])
			emptyAt = append(emptyAt, i)
		}
	}
	source := "text"
	if len(empty) > 0 && o.readableText(tl.Runs) {
		for j, r := range o.Recognize(g2, empty) {
			if strings.TrimSpace(r.text) != "" {
				rec[emptyAt[j]] = r
				source = "hybrid"
			}
		}
	}
	var kb [][4]float32
	var kr []recLine
	for i := range boxes {
		if rec[i].text != "" {
			kb = append(kb, boxes[i])
			kr = append(kr, rec[i])
		}
	}
	p := o.finish(g2, k, skew, kb, kr)
	p.Source = source
	return p
}

// finish assembles results and undoes deskew on the output geometry.
func (o *OCR) finish(g *Gray, k int, skew float64, boxes [][4]float32, rec []recLine) *Page {
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

// CallAll OCRs pages in order, overlapping the per-page stages so the
// recognizer (the bottleneck) runs concurrently with orientation+detection of
// the next page and assembly of the previous one. Results keep input order.
//
// ONNX Runtime sessions are not safe for concurrent Run, so each stage is a
// single goroutine that owns its stage's sessions (orient+det / rec / order):
// at most one page is inside a given stage at a time. Upright's own skew and
// orientation work is unaffected.
func (o *OCR) CallAll(gs []*Gray) []*Page {
	out := make([]*Page, len(gs))
	if len(gs) == 0 {
		return out
	}
	type detRes struct {
		idx   int
		g     *Gray
		boxes [][4]float32
		k     int
		skew  float64
	}
	type recRes struct {
		detRes
		rec []recLine
	}
	in := make(chan detRes, 2)
	mid := make(chan recRes, 2)
	var wg sync.WaitGroup
	wg.Add(2)
	go func() { // recognize: consumes detects, feeds assembly
		defer wg.Done()
		for d := range in {
			mid <- recRes{detRes: d, rec: o.Recognize(d.g, d.boxes)}
		}
		close(mid)
	}()
	go func() { // assemble: the tail of each page
		defer wg.Done()
		for r := range mid {
			out[r.idx] = o.finish(r.g, r.k, r.skew, r.boxes, r.rec)
		}
	}()
	for i, g := range gs { // orient + detect
		g2, k, skew := o.Upright(g)
		boxes := o.Detect(g2)
		in <- detRes{idx: i, g: g2, boxes: boxes, k: k, skew: skew}
	}
	close(in)
	wg.Wait()
	return out
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
		p.Markdown = PageMarkdown(fb, texts, order, clsM, sameM, o.Furniture, cells, cellBoxes)
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
