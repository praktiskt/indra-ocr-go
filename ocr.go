// OCR pipeline over sessions. Port of OrtOCR in pipeline.py (sans mask.onnx and
// math: Latin/multi + CJK routing).
package indra

import (
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
	s *sessions
	m *meta
	v *vocab
	// Furniture keeps page headers and footers in Markdown (upstream
	// load(furniture=True)). The default, false, drops them. Set before Call.
	Furniture bool
	// Math reads math lines as text with inline LaTeX (upstream math=True,
	// default off). The math models are embedded and load lazily on first use.
	Math       bool
	mathOnce   sync.Once
	mathReader *mathReader
	mathErr    error
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
	m, err := parseMeta(raw)
	if err != nil {
		return nil, err
	}
	s, err := newSessionsWithThreads(tc)
	if err != nil {
		return nil, err
	}
	return &OCR{s: s, m: m, v: m.vocab}, nil
}

// RecHeight is the recognizer crop height (the rec model's input height), the
// value CropLine expects. Exposed for tools that drive the stages directly.
func (o *OCR) RecHeight() int { return o.m.RecHeight }

// Close releases sessions, including lazily loaded math sessions.
func (o *OCR) Close() {
	o.mathReader.Destroy()
	o.s.close()
}

// math returns the lazily loaded math reader, or nil unless OCR.Math is set.
func (o *OCR) math() *mathReader {
	if !o.Math {
		return nil
	}
	o.mathOnce.Do(func() { o.mathReader, o.mathErr = newMathReader() })
	return o.mathReader
}

// token is (id, page-x) from a CTC peak.
type token struct {
	ID int
	X  float64
}

// recLine is (text, conf, tokens) per detected line.
type recLine struct {
	text string
	conf float64
	toks []token
}

func (o *OCR) Call(g *Gray) *Page {
	g, k, skew := o.Upright(g)
	boxes := o.Detect(g)
	rec := o.recognize(g, boxes)
	return o.finish(g, k, skew, boxes, rec)
}

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
			mid <- recRes{detRes: d, rec: o.recognize(d.g, d.boxes)}
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
