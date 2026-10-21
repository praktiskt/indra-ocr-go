package indra

// PDF input: render each page and use its text layer where sound (upstream
// document.py). The PDF engine (libpdfium) is embedded; nothing is installed.

import (
	"fmt"
	"math"
	"strings"
	"unicode"
	"unsafe"
)

// PDFOptions configure PDF conversion.
type PDFOptions struct {
	DPI      float64 // render dpi (default 200, matching upstream)
	Mode     string  // "auto" (default: text layer where sound, OCR elsewhere), "ocr", or "text"
	MaxPages int     // 0 = all pages
	// NoImageOCR turns off OCR of detected lines that hold no text-layer
	// characters (upstream image_ocr=False: drop them).
	NoImageOCR bool
}

// document.py constants.
const (
	checkDPI      = 72.0
	maxPagePixels = 100e6
)

// pageScale is document.Converter.page_scale: pixels per point for a page at the
// set dpi, lowered for pages so large that the render would pass maxPagePixels.
func pageScale(dpi, w, h float64) float64 {
	k := dpi / 72.0
	if s := math.Sqrt(maxPagePixels / math.Max(w*h, 1.0)); s < k {
		k = s
	}
	return k
}

// frameGray renders-frame gray turned by -frame quarter turns (np.rot90(gray,
// -frame)): the coordinates extract stores match this image.
func frameGray(raw *Gray, frame int) *Gray {
	if frame == 0 {
		return raw
	}
	return rot90CCW(raw, -frame)
}

// CallPDF converts PDF bytes to Pages. Each page is rendered; unless
// mode="ocr" its PDF text layer is read, and where the layer is used the line
// text comes from it (Source "text"/"hybrid"), elsewhere the page is OCR'd.
// mode:
//
//   - "auto" (default): the layer is used when the trust checks pass, or when
//     its script is one the recognizer cannot read (a layer it cannot read beats
//     OCR, unless the font map is broken) — upstream's `unreadable` nuance; a
//     trusted layer is checked against the page image (legacy fonts).
//   - "ocr": never read the layer.
//   - "text": use the layer even when the trust checks fail.
//
// Pages the engine cannot open are skipped.
func (o *OCR) CallPDF(data []byte, opts *PDFOptions) ([]*Page, error) {
	dpi, mode, maxPages := 200.0, "auto", 0
	imageOCR := true
	if opts != nil {
		if opts.DPI > 0 {
			dpi = opts.DPI
		}
		if opts.Mode != "" {
			mode = opts.Mode
		}
		maxPages = opts.MaxPages
		imageOCR = !opts.NoImageOCR
	}
	// PDFium is not thread-safe within one loaded copy, so each concurrent
	// CallPDF works on its own instance (see pdfiumPool); renditions on distinct
	// instances run in parallel.
	inst, err := pdfiumInsts.acquire()
	if err != nil {
		return nil, err
	}
	defer pdfiumInsts.release(inst)
	doc, buf, err := inst.open(data)
	if err != nil {
		return nil, err
	}
	defer inst.close(doc, buf)
	n := inst.pageCount(doc)
	if maxPages > 0 && n > maxPages {
		n = maxPages
	}
	out := make([]*Page, 0, n)
	for i := 0; i < n; i++ {
		if p := o.pdfPage(inst, doc, i, dpi, mode, imageOCR); p != nil {
			out = append(out, p)
		}
	}
	return out, nil
}

// pdfPage is document.Converter.pdf_page for one page of the open document.
func (o *OCR) pdfPage(inst *pdfiumInst, doc unsafe.Pointer, index int, dpi float64, mode string, imageOCR bool) *Page {
	page, ok := inst.loadPage(doc, index)
	if !ok {
		return nil
	}
	defer page.close()
	w, h := page.size()
	k := pageScale(dpi, w, h)
	note := "mode=ocr"
	turn := true
	if mode != "ocr" {
		checkImg, ok := page.render(k)
		if !ok {
			return nil
		}
		layer := page.extract(k, 0, 0)
		layer.Trusted, layer.Reason = o.layerVerdict(checkImg, layer)
		kk := pageTurn(layer.Stats)
		rot := page.rotation()
		if kk != 0 && (rot+90*kk)%360 == 0 {
			// The page's /Rotate alone turns the text: read the page without it.
			page.setRotation(0)
			if up, ok := page.render(k); ok {
				tl := page.extract(k, 0, 0)
				tl.Trusted, tl.Reason = o.layerVerdict(up, tl)
				if len(tl.Runs) > 0 && (tl.Trusted || mode == "text") {
					tl.Stats.Turn = kk
					if out := o.textPage2(up, tl, mode == "auto", k, imageOCR); out != nil {
						return out
					}
				}
			}
			page.setRotation(rot)
			layer = page.extract(k, kk, kk)
			layer.Trusted, layer.Reason = o.layerVerdict(frameGray(checkImg, kk), layer)
		} else if kk != 0 {
			layer = page.extract(k, kk, kk)
			layer.Trusted, layer.Reason = o.layerVerdict(frameGray(checkImg, kk), layer)
		} else if verticalLayout(layer.Stats) {
			layer = page.extract(k, 0, 3)
			layer.Trusted, layer.Reason = o.layerVerdict(frameGray(checkImg, 3), layer)
			layer.Stats.Vertical = true
		}
		// A layer the recognizer cannot read beats OCR, unless its codes are junk.
		unreadable := layer.Reason != brokenFontMap && !o.readableText(layer.Runs)
		note = layer.Reason
		if len(layer.Runs) > 0 && (layer.Trusted || mode == "text" || unreadable) {
			// A trusted layer in a script the recognizer reads is checked
			// against what the page shows (legacy fonts store other codes).
			checkLines := mode == "auto" && layer.Trusted && !unreadable
			if out := o.textPage2(checkImg, layer, checkLines, k, imageOCR); out != nil {
				return out
			}
			note = notThePage
		}
		// A layer with a broken font map still places its glyphs: when most run
		// upright, the page is upright (orient.onnx is not asked).
		st := layer.Stats
		turn = !(layer.Reason == brokenFontMap && float64(st.Rotated) < 0.5*float64(st.Chars))
	}
	gray, ok := page.render(k)
	if !ok {
		return nil
	}
	out := o.CallTurn(gray, turn)
	out.Note = joinNote(note, out.Note)
	return out
}

func joinNote(a, b string) string {
	switch {
	case a == "":
		return b
	case b == "":
		return a
	}
	return a + "; " + b
}

// textPage2 is document.Converter._text_page: the page from its text layer.
// check: first compare the script of up to agreeLines of the layer's lines with
// the script the line classifier sees in their images; when they do not match
// (layerAgrees), return nil so OCR reads the page. raw is the page render in its
// displayed frame at scale k.
func (o *OCR) textPage2(raw *Gray, layer *TextLayer, check bool, k float64, imageOCR bool) *Page {
	turn := layer.Stats.Turn
	frame := layer.Stats.Frame
	vertical := layer.Stats.Vertical
	gray := frameGray(raw, frame)
	boxes := o.Detect(gray)
	toks, lost := assignRuns(layer.Runs, f64boxes(boxes))
	rec := make([]recLine, len(boxes))
	var empty []int
	for i, tk := range toks {
		text := ""
		if len(tk) > 0 {
			text = o.decodeToks(tk, nil)
		}
		if text != "" {
			rec[i] = recLine{text: text, conf: 1.0, toks: tk}
		} else {
			empty = append(empty, i)
		}
	}
	if check {
		var full []int
		for i, r := range rec {
			if r.text != "" && countAlpha(r.text) >= 4 {
				full = append(full, i)
			}
		}
		pick := sampleEvery(full, agreeLines)
		if len(pick) > 0 {
			sub := make([][4]float32, len(pick))
			for q, i := range pick {
				sub[q] = boxes[i]
			}
			groups, sure := o.boxGroups(gray, sub)
			agree := make([]bool, len(pick))
			for q, i := range pick {
				agree[q] = !(latinShare(rec[i].text) >= latinCodes && groups[q] != "base" && sure[q])
			}
			if !layerAgrees(agree) {
				return nil
			}
		}
	}
	source := "text"
	// Sideways runs (a rotated table header, a chart's axis label) come from the
	// layer as their own lines; detected lines on them are not OCR'd.
	var side []TextRun
	for _, r := range layer.Rotated {
		if strings.TrimSpace(r.Text) != "" {
			side = append(side, r)
		}
	}
	var sideBoxes [][4]float64
	for _, r := range side {
		sideBoxes = append(sideBoxes, r.Box)
	}
	if len(side) > 0 && len(empty) > 0 {
		var ne []int
		for _, i := range empty {
			if !boxCovered(f64box(boxes[i]), sideBoxes, 0.5) {
				ne = append(ne, i)
			}
		}
		empty = ne
	}
	if len(empty) > 0 && imageOCR && o.readableText(layer.Runs) {
		sub := make([][4]float32, len(empty))
		for q, i := range empty {
			sub[q] = boxes[i]
		}
		for q, r := range o.recognize(gray, sub, vertical) {
			if strings.TrimSpace(r.text) != "" {
				rec[empty[q]] = r
				source = "hybrid"
			}
		}
	}
	o.readMathTextPage(gray, boxes, rec)

	var keep []int
	for i, r := range rec {
		if r.text != "" {
			keep = append(keep, i)
		}
	}
	kb := make([][4]float32, len(keep))
	kr := make([]recLine, len(keep))
	for q, i := range keep {
		kb[q], kr[q] = boxes[i], rec[i]
	}
	if len(side) > 0 {
		for q, r := range side {
			b := sideBoxes[q]
			kb = append(kb, [4]float32{float32(b[0]), float32(b[1]), float32(b[2]), float32(b[3])})
			cx := (b[0] + b[2]) / 2 // one x: a sideways line is never cut
			toks := make([]token, 0, len(r.Text))
			for _, c := range r.Text {
				toks = append(toks, token{Ch: string(c), X: cx, Seq: r.Seq, hasSeq: true})
			}
			kr = append(kr, recLine{text: r.Text, conf: 1.0, toks: toks})
		}
	}
	out := o.assemble([2]float64{float64(gray.W), float64(gray.H)}, kb, kr, vertical)
	out.Source = source
	out.Rotation = 90 * turn
	if vertical { // back on the page as it is: (x, y) -> (W - y, x)
		W := float64(gray.H)
		for i := range out.Lines {
			out.Lines[i].Box = verticalBack(out.Lines[i].Box, W)
		}
		for i := range out.Blocks {
			out.Blocks[i].BBox = verticalBack(out.Blocks[i].BBox, W)
		}
		out.Vertical = true
	}
	if lost > 0 {
		out.Note = fmt.Sprintf("%d chars outside detected lines", lost)
	}
	return out
}

// readMathTextPage is document.Converter._read_math: text-layer lines that hold
// math (the classifier, on their crops) are read by MathRec instead — the layer
// has the symbols but not the LaTeX.
func (o *OCR) readMathTextPage(gray *Gray, boxes [][4]float32, rec []recLine) {
	mr := o.math()
	if mr == nil {
		return
	}
	var idx []int
	for i, r := range rec {
		if len(r.toks) > 0 && r.toks[0].Ch != "" && r.toks[0].hasSeq {
			idx = append(idx, i)
		}
	}
	if len(idx) == 0 {
		return
	}
	sub := make([][4]float32, len(idx))
	for q, i := range idx {
		sub[q] = boxes[i]
	}
	flags := mr.flags(o.cropLines(gray, sub, o.m.RecHeight, 1, nil))
	var mathIdx []int
	var mathBoxes [][4]float32
	for q, f := range flags {
		if f {
			mathIdx = append(mathIdx, idx[q])
			mathBoxes = append(mathBoxes, boxes[idx[q]])
		}
	}
	if len(mathIdx) == 0 {
		return
	}
	reads := mr.read(gray, mathBoxes)
	for q, i := range mathIdx {
		text, conf := reads[q].text, reads[q].conf
		// A reading without math, or not confident, leaves the layer's text. A
		// garbled reading leaves it too: the decoder renders prose as single
		// letters and drops the diacritics, so the layer's own characters are
		// better than the decoder's re-reading.
		if !strings.Contains(text, "$") || conf < mathMinConf || mathGarbled(text) {
			continue
		}
		first := rec[i].toks[0].Seq
		for _, tk := range rec[i].toks {
			if tk.Seq < first {
				first = tk.Seq
			}
		}
		xc := float64(boxes[i][0]+boxes[i][2]) / 2
		rec[i] = recLine{text: text, conf: 1.0, toks: []token{{Ch: text, X: xc, Seq: first, hasSeq: true}}}
	}
}

func countAlpha(s string) int {
	n := 0
	for _, c := range s {
		if unicode.IsLetter(c) {
			n++
		}
	}
	return n
}

// sampleEvery is full[::max(1, len//n)][:n].
func sampleEvery(full []int, n int) []int {
	if len(full) == 0 {
		return nil
	}
	stride := max(1, len(full)/n)
	var out []int
	for j := 0; j < len(full) && len(out) < n; j += stride {
		out = append(out, full[j])
	}
	return out
}

// CallPDF converts PDF bytes over the pool.
func (p *Pool) CallPDF(data []byte, opts *PDFOptions) ([]*Page, error) {
	o := <-p.ch
	defer func() { p.ch <- o }()
	return o.CallPDF(data, opts)
}
