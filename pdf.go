package indra

// PDF input: render each page and use its text layer where sound (upstream
// document.py). The PDF engine (libpdfium) is embedded; nothing is installed.

// PDFOptions configure PDF conversion.
type PDFOptions struct {
	DPI      float64 // render dpi (default 200, matching upstream)
	Mode     string  // "auto" (default: text layer where sound, OCR elsewhere), "ocr", or "text"
	MaxPages int     // 0 = all pages
}

// CallPDF converts PDF bytes to Pages. Each page is rendered and, unless
// mode="ocr", its PDF text layer is read; where the layer is used the line text
// comes from it (Source "text"/"hybrid"), elsewhere the page is OCR'd. mode:
//
//   - "auto" (default): the layer is used when the trust checks pass, or when
//     its script is one the recognizer cannot read (a layer it cannot read beats
//     OCR, unless the font map is broken) — upstream's `unreadable` nuance.
//   - "ocr": never read the layer.
//   - "text": use the layer even when the trust checks fail (upstream mode="text").
//
// Pages the engine cannot open are skipped.
func (o *OCR) CallPDF(data []byte, opts *PDFOptions) ([]*Page, error) {
	dpi, mode, maxPages := 200.0, "auto", 0
	if opts != nil {
		if opts.DPI > 0 {
			dpi = opts.DPI
		}
		if opts.Mode != "" {
			mode = opts.Mode
		}
		maxPages = opts.MaxPages
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
		g, runs, chars, rot, invis, ok := inst.process(doc, i, dpi/72.0)
		if !ok {
			continue
		}
		if mode == "ocr" || len(runs) == 0 {
			p := o.Call(g)
			p.Source = "ocr"
			out = append(out, p)
			continue
		}
		layer := &TextLayer{Runs: runs, Invisible: invis > maxInvisibleFrac, Rotated: rot, Chars: chars}
		if o.useTextLayer(g, layer, mode) {
			out = append(out, o.textPage(g, layer))
			continue
		}
		p := o.Call(g)
		p.Source = "ocr"
		out = append(out, p)
	}
	return out, nil
}

// useTextLayer is document.pdf_page's text-layer decision: mode="text" always
// uses the layer; "auto" uses it when the trust checks pass, or when the
// recognizer cannot read the layer's script and its codes are not junk (broken
// font map). A layer the recognizer cannot read beats OCR; a broken map does not.
func (o *OCR) useTextLayer(g *Gray, layer *TextLayer, mode string) bool {
	if mode == "text" {
		return true
	}
	trusted, reason := o.readTextLayer(g, layer)
	return trusted || (reason != brokenFontMap && !o.readableText(layer.Runs))
}

// CallPDF converts PDF bytes over the pool.
func (p *Pool) CallPDF(data []byte, opts *PDFOptions) ([]*Page, error) {
	o := <-p.ch
	defer func() { p.ch <- o }()
	return o.CallPDF(data, opts)
}
