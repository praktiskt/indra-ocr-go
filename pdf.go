package indra

// PDF input: render each page and use its text layer where sound (upstream
// document.py). The PDF engine (libpdfium) is embedded; nothing is installed.

// PDFOptions configure PDF conversion.
type PDFOptions struct {
	DPI      float64 // render dpi (default 200, matching upstream)
	Mode     string  // "auto" (default: text layer where sound, OCR elsewhere) or "ocr"
	MaxPages int     // 0 = all pages
}

// CallPDF converts PDF bytes to Pages. Each page is rendered and, in "auto"
// mode, its PDF text layer is read; where the layer is sound the line text comes
// from it (Source "text"/"hybrid"), elsewhere the page is OCR'd. "ocr" mode
// ignores the text layer. Pages the engine cannot open are skipped.
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
	if err := loadPDFium(); err != nil {
		return nil, err
	}
	doc, buf, err := pdfOpen(data)
	if err != nil {
		return nil, err
	}
	defer pdfClose(doc, buf)
	n := pdfPageCount(doc)
	if maxPages > 0 && n > maxPages {
		n = maxPages
	}
	out := make([]*Page, 0, n)
	for i := 0; i < n; i++ {
		g, runs, chars, rot, invis, ok := pdfProcess(doc, i, dpi/72.0)
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
		out = append(out, o.CallText(g, layer))
	}
	return out, nil
}

// CallPDF converts PDF bytes over the pool.
func (p *Pool) CallPDF(data []byte, opts *PDFOptions) ([]*Page, error) {
	o := <-p.ch
	defer func() { p.ch <- o }()
	return o.CallPDF(data, opts)
}
