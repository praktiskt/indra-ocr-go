// Text-layer page path (CallText): trust-gated layer text + hybrid OCR.
package indra

// CallText OCRs a page image but uses the caller's PDF text layer where it is
// sound (upstream mode="auto"): trusted runs supply each detected line's text,
// and lines the layer does not cover are OCR'd (hybrid). Falls back to plain
// OCR when the layer is not trusted (scan, invisible layer, broken font map,
// runs with no ink). Ordering uses the order model; upstream's pdf_order
// (content order) was ported and rejected (neutral headline, multi-column
// regression — see docs/OPTIMIZATIONS_LOG.md). Source is "text" or "hybrid".
//
// The caller's run boxes must be in the same pixels as g; CallPDF drives the
// embedded engine through the same textPage2 path with a real scale.
func (o *OCR) CallText(g *Gray, tl *TextLayer) *Page {
	trusted, reason := o.readTextLayer(g, tl)
	tl.Trusted, tl.Reason = trusted, reason
	if !trusted {
		return o.Call(g)
	}
	return o.textPage2(g, tl, true, 1, true)
}
