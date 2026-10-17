// Text-layer page path (CallText): trust-gated layer text + hybrid OCR.
package indra

import (
	"strings"
)

// CallText OCRs a page image but uses the caller's PDF text layer where it is
// sound (upstream mode="auto"): trusted runs supply each detected line's text,
// and lines the layer does not cover are OCR'd (hybrid). Falls back to plain
// OCR when the layer is not trusted (scan, invisible layer, broken font map,
// runs with no ink). Ordering uses the order model; upstream's pdf_order
// (content order) was ported and rejected (neutral headline, multi-column
// regression — see docs/OPTIMIZATIONS_LOG.md). Source is "text" or "hybrid".
func (o *OCR) CallText(g *Gray, tl *TextLayer) *Page {
	if trusted, _ := o.readTextLayer(g, tl); !trusted {
		return o.Call(g)
	}
	return o.textPage(g, tl)
}

// textPage runs the text-layer page path on a layer the caller has decided to
// use (trusted, or forced by mode="text" / an unreadable script): detect on the
// page, take each line's text from the layer, OCR the lines it does not cover
// (hybrid) when the layer's script is readable, and assemble. Source is "text"
// or "hybrid" (upstream _text_page).
func (o *OCR) textPage(g *Gray, tl *TextLayer) *Page {
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
		for j, r := range o.recognize(g2, empty) {
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
