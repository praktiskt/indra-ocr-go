// Line recognition: script routing (rec/rec_cjk), math, CTC decode.
package indra

import (
	"math"
	"strings"
)

// hanziLines runs script.onnx over the crops: true where the line holds Chinese
// characters (upstream hanzi_lines). All false without script.onnx.
func (o *OCR) hanziLines(crops []lineCrop) []bool {
	out := make([]bool, len(crops))
	if len(crops) == 0 || o.s.script == nil || o.m.vocabCJK == nil {
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
			hs[k] = o.m.RecHeight
			bw[k] = crops[j].w
		}
		batched, wOut, wm := PadBatch(lines, hs, bw)
		p, err := o.s.runScript(batched, len(part), o.m.RecHeight, wm, wOut)
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
	run func([]float32, int, int, int, []int64) (*recOut, error), vocab *vocab) {
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
			hs[k] = o.m.RecHeight
			bw[k] = crops[j].w
		}
		batched, wOut, wm := PadBatch(lines, hs, bw)
		ro, err := run(batched, len(part), o.m.RecHeight, wm, wOut)
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
			ix0 := math.Floor(float64(b[0]) - padX*h)
			cw := math.Ceil(float64(b[2])+padX*h) - ix0
			f := cw / float64(crops[j].w)
			toks := make([]token, 0, len(kept))
			for q, id := range kept {
				x := ix0 + (float64(keptT[q])+0.5)*stride*f
				toks = append(toks, token{ID: id, X: x})
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

// recognize runs the recognizer over line boxes, routing lines the script
// classifier (script.onnx) calls Chinese to rec_cjk.onnx (upstream recognize),
// returns (text, conf, tokens).
func (o *OCR) recognize(g *Gray, boxes [][4]float32) []recLine {
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
	o.recognizeSel(out, boxes, crops, latin, o.s.runRec, o.v)
	if len(cjk) > 0 {
		o.recognizeSel(out, boxes, crops, cjk, o.s.runRecCJK, o.m.vocabCJK)
	}
	// Math lines: replace the recognizer's reading with LaTeX when confident.
	if mr := o.math(); mr != nil {
		flagged := mr.flags(crops)
		var sel []int
		for j, f := range flagged {
			if f {
				sel = append(sel, j)
			}
		}
		if len(sel) > 0 {
			sub := make([][4]float32, len(sel))
			for k, j := range sel {
				sub[k] = boxes[j]
			}
			reads := mr.read(g, sub)
			for k, j := range sel {
				if reads[k].conf >= mathMinConf && strings.Contains(reads[k].text, "$") {
					xc := float64(boxes[j][0]+boxes[j][2]) / 2
					out[j] = recLine{text: reads[k].text, conf: 1.0, toks: []token{{ID: -1, X: xc}}}
				}
			}
		}
	}
	return out
}
