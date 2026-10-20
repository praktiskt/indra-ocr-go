// Line recognition: script routing (rec/rec_cjk/rec_rtl/rec_world), the
// n-gram LM beam search on unsure lines, math, CTC decode.
package indra

import (
	"math"
	"strings"
)

// RouteSure: a line goes to one recognizer when the classifier is surer than
// this; otherwise base reads it as well and the more confident reading wins.
const RouteSure = 0.95

// LM beam search constants (pipeline.py): lines under this confidence are
// re-read by the beam search with the language model; alpha weighs the LM,
// beta rewards each token. meta.json "lm_params" overrides them per group.
const (
	LMGate  = 0.9
	LMAlpha = 0.5
	LMBeta  = 1.5
	LMBeam  = 8
	// A class under this probability at a step is not tried by the beam search.
	LMCandFloor = 3e-3
)

// groupRec is one recognizer and its script group.
type groupRec struct {
	name    string
	run     func([]float32, int, int, int, []int64) (*recOut, error)
	vocab   *vocab
	asChars bool // the group's ids decode as chars via its vocab, then NFC/flip
}

// groupRecs returns the available recognizers in meta script_groups order
// (base first). A group without its recognizer or vocab is skipped.
func (o *OCR) groupRecs() []groupRec {
	var out []groupRec
	for _, g := range o.m.scriptGroups {
		if g == "base" {
			out = append(out, groupRec{"base", o.s.runRec, o.m.vocab, false})
			continue
		}
		v := o.m.groupVocab(g)
		if v == nil {
			continue
		}
		var run func([]float32, int, int, int, []int64) (*recOut, error)
		switch g {
		case "cjk":
			run = o.s.runRecCJK
		case "rtl":
			run = o.s.runRecRTL
		case "world":
			run = o.s.runRecWorld
		default:
			continue
		}
		out = append(out, groupRec{g, run, v, true})
	}
	return out
}

// lmFor returns the group's LM and its alpha/beta/gate (meta lm_params per
// group, else the base defaults); nil when the group has no LM.
func (o *OCR) lmForGroup(g string) (*ngramLM, float64, float64, float64) {
	lm := lmFor(g)
	if lm == nil {
		return nil, 0, 0, 0
	}
	p, ok := o.m.lmParams[g]
	if !ok {
		return lm, LMAlpha, LMBeta, LMGate
	}
	return lm, p.Alpha, p.Beta, p.Gate
}

// lineGroups runs script.onnx over the crops: per line, the group whose
// recognizer reads it, and whether the classifier is sure of it (probability
// over RouteSure). A group without its recognizer goes to base. Port of
// line_groups (the 4-class router; an older 1-output script.onnx still works).
func (o *OCR) lineGroups(crops []lineCrop) ([]string, []bool) {
	out := make([]string, len(crops))
	sure := make([]bool, len(crops))
	for i := range out {
		out[i] = "base"
		sure[i] = true
	}
	if len(crops) == 0 || o.s.script == nil || len(o.groupRecs()) <= 1 {
		return out, sure
	}
	ws := make([]int, len(crops))
	for i := range crops {
		ws[i] = crops[i].w
	}
	for _, part := range WidthBatches(ws, 512) {
		lines := make([][]float32, len(part))
		bw := make([]int, len(part))
		for k, j := range part {
			lines[k] = crops[j].ink
			bw[k] = crops[j].w
		}
		h := o.m.RecHeight
		batched, _, wm := PadBatch(lines, h, bw)
		p, nc, err := o.s.runScript(batched, len(part), h, wm, int64Widths(bw))
		if err != nil {
			continue
		}
		for k, j := range part {
			var g string
			var pk float64
			if nc <= 1 {
				// an older script.onnx: one P(Chinese) value per line
				if p[k] > 0.5 {
					g = "cjk"
				} else {
					g = "base"
				}
				pk = math.Max(float64(p[k]), 1-float64(p[k]))
			} else {
				best := 0
				for c := 1; c < nc; c++ {
					if p[k*nc+c] > p[k*nc+best] {
						best = c
					}
				}
				g = "base"
				if best < len(o.m.scriptGroups) {
					g = o.m.scriptGroups[best]
				}
				pk = float64(p[k*nc+best])
			}
			if o.m.groupVocab(g) == nil {
				g = "base"
			}
			out[j] = g
			sure[j] = pk > RouteSure
		}
	}
	return out, sure
}

// int64Widths lifts []int widths to []int64.
func int64Widths(ws []int) []int64 {
	out := make([]int64, len(ws))
	for i, w := range ws {
		out[i] = int64(w)
	}
	return out
}

// recognizeSel runs one recognizer session over crops[sel], writing out[sel].
// The base recognizer reads the tall window crops (when the model wants a
// window); group recognizers read the line band. A line whose greedy
// confidence lands under the LM gate is re-read by the CTC prefix beam
// search with the group's n-gram language model. Fine-grained width batches
// create many distinct (B,W) session shapes; ORT retains per-shape
// allocations (~the whole leak), so batch on a coarse 512px grid:
// bit-identical output, equal speed, less RSS.
func (o *OCR) recognizeSel(out []recLine, boxes [][4]float32, band, tall []lineCrop, sel []int,
	gr groupRec, lm *ngramLM, alpha, beta, gate float64, bestConf []float64) {
	if len(sel) == 0 {
		return
	}
	src := band
	h := o.m.RecHeight
	if gr.name == "base" && o.m.RecWindow > 1 && len(tall) == len(band) {
		src = tall
		h = o.m.RecHeight * o.m.RecWindow
	}
	ws := make([]int, len(sel))
	for k, j := range sel {
		ws[k] = src[j].w
	}
	vocab := gr.vocab
	for _, part := range WidthBatches(ws, 512) {
		lines := make([][]float32, len(part))
		bw := make([]int, len(part))
		for k, p := range part {
			j := sel[p]
			lines[k] = src[j].ink
			bw[k] = src[j].w
		}
		batched, _, wm := PadBatch(lines, h, bw)
		ro, err := gr.run(batched, len(part), h, wm, int64Widths(bw))
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
			bh := math.Max(1, float64(b[3]-b[1]))
			ix0 := math.Floor(float64(b[0]) - padX*bh)
			cw := math.Ceil(float64(b[2])+padX*bh) - ix0
			f := cw / float64(band[j].w) // x mapping stays on the band's width
			conf := 0.0
			if len(keptP) > 0 {
				for _, pp := range keptP {
					conf += pp
				}
				conf /= float64(len(keptP))
			}
			stepT, tokIDs := keptT, kept
			if lm != nil && MinConf <= conf && conf < gate {
				// CTC prefix beam search with the language model breaks close
				// calls; its steps replace the greedy peaks.
				ids2, steps := lmBeamSearch(
					ro.Blank[base:base+ln], ro.TopV[base*ro.K:base*ro.K+ln*ro.K],
					ro.TopI[base*ro.K:base*ro.K+ln*ro.K], ln, ro.K, lm, alpha, beta, LMBeam, LMCandFloor)
				tokIDs, stepT = ids2, steps
			}
			toks := make([]token, 0, len(tokIDs))
			for q, id := range tokIDs {
				x := ix0 + (float64(stepT[q])+0.5)*stride*f
				if gr.asChars {
					ch := ""
					if 0 < id && id <= len(vocab.Chars) {
						ch = vocab.Chars[id-1]
					}
					toks = append(toks, token{Ch: ch, X: x})
				} else {
					toks = append(toks, token{ID: id, X: x})
				}
			}
			if conf <= bestConf[j] {
				continue
			}
			bestConf[j] = conf
			if conf >= MinConf {
				if gr.asChars {
					out[j] = recLine{text: joinChars(vocab, ids), conf: conf, toks: toks}
				} else {
					out[j] = recLine{text: vocab.Decode(ids), conf: conf, toks: toks}
				}
			} else {
				out[j] = recLine{text: "", conf: conf}
			}
		}
	}
}

// bestConf returns the running confidence of a slot.
func bestConfOf(r recLine) float64 {
	return r.conf
}

// joinChars decodes group ids as characters (the group vocab is the label
// set); whitespace collapse and NFC/flip happen in the assembly decode.
func joinChars(v *vocab, ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if 0 < id && id <= len(v.Chars) {
			sb.WriteString(v.Chars[id-1])
		}
	}
	return sb.String()
}

// recognize runs the recognizers over line boxes, routing each line by the
// script classifier (upstream recognize), returns (text, conf, tokens).
func (o *OCR) recognize(g *Gray, boxes [][4]float32, vertical bool) []recLine {
	n := len(boxes)
	out := make([]recLine, n)
	if n == 0 {
		return out
	}
	// The reading with the higher confidence wins; base runs first, so it
	// wins ties (upstream best_conf starts at -1.0).
	bestConf := make([]float64, n)
	for j := range bestConf {
		bestConf[j] = -1.0
	}
	var dark []bool
	if o.m.RecWindow > 1 {
		// Light-on-dark is decided once per line box, for both crops.
		dark = lineDarkness(g, boxes)
	}
	crops := o.cropLines(g, boxes, o.m.RecHeight, 1, dark)
	tall := crops
	if o.m.RecWindow > 1 {
		tall = o.cropLines(g, boxes, o.m.RecHeight, o.m.RecWindow, dark)
	}
	groups := o.groupRecs()
	type route struct {
		gr  groupRec
		sel []int
	}
	var routes []route
	base := groups[0]
	switch {
	case vertical && o.m.vocabCJK != nil && len(groups) > 1:
		// A vertical page's lines are CJK columns, characters on their side;
		// the CJK recognizer reads those (and the Latin set across them).
		all := make([]int, n)
		for j := range all {
			all[j] = j
		}
		for _, gr := range groups {
			if gr.name == "cjk" {
				routes = []route{{gr, all}}
			}
		}
	case len(groups) > 1 && len(crops) > 0:
		names, sure := o.lineGroups(crops)
		// A line the classifier is not sure about is read by base as well;
		// the reading with the higher confidence wins (below).
		var baseSel []int
		for j, g := range names {
			if g == "base" || !sure[j] {
				baseSel = append(baseSel, j)
			}
		}
		routes = append(routes, route{base, baseSel})
		for _, gr := range groups {
			if gr.name == "base" {
				continue
			}
			var sel []int
			for j, g := range names {
				if g == gr.name {
					sel = append(sel, j)
				}
			}
			if len(sel) > 0 {
				routes = append(routes, route{gr, sel})
			}
		}
	default:
		all := make([]int, n)
		for j := range all {
			all[j] = j
		}
		routes = []route{{base, all}}
	}
	for _, r := range routes {
		lm, alpha, beta, gate := o.lmForGroup(r.gr.name)
		o.recognizeSel(out, boxes, crops, tall, r.sel, r.gr, lm, alpha, beta, gate, bestConf)
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
					out[j] = recLine{text: reads[k].text, conf: 1.0, toks: []token{{Ch: reads[k].text, X: xc}}}
				}
			}
		}
	}
	return out
}
