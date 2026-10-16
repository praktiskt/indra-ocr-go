package indra

// Math lines: mathcls.onnx flags lines that hold math, and a small
// encoder/decoder reads them as text with inline LaTeX (upstream math.py). The
// models are embedded; sessions load lazily, only when OCR.Math is on.

import (
	"encoding/json"
	"math"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	mathMin       = 0.5 // classifier probability above which a line is math
	mathMinConf   = 0.9 // a reading replaces a line above this mean token prob
	mathEarlyStop = 0.1 // stop after 8 tokens below mathMinConf-mathEarlyStop
	mathBOS       = 1
	mathEOS       = 2
	mathNSpecial  = 4
	mathEncStride = 8  // encoder downsample: memory length = ceil(width/8)
	mathClsHeight = 32 // mathcls reads the recognizer's crops (rec height)
)

type mathLine struct {
	text string
	conf float64
}

// MathReader reads math lines as LaTeX. Built lazily by OCR when OCR.Math is on;
// its sessions belong to the OCR handle and are freed by OCR.Close.
type MathReader struct {
	cls, enc, init, step   *ort.DynamicAdvancedSession
	height, channels       int
	maxTokens, nTokens     int
	layers, heads, headDim int
	itos                   []string
}

func newMathReader() (*MathReader, error) {
	raw, err := modelFS.ReadFile("models/math_meta.json")
	if err != nil {
		return nil, err
	}
	var mm struct {
		Config struct {
			Height    int `json:"height"`
			Channels  int `json:"channels"`
			DecLayers int `json:"dec_layers"`
			DecHeads  int `json:"dec_heads"`
			MaxTokens int `json:"max_tokens"`
		} `json:"config"`
		Tokens []string `json:"tokens"`
	}
	if err := json.Unmarshal(raw, &mm); err != nil {
		return nil, err
	}
	r := &MathReader{
		height: mm.Config.Height, channels: mm.Config.Channels,
		maxTokens: mm.Config.MaxTokens, layers: mm.Config.DecLayers,
		heads: mm.Config.DecHeads,
	}
	if r.heads > 0 {
		r.headDim = r.channels / r.heads
	}
	r.itos = append(make([]string, mathNSpecial), mm.Tokens...)
	r.nTokens = len(r.itos)
	mk := func(file string, in, out []string) (*ort.DynamicAdvancedSession, error) {
		o, err := newOptions(2)
		if err != nil {
			return nil, err
		}
		defer o.Destroy()
		data, err := modelFS.ReadFile("models/" + file)
		if err != nil {
			return nil, err
		}
		return ort.NewDynamicAdvancedSessionWithONNXData(data, in, out, o)
	}
	if r.cls, err = mk("mathcls.onnx", []string{"x", "widths"}, []string{"p"}); err != nil {
		return nil, err
	}
	if r.enc, err = mk("math_enc.onnx", []string{"x", "widths"}, []string{"memory", "mask"}); err != nil {
		return nil, err
	}
	if r.init, err = mk("math_dec_init.onnx", []string{"memory"}, []string{"cross_k", "cross_v"}); err != nil {
		return nil, err
	}
	if r.step, err = mk("math_dec_step.onnx",
		[]string{"token", "pos", "self_k", "self_v", "cross_k", "cross_v", "mask"},
		[]string{"logits", "new_k", "new_v"}); err != nil {
		return nil, err
	}
	return r, nil
}

// Destroy releases the math sessions.
func (m *MathReader) Destroy() {
	if m == nil {
		return
	}
	for _, s := range []*ort.DynamicAdvancedSession{m.cls, m.enc, m.init, m.step} {
		if s != nil {
			s.Destroy()
		}
	}
}

// detokenize is math.detokenize: tokens -> text, math spans joined by spaces.
func detokenize(tokens []string) string {
	var out, mth []string
	inMath := false
	for _, t := range tokens {
		if t == "$" {
			if inMath {
				out = append(out, "$"+strings.Join(mth, " ")+"$")
				mth = nil
			}
			inMath = !inMath
		} else if inMath {
			mth = append(mth, t)
		} else {
			out = append(out, t)
		}
	}
	if inMath && len(mth) > 0 {
		out = append(out, "$"+strings.Join(mth, " ")+"$")
	}
	return strings.Join(strings.Fields(strings.Join(out, "")), " ")
}

// flags runs mathcls over the recognizer's crops: true where the line is math.
func (m *MathReader) flags(crops []lineCrop) []bool {
	out := make([]bool, len(crops))
	if m == nil || len(crops) == 0 {
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
			hs[k] = mathClsHeight
			bw[k] = crops[j].w
		}
		batched, wOut, wm := PadBatch(lines, hs, bw)
		x, err := ort.NewTensor(ort.Shape{int64(len(part)), 1, mathClsHeight, int64(wm)}, batched)
		if err != nil {
			continue
		}
		w, err := ort.NewTensor(ort.Shape{int64(len(part))}, wOut)
		if err != nil {
			x.Destroy()
			continue
		}
		p, err := ort.NewEmptyTensor[float32](ort.Shape{int64(len(part))})
		if err == nil {
			if m.cls.Run([]ort.Value{x, w}, []ort.Value{p}) == nil {
				for k, j := range part {
					out[j] = p.GetData()[k] > mathMin
				}
			}
			p.Destroy()
		}
		x.Destroy()
		w.Destroy()
	}
	return out
}

// read crops each box at the math height and decodes LaTeX.
func (m *MathReader) read(g *Gray, boxes [][4]float32) []mathLine {
	n := len(boxes)
	out := make([]mathLine, n)
	if m == nil || n == 0 {
		return out
	}
	crops := make([]lineCrop, n)
	for i, b := range boxes {
		fb := [4]float64{float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}
		ink := CropLine(g, fb, m.height)
		crops[i] = lineCrop{ink, len(ink) / m.height}
	}
	ws := make([]int, n)
	for i := range crops {
		ws[i] = crops[i].w
	}
	for _, part := range WidthBatches(ws, 512) {
		m.decodeBatch(out, crops, part)
	}
	return out
}

func (m *MathReader) decodeBatch(out []mathLine, crops []lineCrop, part []int) {
	b := len(part)
	lines := make([][]float32, b)
	hs := make([]int, b)
	bw := make([]int, b)
	for k, j := range part {
		lines[k] = crops[j].ink
		hs[k] = m.height
		bw[k] = crops[j].w
	}
	batched, wOut, wm := PadBatch(lines, hs, bw)
	x, err := ort.NewTensor(ort.Shape{int64(b), 1, int64(m.height), int64(wm)}, batched)
	if err != nil {
		return
	}
	defer x.Destroy()
	w, err := ort.NewTensor(ort.Shape{int64(b)}, wOut)
	if err != nil {
		return
	}
	defer w.Destroy()
	t := int64((wm + mathEncStride - 1) / mathEncStride)
	kvShape := func(p int64) ort.Shape {
		return ort.Shape{int64(m.layers), int64(b), int64(m.heads), p, int64(m.headDim)}
	}
	mem, err := ort.NewEmptyTensor[float32](ort.Shape{int64(b), t, int64(m.channels)})
	if err != nil {
		return
	}
	defer mem.Destroy()
	mask, err := ort.NewEmptyTensor[float32](ort.Shape{int64(b), t})
	if err != nil {
		return
	}
	defer mask.Destroy()
	if err := m.enc.Run([]ort.Value{x, w}, []ort.Value{mem, mask}); err != nil {
		return
	}
	ck, err := ort.NewEmptyTensor[float32](kvShape(t))
	if err != nil {
		return
	}
	defer ck.Destroy()
	cv, err := ort.NewEmptyTensor[float32](kvShape(t))
	if err != nil {
		return
	}
	defer cv.Destroy()
	if err := m.init.Run([]ort.Value{mem}, []ort.Value{ck, cv}); err != nil {
		return
	}
	sk, err := ort.NewEmptyTensor[float32](kvShape(0))
	if err != nil {
		return
	}
	sv, err := ort.NewEmptyTensor[float32](kvShape(0))
	if err != nil {
		sk.Destroy()
		return
	}
	tok := make([]int64, b)
	for k := range tok {
		tok[k] = mathBOS
	}
	seqs := make([][]int, b)
	probs := make([][]float64, b)
	done := make([]bool, b)
	for step := 0; step < m.maxTokens-1; step++ {
		tokT, err := ort.NewTensor(ort.Shape{int64(b)}, tok)
		if err != nil {
			break
		}
		posT, err := ort.NewScalar[int64](int64(step))
		if err != nil {
			tokT.Destroy()
			break
		}
		nk, err := ort.NewEmptyTensor[float32](kvShape(int64(step + 1)))
		if err != nil {
			tokT.Destroy()
			posT.Destroy()
			break
		}
		nv, err := ort.NewEmptyTensor[float32](kvShape(int64(step + 1)))
		if err != nil {
			tokT.Destroy()
			posT.Destroy()
			nk.Destroy()
			break
		}
		logits, err := ort.NewEmptyTensor[float32](ort.Shape{int64(b), int64(m.nTokens)})
		if err != nil {
			tokT.Destroy()
			posT.Destroy()
			nk.Destroy()
			nv.Destroy()
			break
		}
		err = m.step.Run([]ort.Value{tokT, posT, sk, sv, ck, cv, mask},
			[]ort.Value{logits, nk, nv})
		tokT.Destroy()
		posT.Destroy()
		if err != nil {
			logits.Destroy()
			nk.Destroy()
			nv.Destroy()
			break
		}
		lg := logits.GetData()
		allDone := true
		for k := 0; k < b; k++ {
			base := k * m.nTokens
			amax, best := lg[base], 0
			for c := 1; c < m.nTokens; c++ {
				if lg[base+c] > amax {
					amax, best = lg[base+c], c
				}
			}
			var sum float64
			for c := 0; c < m.nTokens; c++ {
				sum += math.Exp(float64(lg[base+c] - amax))
			}
			tok[k] = int64(best)
			if !done[k] {
				probs[k] = append(probs[k], 1.0/sum)
				switch {
				case best == mathEOS:
					done[k] = true
				case len(probs[k]) >= 8 && mean(probs[k]) < mathMinConf-mathEarlyStop:
					done[k] = true
				case best >= mathNSpecial:
					seqs[k] = append(seqs[k], best)
				}
			}
			if !done[k] {
				allDone = false
			}
		}
		logits.Destroy()
		sk.Destroy()
		sv.Destroy()
		sk, sv = nk, nv
		if allDone {
			break
		}
	}
	sk.Destroy()
	sv.Destroy()
	for k, j := range part {
		toks := make([]string, len(seqs[k]))
		for i, id := range seqs[k] {
			if id >= 0 && id < len(m.itos) {
				toks[i] = m.itos[id]
			}
		}
		out[j] = mathLine{text: detokenize(toks), conf: mean(probs[k])}
	}
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}
