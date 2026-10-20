// Model metadata parsing. Models (datalab-to/indra-ocr v1.1.0) are
// embedded via embed.go; nothing is downloaded at runtime.
package indra

import (
	"encoding/json"
	"fmt"
)

// lmParams are the language-model beam-search weights for one recognizer
// group (meta.json "lm_params"); the base recognizer uses the defaults.
type lmParams struct {
	Alpha float64
	Beta  float64
	Gate  float64
}

// meta holds det_long_side, rec_height, the vocabs and the script groups from
// meta.json.
type meta struct {
	DetLongSide int
	RecHeight   int
	RecWindow   int // rec.onnx reads a tall window of this many bands (1 = line only)
	vocab       *vocab
	vocabCJK    *vocab // rec_cjk recognizer vocab
	vocabRTL    *vocab // rec_rtl recognizer vocab
	vocabWorld  *vocab // rec_world recognizer vocab
	// scriptGroups names script.onnx's classes in order (base first).
	scriptGroups []string
	// lmParams per group name ("base", "cjk", ...); missing keys use the
	// LMAlpha/LMBeta/LMGate defaults.
	lmParams map[string]lmParams
}

// parseMeta parses meta.json content.
func parseMeta(data []byte) (*meta, error) {
	var d struct {
		DetLongSide  int               `json:"det_long_side"`
		RecHeight    int               `json:"rec_height"`
		RecWindow    int               `json:"rec_window"`
		Vocab        string            `json:"vocab"`
		VocabCJK     string            `json:"vocab_cjk"`
		VocabRTL     string            `json:"vocab_rtl"`
		VocabWorld   string            `json:"vocab_world"`
		ScriptGroups []string          `json:"script_groups"`
		LMParams     map[string]lmJSON `json:"lm_params"`
	}
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, fmt.Errorf("indra: bad meta.json: %w", err)
	}
	v, err := vocabFromJSON(d.Vocab)
	if err != nil {
		return nil, err
	}
	m := &meta{
		DetLongSide: d.DetLongSide, RecHeight: d.RecHeight,
		RecWindow: d.RecWindow, vocab: v,
		lmParams: map[string]lmParams{},
	}
	if d.DetLongSide == 0 {
		m.DetLongSide = 1024
	}
	if m.RecHeight == 0 {
		m.RecHeight = 32
	}
	if m.RecWindow == 0 {
		m.RecWindow = 1
	}
	for k, s := range map[string]string{"cjk": d.VocabCJK, "rtl": d.VocabRTL, "world": d.VocabWorld} {
		if s == "" {
			continue
		}
		vg, err := vocabFromJSON(s)
		if err != nil {
			return nil, err
		}
		switch k {
		case "cjk":
			m.vocabCJK = vg
		case "rtl":
			m.vocabRTL = vg
		case "world":
			m.vocabWorld = vg
		}
	}
	if len(d.ScriptGroups) > 0 {
		m.scriptGroups = d.ScriptGroups
	} else {
		m.scriptGroups = []string{"base", "cjk", "rtl", "world"}
	}
	for g, p := range d.LMParams {
		m.lmParams[g] = lmParams{Alpha: p.Alpha, Beta: p.Beta, Gate: p.Gate}
	}
	return m, nil
}

type lmJSON struct {
	Alpha float64 `json:"alpha"`
	Beta  float64 `json:"beta"`
	Gate  float64 `json:"gate"`
}

// groupVocab returns the recognizer vocab of a script group ("base" = main).
func (m *meta) groupVocab(g string) *vocab {
	switch g {
	case "base", "":
		return m.vocab
	case "cjk":
		return m.vocabCJK
	case "rtl":
		return m.vocabRTL
	case "world":
		return m.vocabWorld
	}
	return nil
}
