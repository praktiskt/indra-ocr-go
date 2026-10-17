// Model metadata parsing. Models (datalab-to/indra-ocr v3.1) are
// embedded via embed.go; nothing is downloaded at runtime.
package indra

import (
	"encoding/json"
	"fmt"
)

// meta holds det_long_side, rec_height and the vocabs from meta.json.
type meta struct {
	DetLongSide int
	RecHeight   int
	vocab       *vocab
	vocabCJK    *vocab // rec_cjk recognizer vocab (CJK lines routed by script.onnx)
}

// parseMeta parses meta.json content.
func parseMeta(data []byte) (*meta, error) {
	var d struct {
		DetLongSide int    `json:"det_long_side"`
		RecHeight   int    `json:"rec_height"`
		Vocab       string `json:"vocab"`
		VocabCJK    string `json:"vocab_cjk"`
	}
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, fmt.Errorf("indra: bad meta.json: %w", err)
	}
	v, err := vocabFromJSON(d.Vocab)
	if err != nil {
		return nil, err
	}
	m := &meta{DetLongSide: d.DetLongSide, RecHeight: d.RecHeight, vocab: v}
	if d.DetLongSide == 0 {
		m.DetLongSide = 1024
	}
	if m.RecHeight == 0 {
		m.RecHeight = 32
	}
	if d.VocabCJK != "" {
		vc, err := vocabFromJSON(d.VocabCJK)
		if err != nil {
			return nil, err
		}
		m.vocabCJK = vc
	}
	return m, nil
}
