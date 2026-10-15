// Model metadata parsing. Models (datalab-to/indra-ocr v3.1) are
// embedded via embed.go; nothing is downloaded at runtime.
package indra

import (
	"encoding/json"
	"fmt"
)

// Meta holds det_long_side, rec_height and the vocabs from meta.json.
type Meta struct {
	DetLongSide int
	RecHeight   int
	Vocab       *Vocab
	VocabCJK    *Vocab // parsed for completeness; CJK routing not implemented
}

// ParseMeta parses meta.json content.
func ParseMeta(data []byte) (*Meta, error) {
	var d struct {
		DetLongSide int    `json:"det_long_side"`
		RecHeight   int    `json:"rec_height"`
		Vocab       string `json:"vocab"`
		VocabCJK    string `json:"vocab_cjk"`
	}
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, fmt.Errorf("indra: bad meta.json: %w", err)
	}
	v, err := VocabFromJSON(d.Vocab)
	if err != nil {
		return nil, err
	}
	m := &Meta{DetLongSide: d.DetLongSide, RecHeight: d.RecHeight, Vocab: v}
	if d.DetLongSide == 0 {
		m.DetLongSide = 1024
	}
	if m.RecHeight == 0 {
		m.RecHeight = 32
	}
	if d.VocabCJK != "" {
		vc, err := VocabFromJSON(d.VocabCJK)
		if err != nil {
			return nil, err
		}
		m.VocabCJK = vc
	}
	return m, nil
}
