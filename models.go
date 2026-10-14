// Model metadata parsing. Models are embedded via embed.go.
package indra

import (
	"encoding/json"
	"fmt"
)

// HFRepo and HFRevision pin the weights this port was built against.
const (
	HFRepo     = "datalab-to/indra-ocr"
	HFRevision = "v3.1"
)

// Meta holds det_long_side, rec_height and the vocabs from meta.json.
type Meta struct {
	DetLongSide int
	RecHeight   int
	Vocab       *Vocab
	VocabCJK    *Vocab // nil when the folder has no CJK models
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
