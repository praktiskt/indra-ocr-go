// Token vocabulary for the line recognizer. Port of vocab.py (inference
// subset): class 0 is the CTC blank, class i+1 is Chars[i].
package indra

import (
	"encoding/json"
	"fmt"
	"strings"
	"unicode"
)

// Vocab holds the recognizer tokens (characters, or LaTeX commands).
type Vocab struct {
	Chars []string
	Index map[string]int // token -> class id (blank excluded)
	cmds  bool           // any multi-char token
}

// FromJSON parses {"blank":0,"chars":[...]} as stored in meta.json.
func VocabFromJSON(s string) (*Vocab, error) {
	var d struct {
		Blank int      `json:"blank"`
		Chars []string `json:"chars"`
	}
	if err := json.Unmarshal([]byte(s), &d); err != nil {
		return nil, fmt.Errorf("indra: bad vocab json: %w", err)
	}
	if d.Blank != 0 {
		return nil, fmt.Errorf("indra: only blank=0 vocabularies are supported")
	}
	return NewVocab(d.Chars), nil
}

// NewVocab builds the index over chars.
func NewVocab(chars []string) *Vocab {
	v := &Vocab{Chars: append([]string(nil), chars...), Index: make(map[string]int, len(chars))}
	for i, c := range v.Chars {
		v.Index[c] = i + 1
		if len([]rune(c)) > 1 {
			v.cmds = true
		}
	}
	return v
}

// NClasses counts blank + tokens.
func (v *Vocab) NClasses() int { return len(v.Chars) + 1 }

// SpaceID returns the class id of " ", or -1.
func (v *Vocab) SpaceID() int {
	if id, ok := v.Index[" "]; ok {
		return id
	}
	return -1
}

// Decode maps class ids back to text. Mirrors Vocab.decode: a space is put
// back between a LaTeX command and a following letter so "\alpha"+"x" does
// not read back as "\alphax".
func (v *Vocab) Decode(ids []int) string {
	var sb strings.Builder
	var toks []string
	for _, id := range ids {
		if id > 0 && id <= len(v.Chars) {
			toks = append(toks, v.Chars[id-1])
		}
	}
	if !v.cmds {
		for _, t := range toks {
			sb.WriteString(t)
		}
		return sb.String()
	}
	for k, t := range toks {
		sb.WriteString(t)
		var nxt string
		if k+1 < len(toks) {
			nxt = toks[k+1]
		}
		if len(t) > 2 && t[0] == '\\' && isASCIILetters(t[1:]) && len(nxt) > 0 && isASCIILetter(rune(nxt[0])) {
			sb.WriteByte(' ')
		}
	}
	return sb.String()
}

func isASCIILetters(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if !isASCIILetter(r) {
			return false
		}
	}
	return true
}

func isASCIILetter(r rune) bool {
	return (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z')
}

// IsSpace reports whether tok is a blank separator for gap splitting.
func (v *Vocab) IsSpace(tok any, spaceID int) bool {
	switch t := tok.(type) {
	case int:
		return t == 0 || t == spaceID
	case string:
		return t == " "
	}
	return false
}

// Indic helpers, ported from vocab.py for text-layer validation.

func indicBlock(r rune) bool {
	return (r >= 0x0900 && r <= 0x097F) || (r >= 0x0980 && r <= 0x09FF)
}

// IndicValid is false when a Devanagari/Bengali mark has no base to attach to.
func IndicValid(text string) bool {
	var prev rune
	var hasPrev bool
	for _, c := range text {
		if indicBlock(c) && (unicode.Is(unicode.Mn, c) || unicode.Is(unicode.Mc, c)) {
			o := int(c)
			pOK := hasPrev && indicBlock(prev)
			pCatIsLo := hasPrev && unicode.Is(unicode.Lo, prev)
			switch {
			case o == 0x0900 || o == 0x0901 || o == 0x0902 || o == 0x0903 ||
				o == 0x0981 || o == 0x0982 || o == 0x0983:
				if !pOK {
					return false
				}
			case o == 0x094D || o == 0x09CD || o == 0x093C || o == 0x09BC:
				if !(pOK && (pCatIsLo || int(prev) == 0x093C || int(prev) == 0x09BC)) {
					return false
				}
			default:
				if !(pOK && (pCatIsLo || int(prev) == 0x093C || int(prev) == 0x09BC)) {
					return false
				}
			}
		}
		prev, hasPrev = c, true
	}
	return true
}
