// Right-to-left text for a recognizer that reads every line left to right.
// Port of rtl.py: a small subset of the Unicode bidi algorithm (UAX #9), no
// embeddings or isolates, neutrals resolved by runs. It is its own inverse,
// so labels and decoding always agree.
package indra

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/bidi"
	"golang.org/x/text/unicode/norm"
)

var mirrorPairs = map[rune]rune{
	'(': ')', ')': '(', '[': ']', ']': '[',
	'{': '}', '}': '{', '<': '>', '>': '<',
	'«': '»', '»': '«', '‹': '›', '›': '‹',
}

// presentationRange: glyph variants (initial, medial, final, ligatures) that
// map to base letters (NFKC on those blocks only).
func presentationRange(r rune) bool {
	return (r >= 0xFB1D && r <= 0xFDFF) || (r >= 0xFE70 && r <= 0xFEFF)
}

const tatweel = 'ـ'

// isRTLChar reports the bidi class R or AL (right-to-left letters).
func isRTLChar(c rune) bool {
	p, _ := bidi.LookupRune(c)
	cl := p.Class()
	return cl == bidi.R || cl == bidi.AL
}

// isLTRChar: digits (European and Arabic-Indic) read left to right inside
// RTL text, as do Latin letters.
func isLTRChar(c rune) bool {
	p, _ := bidi.LookupRune(c)
	cl := p.Class()
	return cl == bidi.L || cl == bidi.EN || cl == bidi.AN
}

func hasRTL(text string) bool {
	for _, c := range text {
		if isRTLChar(c) {
			return true
		}
	}
	return false
}

// rtlBase reports whether the line reads right to left: more RTL letters than
// LTR letters. The counts do not depend on the order, so logical and visual
// text agree.
func rtlBase(text string) bool {
	r, lt := 0, 0
	for _, c := range text {
		if isRTLChar(c) {
			r++
		} else if isLTRChar(c) {
			lt++
		}
	}
	return r > lt
}

// rtlNormalize: presentation forms to base letters (NFKC on those blocks
// only), no tatweel, no combining mark without a base character.
func rtlNormalize(text string) string {
	var sb strings.Builder
	out := make([]rune, 0, len(text))
	for _, c := range text {
		if presentationRange(c) {
			out = append(out, []rune(norm.NFKC.String(string(c)))...)
			continue
		}
		if c != tatweel && (len(out) > 0 || !(unicode.Is(unicode.Mn, c) || unicode.Is(unicode.Me, c))) {
			out = append(out, c)
		}
	}
	sb.WriteString(string(out))
	return sb.String()
}

// rtlClusters groups each character with its following combining marks
// (Arabic harakat, Hebrew niqqud), so marks stay after their base.
func rtlClusters(text string) []string {
	out := []string{}
	for _, c := range text {
		if len(out) > 0 && (unicode.Is(unicode.Mn, c) || unicode.Is(unicode.Me, c)) {
			out[len(out)-1] += string(c)
			continue
		}
		out = append(out, string(c))
	}
	return out
}

// clusterKind: R (right-to-left), L (left-to-right) or N (neutral) from the
// cluster's base character.
func clusterKind(cl string) byte {
	c := []rune(cl)[0]
	if isRTLChar(c) {
		return 'R'
	}
	if isLTRChar(c) {
		return 'L'
	}
	return 'N'
}

// rtlRuns returns [start, end) spans from a `strong` cluster to the last
// `strong` cluster before a cluster of a kind in `stop`.
func rtlRuns(kinds []byte, strong byte, stop map[byte]bool) [][2]int {
	var spans [][2]int
	start, last := -1, -1
	for i := 0; i <= len(kinds); i++ {
		k := byte('#')
		if i < len(kinds) {
			k = kinds[i]
		}
		if k == strong {
			if start < 0 {
				start = i
			}
			last = i
		} else if stop[k] || k == '#' {
			if start >= 0 {
				spans = append(spans, [2]int{start, last + 1})
				start, last = -1, -1
			}
		}
	}
	return spans
}

// mirrorCluster mirrors a bracket at right-to-left level, as a renderer
// draws it; the combining marks stay.
func mirrorCluster(cl string) string {
	r := []rune(cl)
	if m, ok := mirrorPairs[r[0]]; ok {
		r[0] = m
	}
	return string(r)
}

// flip turns logical order into visual order, and visual order back into
// logical order. baseRTL: the paragraph direction, when known (a renderer
// lays out a line by its paragraph's direction); nil decides by rtlBase.
func flip(text string, mirror bool, baseRTL *bool) string {
	if !hasRTL(text) {
		return text
	}
	cls := rtlClusters(text)
	kinds := make([]byte, len(cls))
	for i, c := range cls {
		kinds[i] = clusterKind(c)
	}
	base := rtlBase(text)
	if baseRTL != nil {
		base = *baseRTL
	}
	if base {
		// Units: LTR runs kept whole, every other cluster alone; reverse the
		// units.
		units := []string{}
		spans := rtlRuns(kinds, 'L', map[byte]bool{'R': true})
		i := 0
		for _, sp := range append(spans, [2]int{len(cls), len(cls)}) {
			a, b := sp[0], sp[1]
			for j := i; j < a; j++ {
				if mirror {
					units = append(units, mirrorCluster(cls[j]))
				} else {
					units = append(units, cls[j])
				}
			}
			if a < b {
				units = append(units, strings.Join(cls[a:b], ""))
			}
			i = b
		}
		out := make([]string, len(units))
		for j, u := range units {
			out[len(units)-1-j] = u
		}
		return strings.Join(out, "")
	}
	// LTR base: reverse each RTL run in place.
	out := append([]string(nil), cls...)
	for _, sp := range rtlRuns(kinds, 'R', map[byte]bool{'L': true}) {
		a, b := sp[0], sp[1]
		seg := make([]string, b-a)
		for j := a; j < b; j++ {
			if mirror {
				seg[b-1-j] = mirrorCluster(cls[j])
			} else {
				seg[b-1-j] = cls[j]
			}
		}
		copy(out[a:b], seg)
	}
	return strings.Join(out, "")
}
