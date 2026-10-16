package indra

import (
	"strings"
	"testing"
)

func TestFoldSpacingAccents(t *testing.T) {
	fold := func(s string) string {
		seg := make([]TextChar, 0, len(s))
		for _, r := range s {
			seg = append(seg, TextChar{Ch: r})
		}
		var b strings.Builder
		for _, c := range foldSpacingAccents(seg) {
			b.WriteRune(c.Ch)
		}
		return b.String()
	}
	cases := map[string]string{
		// acute
		"Cicho\u00b4n":           "Cicho\u0144",
		"Fr\u00b4echet":          "Fr\u00e9chet",
		"St\u00b4ephane":         "St\u00e9phane",
		"investigaci\u00b4on":    "investigaci\u00f3n",
		"Mej\u00b4\u0131a":       "Mej\u00eda", // dotless i before an acute
		"Ingenier\u00b4\u0131as": "Ingenier\u00edas",
		// diaeresis / double acute / cedilla
		"the Gr\u00a8obner": "the Gr\u00f6bner",
		"Erd\u02ddos":       "Erd\u0151s",
		"Fran\u00b8cois":    "Fran\u00e7ois",
		// primes (next char is not a letter): kept
		"SplExtp\u00b4,":   "SplExtp\u00b4,",
		"r\u00b4, \u00b4s": "r\u00b4, \u00b4s",
		// math operators between letters: never touched
		"g\u00afi":           "g\u00afi",
		"Ham\u005ec":         "Ham\u005ec",
		"U\u02dcpBq":         "U\u02dcpBq",
		"l,k, \u03bd\u02c6l": "l,k, \u03bd\u02c6l",
		"plain ascii":        "plain ascii",
	}
	for in, want := range cases {
		if got := fold(in); got != want {
			t.Errorf("foldSpacingAccents(%q) = %q, want %q", in, got, want)
		}
	}
}
