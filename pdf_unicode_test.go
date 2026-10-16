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
		// word-initial accents: the accent glyph follows a space or starts the run
		"\u00b4unico":         "\u00fanico",
		"Zipf \u00b4e":        "Zipf \u00e9",
		"o \u00b4\u0131ndice": "o \u00edndice",
		"`a Dra.":             "\u00e0 Dra.",
		"associados `as":      "associados \u00e0s",
		// Portuguese tilde / circumflex (modifier-letter accents, inside a word)
		"disserta\u00e7\u02dcao": "disserta\u00e7\u00e3o",
		"orienta\u00e7\u02dcao":  "orienta\u00e7\u00e3o",
		"reuni\u02dcoes":         "reuni\u00f5es",
		"pot\u02c6encia":         "pot\u00eancia",
		"par\u02c6ametro":        "par\u00e2metro",
		// a prime attached to a number stays; an accent after an apostrophe composes
		"SplExtp\u00b4,":        "SplExtp\u00b4,",
		"5\u00b4s":              "5\u00b4s",
		"d\u2019\u00b4economie": "d\u2019\u00e9conomie",
		// math operators: not in the table (or not a composable base) -> untouched
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
