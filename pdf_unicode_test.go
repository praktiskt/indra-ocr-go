package indra

import (
	"strings"
	"testing"
)

func TestFoldSpacingAcute(t *testing.T) {
	fold := func(s string) string {
		seg := make([]TextChar, 0, len(s))
		for _, r := range s {
			seg = append(seg, TextChar{Ch: r})
		}
		var b strings.Builder
		for _, c := range foldSpacingAcute(seg) {
			b.WriteRune(c.Ch)
		}
		return b.String()
	}
	cases := map[string]string{
		"Cicho\u00b4n":           "Cicho\u0144",
		"Fr\u00b4echet":          "Fr\u00e9chet",
		"St\u00b4ephane":         "St\u00e9phane",
		"investigaci\u00b4on":    "investigaci\u00f3n",
		"Mej\u00b4\u0131a":       "Mej\u00eda", // dotless i before an acute
		"Ingenier\u00b4\u0131as": "Ingenier\u00edas",
		"SplExtp\u00b4,":         "SplExtp\u00b4,", // a prime on p, not an accent: kept
		"r\u00b4, \u00b4s":       "r\u00b4, \u00b4s",
		"plain ascii":            "plain ascii",
	}
	for in, want := range cases {
		if got := fold(in); got != want {
			t.Errorf("foldSpacingAcute(%q) = %q, want %q", in, got, want)
		}
	}
}
