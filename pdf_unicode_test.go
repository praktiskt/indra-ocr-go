package indra

import (
	"strings"
	"testing"
)

func foldStr(s string) string {
	seg := make([]TextChar, 0, len(s))
	for _, r := range s {
		seg = append(seg, TextChar{Ch: r})
	}
	return runes(foldSpacingAccents(seg))
}

func runes(cs []TextChar) string {
	var b strings.Builder
	for _, c := range cs {
		b.WriteRune(c.Ch)
	}
	return b.String()
}

// TestFoldSpacingAccents covers the prefix orientation (the text layer draws the
// accent before its base) with no boxes.
func TestFoldSpacingAccents(t *testing.T) {
	cases := map[string]string{
		// acute
		"Cicho\u00b4n":           "Cicho\u0144",
		"Fr\u00b4echet":          "Fr\u00e9chet",
		"St\u00b4ephane":         "St\u00e9phane",
		"investigaci\u00b4on":    "investigaci\u00f3n",
		"Mej\u00b4\u0131a":       "Mej\u00eda", // dotless i before an acute
		"Ingenier\u00b4\u0131as": "Ingenier\u00edas",
		"SplExtp\u00b4m":         "SplExtp\u1e3f", // acute m (base previously missing)
		// diaeresis / double acute / cedilla / caron
		"the Gr\u00a8obner":  "the Gr\u00f6bner",
		"Erd\u02ddos":        "Erd\u0151s",
		"Fran\u00b8cois":     "Fran\u00e7ois",
		"Sch\u00a8utzen":     "Sch\u00fctzen",
		"\u02c7c":            "\u010d", // caron c
		"\u02c7s":            "\u0161",
		"\u02c7z":            "\u017e",
		"\u02c7r":            "\u0159",
		"\u02c7e":            "\u011b",
		"Dvo\u02c7r\u00b4ak": "Dvo\u0159\u00e1k",
		"\u00b8g":            "\u0123", // cedilla g (Latvian ģ)
		"\u00b8n":            "\u0146", // cedilla n (Latvian ņ)
		// macron / breve / dot above / ring above / ogonek
		"\u00afa": "\u0101", // macron a (Latvian ā)
		"\u00afu": "\u016b",
		"\u00afe": "\u0113",
		"\u02d8g": "\u011f", // breve g (Turkish ğ)
		"\u02d8a": "\u0103", // breve a (Romanian ă)
		"\u02d8u": "\u016d", // breve u (Esperanto ŭ)
		"\u02d9e": "\u0117", // dot above e (Lithuanian ė)
		"\u02d9z": "\u017c", // dot above z (Polish ż)
		"\u02daa": "\u00e5", // ring a (Nordic å)
		"\u02dba": "\u0105", // ogonek a (Polish ą)
		"\u02dbe": "\u0119", // ogonek e (Polish ę)
		// Esperanto: circumflex bases that used to be missing
		"\u02c6c": "\u0109",
		"\u02c6g": "\u011d",
		"\u02c6s": "\u015d",
		// word-initial accents
		"\u00b4unico":         "\u00fanico",
		"Zipf \u00b4e":        "Zipf \u00e9",
		"o \u00b4\u0131ndice": "o \u00edndice",
		"`a Dra.":             "\u00e0 Dra.",
		"associados `as":      "associados \u00e0s",
		// Portuguese tilde / circumflex
		"disserta\u00e7\u02dcao": "disserta\u00e7\u00e3o",
		"pot\u02c6encia":         "pot\u00eancia",
		"par\u02c6ametro":        "par\u00e2metro",
		"reuni\u02dcoes":         "reuni\u00f5es",
		// a prime attached to a number stays; apostrophe-preceded accent composes
		"5\u00b4s":              "5\u00b4s",
		"d\u2019\u00b4economie": "d\u2019\u00e9conomie",
		// a base a mark cannot join is left alone; ASCII ^ ~ are not marks
		"g\u02dcpBq":         "g\u02dcpBq",
		"Ham\u005ec":         "Ham\u005ec",
		"U\u007epBq":         "U\u007epBq",
		"l,k, \u03bd\u02c6l": "l,k, \u03bd\u02c6l",
		"plain ascii":        "plain ascii",
	}
	for in, want := range cases {
		if got := foldStr(in); got != want {
			t.Errorf("foldSpacingAccents(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestFoldSpacingAccentsOrder checks the prefix/postfix decision from glyph
// boxes: the mark is joined to the neighbour whose box it overlaps.
func TestFoldSpacingAccentsOrder(t *testing.T) {
	box := func(s string, markOver float64) []TextChar {
		// one slot per rune, slot width 10; the mark's box is put over the
		// character at the given x offset.
		seg := make([]TextChar, 0, len(s))
		for i, r := range s {
			x := float64(i * 10)
			seg = append(seg, TextChar{Ch: r, Box: [4]float64{x, 0, x + 10, 10}})
		}
		for i := range seg {
			if seg[i].Ch == '\u00a8' || seg[i].Ch == '\u00b4' || seg[i].Ch == '\u02c6' {
				seg[i].Box = [4]float64{markOver, 0, markOver + 10, 10}
			}
		}
		return seg
	}
	// postfix: diaeresis over the preceding u -> Schütz
	if got := runes(foldSpacingAccents(box("Schu\u00a8tz", 30))); got != "Sch\u00fctz" {
		t.Errorf("postfix diaeresis = %q, want %q", got, "Sch\u00fctz")
	}
	// prefix: diaeresis over the following t -> t+diaeresis = ẗ
	if got := runes(foldSpacingAccents(box("Schu\u00a8tz", 40))); got != "Schu\u1e97z" {
		t.Errorf("prefix diaeresis on t = %q, want %q", got, "Schu\u1e97z")
	}
	// ambiguous neighbours: postfix acute over y -> ýz, prefix over z -> yź
	if got := runes(foldSpacingAccents(box("y\u00b4z", 0))); got != "\u00fdz" {
		t.Errorf("postfix acute = %q, want %q", got, "\u00fdz")
	}
	if got := runes(foldSpacingAccents(box("y\u00b4z", 10))); got != "y\u017a" {
		t.Errorf("prefix acute = %q, want %q", got, "y\u017a")
	}
	// stacked (Vietnamese): base then two postfix marks -> e + circumflex + acute = ế
	st := []TextChar{
		{Ch: 'e', Box: [4]float64{0, 0, 10, 10}},
		{Ch: '\u02c6', Box: [4]float64{0, 0, 10, 10}},
		{Ch: '\u00b4', Box: [4]float64{0, 0, 10, 10}},
	}
	if got := runes(foldSpacingAccents(st)); got != "\u1ebf" {
		t.Errorf("stacked marks = %q, want %q", got, "\u1ebf")
	}
	// real postfix: Geocieˆncias (mark over the e before n) -> Geociências
	if got := runes(foldSpacingAccents(box("Geocie\u02c6ncias", 50))); got != "Geoci\u00eancias" {
		t.Errorf("postfix circumflex = %q, want %q", got, "Geoci\u00eancias")
	}
}
