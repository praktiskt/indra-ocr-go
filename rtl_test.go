package indra

import (
	"strings"
	"testing"
)

// flip turns logical order into visual order and back: it is its own inverse.
func TestFlipInvolution(t *testing.T) {
	cases := []string{
		"مرحبا بالعالم",
		"שלום עולם",
		"abc مرحبا def 123",
		"(مرحبا) [עולם]",
		"مرحبا 123 עולם",
		"العربية",
	}
	for _, s := range cases {
		vis := flip(s, true, nil)
		back := flip(vis, true, nil)
		// mirroring makes an odd number of [()] applications differ only in
		// brackets; compare with mirrored brackets normalized away.
		if back != s {
			t.Errorf("flip not involutive for %q: %q -> %q -> %q", s, s, vis, back)
		}
	}
}

func TestHasRTLAndBase(t *testing.T) {
	if !hasRTL("مرحبا") || !hasRTL("שלום") {
		t.Error("Arabic/Hebrew should be RTL")
	}
	if hasRTL("hello world 123") {
		t.Error("Latin should not be RTL")
	}
	if !rtlBase("مرحبا بالعالم") {
		t.Error("Arabic page should have an RTL base")
	}
	if rtlBase("hello world") {
		t.Error("Latin page should have an LTR base")
	}
	if !isRTLChar('م') || !isRTLChar('א') || isRTLChar('a') {
		t.Error("isRTLChar classes wrong")
	}
}

func TestRTLNormalizeTatweel(t *testing.T) {
	// Tatweel (kashida) is dropped.
	if got := rtlNormalize("مـرحـبا"); strings.ContainsRune(got, tatweel) {
		t.Errorf("tatweel not removed: %q", got)
	}
	// A combining mark with no base is dropped.
	if got := rtlNormalize("\u064Bمرحبا"); strings.HasPrefix(got, "\u064B") {
		t.Errorf("leading combining mark not dropped: %q", got)
	}
}

// lmFor returns the embedded LMs by group: base/cjk/world exist, rtl does not;
// their orders match the shipped files.
func TestLMForGroups(t *testing.T) {
	want := map[string]int{"base": 5, "cjk": 3, "world": 4}
	for g, order := range want {
		lm := lmFor(g)
		if lm == nil {
			t.Fatalf("lmFor(%q) = nil", g)
		}
		if lm.order != order {
			t.Errorf("lmFor(%q).order = %d, want %d", g, lm.order, order)
		}
		if lm.vocabSize <= 0 {
			t.Errorf("lmFor(%q).vocabSize = %d", g, lm.vocabSize)
		}
	}
	if lmFor("rtl") != nil {
		t.Error("rtl has no LM upstream; want nil")
	}
}

// logp is finite and bounded by maxLogp.
func TestLMLogpBounded(t *testing.T) {
	lm := lmFor("base")
	if lm == nil {
		t.Skip("no base lm")
	}
	mx := lm.maxLogp()
	for n := 1; n <= lm.order; n++ {
		hist := make([]int, 0, n)
		for i := 0; i < n; i++ {
			hist = append(hist, 1+i)
		}
		v := lm.logp(hist, 2)
		if v > mx+1e-9 {
			t.Errorf("logp(%v,2) = %g > maxLogp %g", hist, v, mx)
		}
	}
}

func TestLmLSE(t *testing.T) {
	// logsumexp in log space: ties add ln 2.
	got := lmLSE(0, 0)
	if got < 0.6931 || got > 0.6932 {
		t.Errorf("lmLSE(0,0) = %g, want ln2", got)
	}
}
