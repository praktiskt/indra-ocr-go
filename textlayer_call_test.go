package indra

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBadCharCount(t *testing.T) {
	cases := map[string]int{
		"hello":        0,
		"a\tb\nc":      1, // newline is Cc (pdfium consumes it before this check)
		"\uFFFD":       1, // replacement
		"\uE000\uE001": 2, // private use
		"a\x01b":       1, // control
		"café":         0,
	}
	for in, want := range cases {
		if got := badCharCount(in); got != want {
			t.Errorf("badCharCount(%q) = %d, want %d", in, got, want)
		}
	}
}

func TestNoInkFrac(t *testing.T) {
	g := &Gray{Pix: make([]uint8, 100*100), W: 100, H: 100}
	for i := range g.Pix {
		g.Pix[i] = 255
	}
	// ink under box (10,10)-(30,30)
	for y := 10; y < 30; y++ {
		for x := 10; x < 30; x++ {
			g.Pix[y*100+x] = 0
		}
	}
	runs := []TextRun{
		{Box: [4]float64{10, 10, 30, 30}}, // ink
		{Box: [4]float64{50, 50, 70, 70}}, // none
	}
	if f := noInkFrac(g, runs); f != 0.5 {
		t.Errorf("noInkFrac = %v, want 0.5", f)
	}
}

func TestAssignRuns(t *testing.T) {
	// two boxes side by side; a run whose chars fall in each
	boxes := [][4]float64{{0, 0, 10, 10}, {10, 0, 20, 10}}
	run := TextRun{
		Text: "ab",
		Box:  [4]float64{2, 2, 12, 8},
		Chars: []TextChar{
			{Ch: 'a', Box: [4]float64{2, 2, 6, 8}},
			{Ch: 'b', Box: [4]float64{11, 2, 15, 8}},
		},
	}
	out, lost := assignRuns([]TextRun{run}, boxes)
	if lost != 0 || tokText(out[0]) != "a" || tokText(out[1]) != "b" {
		t.Fatalf("assignRuns = %v lost=%d, want [a b]", out, lost)
	}
	// whole-run placement (no per-char boxes)
	out2, _ := assignRuns([]TextRun{{Text: "hello", Box: [4]float64{11, 2, 19, 8}}}, boxes)
	if tokText(out2[0]) != "" || tokText(out2[1]) != "hello" {
		t.Fatalf("whole-run assignRuns = %v, want [ hello]", out2)
	}
}

func tokText(ts []token) string {
	var sb strings.Builder
	for _, tk := range ts {
		sb.WriteString(tk.Ch)
	}
	return sb.String()
}

// TestCallTextWithLayer feeds CallText a text layer built from the page's own
// OCR result, proving the trusted text path reproduces the line text.
func TestCallTextWithLayer(t *testing.T) {
	if os.Getenv("INDRA_TEST_IMAGES") == "" {
		t.Skip("set INDRA_TEST_IMAGES=1")
	}
	if err := InitLib(); err != nil {
		t.Fatal(err)
	}
	files, _ := filepath.Glob("testdata/*.png")
	if len(files) == 0 {
		t.Skip("no testdata")
	}
	g := loadTestGray(t, files[0])
	ocr, err := NewOCR(2)
	if err != nil {
		t.Fatal(err)
	}
	defer ocr.Close()

	// Text layer == the recognizer's own output per detected box (the natural
	// shape of a PDF layer: one run per line). CallText must reproduce OCR.
	g2, _, _ := ocr.Upright(g)
	boxes := ocr.Detect(g2)
	recs := ocr.recognize(g2, boxes, false)
	var runs []TextRun
	for i, b := range boxes {
		if strings.TrimSpace(recs[i].text) != "" {
			runs = append(runs, TextRun{Text: recs[i].text, Box: [4]float64{
				float64(b[0]), float64(b[1]), float64(b[2]), float64(b[3])}})
		}
	}
	p := ocr.CallText(g, &TextLayer{Runs: runs})
	if p.Source != "text" {
		t.Fatalf("Source = %q, want text", p.Source)
	}
	base := ocr.Call(g)

	// Word overlap between the text-layer page and plain OCR.
	want := map[string]bool{}
	for _, w := range strings.Fields(strings.ToLower(base.Text())) {
		want[w] = true
	}
	got := 0
	words := strings.Fields(strings.ToLower(p.Text()))
	for _, w := range words {
		if want[w] {
			got++
		}
	}
	if len(words) == 0 || float64(got)/float64(len(words)) < 0.9 {
		t.Fatalf("word overlap %d/%d too low", got, len(words))
	}
}

// inkGray is a white page with ink under box (so the text-layer no-ink check
// passes).
func inkGray(box [4]float64) *Gray {
	g := &Gray{Pix: make([]uint8, 100*100), W: 100, H: 100}
	for i := range g.Pix {
		g.Pix[i] = 255
	}
	for y := int(box[1]); y < int(box[3]); y++ {
		for x := int(box[0]); x < int(box[2]); x++ {
			g.Pix[y*100+x] = 0
		}
	}
	return g
}

// useLayer replicates document.pdf_page's text-layer decision: mode="text"
// forces the layer; "auto" trusts it, or rescues it when the recognizer cannot
// read the script (but never a broken font map).
func useLayer(o *OCR, g *Gray, tl *TextLayer, mode string) bool {
	if mode == "text" {
		return true
	}
	trusted, reason := o.readTextLayer(g, tl)
	return trusted || (reason != brokenFontMap && !o.readableText(tl.Runs))
}

// TestUseTextLayer covers document.pdf_page's decision: mode="text" forces the
// layer; "auto" trusts it, or rescues it when the recognizer cannot read the
// script (but never a broken font map).
func TestUseTextLayer(t *testing.T) {
	ocr := &OCR{m: &meta{vocab: newVocab(strings.Split("abcdefghijklmnopqrstuvwxyz", ""))}}
	box := [4]float64{10, 10, 40, 20}
	g := inkGray(box)
	latin := []TextRun{{Text: "hello", Box: box}}
	arabic := []TextRun{{Text: "مرحبا", Box: box}}

	// mode="text" uses the layer even when the trust checks fail
	if !useLayer(ocr, g, &TextLayer{Runs: latin, Chars: 5, Invisible: true}, "text") {
		t.Errorf(`mode="text" with an untrusted layer: want use`)
	}
	// auto: a trusted layer is used
	if !useLayer(ocr, g, &TextLayer{Runs: latin, Chars: 5}, "auto") {
		t.Errorf("auto with a trusted layer: want use")
	}
	// auto: untrusted but the script is unreadable -> use (beats OCR)
	if !useLayer(ocr, g, &TextLayer{Runs: arabic, Chars: 5, Invisible: true}, "auto") {
		t.Errorf("auto with an unreadable untrusted layer: want use")
	}
	// auto: untrusted and the script is readable -> OCR
	if useLayer(ocr, g, &TextLayer{Runs: latin, Chars: 5, Invisible: true}, "auto") {
		t.Errorf("auto with a readable untrusted layer: want OCR")
	}
	// auto: a broken font map is never rescued by the unreadable rule
	broken := &TextLayer{Runs: []TextRun{{Text: "\uFFFDمرحبا", Box: box}}, Chars: 6}
	if useLayer(ocr, g, broken, "auto") {
		t.Errorf("auto with a broken font map: want OCR")
	}
}
