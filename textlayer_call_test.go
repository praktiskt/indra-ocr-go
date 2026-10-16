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
		"a\tb\nc":      0,
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
	if lost != 0 || out[0] != "a" || out[1] != "b" {
		t.Fatalf("assignRuns = %v lost=%d, want [a b]", out, lost)
	}
	// whole-run placement (no per-char boxes)
	out2, _ := assignRuns([]TextRun{{Text: "hello", Box: [4]float64{11, 2, 19, 8}}}, boxes)
	if out2[0] != "" || out2[1] != "hello" {
		t.Fatalf("whole-run assignRuns = %v, want [ hello]", out2)
	}
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
	recs := ocr.Recognize(g2, boxes)
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
