package indra

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Ports of tests/test_textlayer.py and layout_post tests from
// github.com/datalab-to/indra (Apache-2.0). No models needed.

func TestJoinLinesCJK(t *testing.T) {
	cases := []struct {
		in   []string
		want string
	}{
		{[]string{"或其中的", "任何部分，在 2020 年"}, "或其中的任何部分，在 2020 年"},
		{[]string{"説明します。", "次に"}, "説明します。次に"},
		{[]string{"한국어 문장", "다음 줄"}, "한국어 문장 다음 줄"}, // Korean keeps the space
		{[]string{"tumori-", "genesis is"}, "tumorigenesis is"},
		{[]string{"first line", "second"}, "first line second"},
	}
	for _, c := range cases {
		if got := joinLines(c.in); got != c.want {
			t.Errorf("joinLines(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestIsListItem(t *testing.T) {
	for _, s := range []string{"•", "• text", "12. text", "- item", "* a"} {
		if !isListItem(s) {
			t.Errorf("isListItem(%q) = false, want true", s)
		}
	}
	for _, s := range []string{"•text", "12.5 kg", "plain"} {
		if isListItem(s) {
			t.Errorf("isListItem(%q) = true, want false", s)
		}
	}
}

func TestOrderXYCutTwoColumnsReadingOrder(t *testing.T) {
	// two columns: left x 50-290, right x 310-550, y increasing.
	// reading order: entire left column before the right column.
	var boxes [][4]float64
	for i := 0; i < 20; i++ {
		boxes = append(boxes,
			[4]float64{50, float64(100 + 15*i), 290, float64(110 + 15*i)},
			[4]float64{310, float64(100 + 15*i), 550, float64(110 + 15*i)},
		)
	}
	o := orderXYCut(boxes)
	if len(o) != 40 {
		t.Fatalf("want 40, got %d", len(o))
	}
	for k, i := range o {
		col := i % 2 // 0 = left, 1 = right
		wantCol := 0
		if k >= 20 {
			wantCol = 1
		}
		if col != wantCol {
			t.Fatalf("xycut puts right-column line at pos %d (idx %d)", k, i)
		}
	}
}

func TestOrderTopDownBottomUpNotCoherent(t *testing.T) {
	// lines written bottom-up: input order reversed is the reading order.
	var boxes [][4]float64
	for i := 0; i < 20; i++ {
		y := 400 - 15*i
		boxes = append(boxes, [4]float64{50, float64(y), 500, float64(y + 10)})
	}
	o := orderTopDown(boxes)
	if len(o) != 20 {
		t.Fatalf("want 20, got %d", len(o))
	}
	// reading order must start with the lowest (last) input line.
	if o[0] != 19 {
		t.Fatalf("bottom-up page: first read line = %d, want 19", o[0])
	}
}

func TestEstimateSkewRecoversTilt(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	g := newPage(2200, 1700)
	for y := 150; y < 2050; y += 45 {
		x := 150
		for x < 1550 {
			w := 20 + rng.Intn(70)
			fillRect(g, x, y, w, 18, 30)
			x += w + 10 + rng.Intn(15)
		}
	}
	if got := estimateSkew(g); math.Abs(got) > 0.15 {
		t.Fatalf("untilted skew = %v, want |<0.15|", got)
	}
	for _, tilt := range []float64{2.0, -1.3, 0.6} {
		est := estimateSkew(rotatePage(g, tilt))
		if math.Abs(est+tilt) > 0.15 {
			t.Errorf("tilt %v: estimate %v, want |est+tilt|<=0.15", tilt, est)
		}
	}
	if got := estimateSkew(newPage(800, 600)); got != 0 {
		t.Errorf("blank page skew = %v, want 0", got)
	}
}

func TestUnrotateBoxMapsDeskewedBoxBack(t *testing.T) {
	g := newPage(400, 600)
	fillRect(g, 100, 180, 400, 40, 0) // long dark bar
	tilted := rotatePage(g, -3.0)
	var xs, ys []int
	for y := 0; y < tilted.H; y++ {
		for x := 0; x < tilted.W; x++ {
			if int(tilted.At(x, y)) < 128 {
				xs = append(xs, x)
				ys = append(ys, y)
			}
		}
	}
	if len(xs) == 0 {
		t.Fatal("no dark pixels after rotation")
	}
	mnX, mxX, mnY, mxY := xs[0], xs[0], ys[0], ys[0]
	for _, v := range xs {
		mnX, mxX = min(mnX, v), max(mxX, v)
	}
	for _, v := range ys {
		mnY, mxY = min(mnY, v), max(mxY, v)
	}
	got := unrotateBox([4]float64{100, 180, 500, 220}, 3.0, 600, 400)
	if got[0] > float64(mnX)+2 || got[2] < float64(mxX)-2 {
		t.Errorf("x coverage: got %v, dark range [%d,%d]", got, mnX, mxX)
	}
	if got[1] > float64(mnY)+2 || got[3] < float64(mxY)-2 {
		t.Errorf("y coverage: got %v, dark range [%d,%d]", got, mnY, mxY)
	}
	if id := unrotateBox([4]float64{10, 10, 20, 20}, 0.0, 600, 400); id != [4]float64{10, 10, 20, 20} {
		t.Errorf("zero-degree unrotate = %v", id)
	}
}

func newPage(w, h int) *Gray {
	g := &Gray{Pix: make([]uint8, w*h), W: w, H: h}
	for i := range g.Pix {
		g.Pix[i] = 255
	}
	return g
}

func fillRect(g *Gray, x, y, w, h int, v uint8) {
	for j := y; j < y+h && j < g.H; j++ {
		for i := x; i < x+w && i < g.W; i++ {
			g.Pix[j*g.W+i] = v
		}
	}
}

// Golden smoke tests on the upstream example pages
// (assets/examples/*.png, Apache-2.0). Needs models + lib:
// run with INDRA_TEST_IMAGES=1.
func TestExamplePages(t *testing.T) {
	if os.Getenv("INDRA_TEST_IMAGES") == "" {
		t.Skip("INDRA_TEST_IMAGES not set")
	}
	if err := InitLib(); err != nil {
		t.Fatalf("InitLib: %v", err)
	}
	ocr, err := NewOCR(2)
	if err != nil {
		t.Fatalf("NewOCR: %v", err)
	}
	defer ocr.Close()
	for _, name := range []string{"rotated_letter", "table_pmc", "multi_column", "chinese_essay"} {
		t.Run(name, func(t *testing.T) {
			g := loadTestGray(t, filepath.Join("testdata", name+".png"))
			p := ocr.Call(g)
			if len(p.Lines) == 0 || strings.TrimSpace(p.Markdown) == "" {
				t.Fatalf("%s: empty output (lines=%d)", name, len(p.Lines))
			}
			t.Logf("%s: lines=%d rotation=%d skew=%.2f", name, len(p.Lines), p.Rotation, p.Skew)
		})
	}
}

// Adaptive second-pass detection must recover small text pasted on a
// large canvas (first pass alone finds <50 of 138 lines). Guards the
// small-print path in Detect. Needs models + lib: INDRA_TEST_IMAGES=1.
func TestAdaptiveSmallText(t *testing.T) {
	if os.Getenv("INDRA_TEST_IMAGES") == "" {
		t.Skip("INDRA_TEST_IMAGES not set")
	}
	if err := InitLib(); err != nil {
		t.Fatalf("InitLib: %v", err)
	}
	ocr, err := NewOCR(2)
	if err != nil {
		t.Fatalf("NewOCR: %v", err)
	}
	defer ocr.Close()
	g := loadTestGray(t, filepath.Join("testdata", "multi_column.png"))
	big := &Gray{Pix: make([]uint8, 2700*3579), W: 2700, H: 3579}
	for i := range big.Pix {
		big.Pix[i] = 255
	}
	for y := 0; y < g.H; y++ {
		copy(big.Pix[(y+1100)*2700+900:(y+1100)*2700+900+g.W], g.Pix[y*g.W:(y+1)*g.W])
	}
	p := ocr.Call(big)
	t.Logf("small-text: lines=%d", len(p.Lines))
	if len(p.Lines) < 100 {
		t.Fatalf("adaptive pass lost small text: lines=%d, want >=100", len(p.Lines))
	}
}
