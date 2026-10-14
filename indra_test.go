package indra

import (
	"math"
	"testing"
)

func TestVocabDecodeBasic(t *testing.T) {
	v := NewVocab([]string{" ", "H", "i", "\\alpha", "x"})
	if got := v.Decode([]int{2, 3}); got != "Hi" {
		t.Fatalf("decode = %q", got)
	}
	// LaTeX command + letter gets a space back.
	if got := v.Decode([]int{4, 5}); got != "\\alpha x" {
		t.Fatalf("latex decode = %q", got)
	}
}

func TestDecodeDetSingleBlob(t *testing.T) {
	h, w := 8, 16
	out := make([]float32, 3*h*w)
	// core logit 10 in a 2x6 block at rows 3-4, cols 4-9.
	for y := 3; y < 5; y++ {
		for x := 4; x < 10; x++ {
			out[y*w+x] = 10
			out[h*w+y*w+x] = 2   // top dist
			out[2*h*w+y*w+x] = 3 // bottom dist
		}
	}
	boxes := DecodeDet(out, h, w)
	if len(boxes) != 1 {
		t.Fatalf("want 1 box, got %d", len(boxes))
	}
	b := boxes[0]
	// centres y=7,9 -> y0~5, y1~11..12
	if math.Abs(float64(b.Y0)-5) > 1 || math.Abs(float64(b.Y1)-11.5) > 1.5 {
		t.Fatalf("bad y: %+v", b)
	}
	if b.X0 >= b.X1 || b.Score < 0.99 {
		t.Fatalf("bad x/score: %+v", b)
	}
}

func TestDecodeDetEmpty(t *testing.T) {
	if boxes := DecodeDet(make([]float32, 3*4*4), 4, 4); len(boxes) != 0 {
		t.Fatalf("want 0 boxes, got %d", len(boxes))
	}
}

func TestLineFeaturesShape(t *testing.T) {
	boxes := [][4]float64{{0, 0, 100, 10}, {0, 20, 100, 30}}
	f := LineFeatures(boxes, 200, 100, []string{"Hello.", "item 1"})
	if len(f) != 2*NFeats {
		t.Fatalf("want %d feats, got %d", 2*NFeats, len(f))
	}
}

func TestDecodeOrderTrivial(t *testing.T) {
	a := [][]float64{{0, 5}, {-5, 0}} // 0 before 1
	if o := DecodeOrder(a, 4); len(o) != 2 || o[0] != 0 || o[1] != 1 {
		t.Fatalf("order = %v", o)
	}
}

func TestOrderXYCutColumns(t *testing.T) {
	boxes := [][4]float64{
		{0, 0, 40, 10}, {0, 20, 40, 30}, // left column
		{60, 0, 100, 10}, {60, 20, 100, 30}, // right column
	}
	o := OrderXYCut(boxes)
	// left column first (x gap dominates or row order keeps columns together)
	if len(o) != 4 || o[0] != 0 || o[1] != 1 {
		t.Fatalf("xycut = %v", o)
	}
}

func TestJoinLines(t *testing.T) {
	if got := JoinLines([]string{"tumori-", "genesis"}); got != "tumorigenesis" {
		t.Fatalf("hyphen join = %q", got)
	}
	if got := JoinLines([]string{"hello", "world"}); got != "hello world" {
		t.Fatalf("join = %q", got)
	}
}

func TestTableHTML(t *testing.T) {
	boxes := [][4]float64{
		{0, 0, 50, 10},
		{60, 0, 110, 10},
		{0, 20, 50, 30},
		{60, 20, 110, 30},
	}
	md := TableHTML(boxes, []string{"a", "b", "c", "d"})
	for _, want := range []string{"<table>", "<th>a</th>", "<td>c</td>", "</table>"} {
		if !contains(md, want) {
			t.Fatalf("table missing %q in:\n%s", want, md)
		}
	}
}

func TestPageToInputMultiple(t *testing.T) {
	g := &Gray{Pix: make([]uint8, 100*50), W: 100, H: 50}
	for i := range g.Pix {
		g.Pix[i] = 200
	}
	ink, pw, ph, scale := PageToInput(g, 1024)
	if pw%32 != 0 || ph%32 != 0 {
		t.Fatalf("not multiple of 32: %dx%d", pw, ph)
	}
	if math.Abs(scale-1024.0/100) > 1e-9 {
		t.Fatalf("scale = %v", scale)
	}
	if len(ink) != pw*ph {
		t.Fatalf("ink len = %d", len(ink))
	}
}

func TestCropLineHeight(t *testing.T) {
	g := &Gray{Pix: make([]uint8, 200*200), W: 200, H: 200}
	for i := range g.Pix {
		g.Pix[i] = 255
	}
	ink := CropLine(g, [4]float64{50, 50, 150, 70}, 32)
	if len(ink)%32 != 0 || len(ink)/32 < 8 {
		t.Fatalf("bad crop len %d", len(ink))
	}
	// blank page -> ink ~0
	for _, v := range ink {
		if v != 0 {
			t.Fatalf("blank crop ink = %v", v)
		}
	}
}

func contains(s, sub string) bool {
	if len(sub) > len(s) {
		return false
	}
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
