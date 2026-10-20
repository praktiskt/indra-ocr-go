package indra

import (
	"image/png"
	"os"
	"testing"
)

func loadTestGray(t *testing.T, path string) *Gray {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open page: %v", err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatalf("decode png: %v", err)
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	px := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			px[y*w+x] = uint8((299*(r>>8) + 587*(g>>8) + 114*(bl>>8) + 500) / 1000)
		}
	}
	return &Gray{Pix: px, W: w, H: h}
}

// End-to-end smoke on a real document page.
// Requires ORT_LIB_PATH + INDRA_TEST_PAGE. Models are embedded.
func TestOCRIntegrationRealPage(t *testing.T) {
	page := os.Getenv("INDRA_TEST_PAGE")
	if page == "" {
		t.Skip("INDRA_TEST_PAGE not set")
	}
	ocr, err := NewOCR(2)
	if err != nil {
		t.Fatalf("NewOCR: %v", err)
	}
	defer ocr.Close()

	g := loadTestGray(t, page)
	boxes := ocr.Detect(g)
	t.Logf("detected %d boxes", len(boxes))
	if len(boxes) < 50 {
		t.Fatalf("want >=50 boxes on real page, got %d", len(boxes))
	}
	rec := ocr.recognize(g, boxes, false)
	empty := 0
	for _, r := range rec {
		if r.text == "" {
			empty++
		}
	}
	t.Logf("recognized %d/%d non-empty", len(rec)-empty, len(rec))
	p := ocr.Call(g)
	t.Logf("lines=%d rotation=%d skew=%.2f", len(p.Lines), p.Rotation, p.Skew)
	if len(p.Lines) > 0 {
		t.Logf("first line: %q (conf %.3f)", p.Lines[0].Text, p.Lines[0].Conf)
	}
	md := p.Markdown
	if len(md) > 600 {
		md = md[:600]
	}
	t.Logf("markdown head:\n%s", md)
	if len(p.Lines) < 20 {
		t.Fatalf("want >=20 lines, got %d", len(p.Lines))
	}
	// Spot-check: markdown should mention something table-ish or have headings.
	if p.Markdown == "" {
		t.Fatalf("empty markdown")
	}
}
