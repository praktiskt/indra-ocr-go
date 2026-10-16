package indra

import (
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
)

// TestCallPDFPages guards the PDFium buffer lifetime: CallPDF must return every
// page and none blank. FPDF_LoadMemDocument64 does not copy its input, so a Go
// heap buffer handed to it may be freed by the GC mid-render — which silently
// drops/blanked pages. Aggressive GC makes it deterministic here.
func TestCallPDFPages(t *testing.T) {
	if os.Getenv("INDRA_TEST_IMAGES") == "" {
		t.Skip("set INDRA_TEST_IMAGES=1")
	}
	if err := InitLib(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/multipage12.pdf")
	if err != nil {
		t.Fatal(err)
	}
	prev := debug.SetGCPercent(1)
	defer debug.SetGCPercent(prev)
	ocr, err := NewOCR(2)
	if err != nil {
		t.Fatal(err)
	}
	defer ocr.Close()

	runtime.GC()
	pages, err := ocr.CallPDF(data, &PDFOptions{MaxPages: 100})
	if err != nil {
		t.Fatal(err)
	}
	if len(pages) != 12 {
		t.Fatalf("CallPDF returned %d pages, want 12", len(pages))
	}
	empty := 0
	for _, p := range pages {
		if strings.TrimSpace(p.Markdown) == "" {
			empty++
		}
	}
	if empty != 0 {
		t.Errorf("%d of 12 pages blank, want 0", empty)
	}
}
