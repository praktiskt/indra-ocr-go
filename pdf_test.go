package indra

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
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

// TestPDFiumConcurrentRender guards PDFium thread safety. PDFium's C API is not
// thread-safe — only one call may be in flight per loaded copy — and the copies
// share no state, so a concurrent caller must land on its own instance
// (pdfiumPool). Rendering the same PDF from several goroutines must reproduce
// the sequential result byte for byte; sharing one instance corrupts PDFium's
// globals (crashes, or pages that differ).
func TestPDFiumConcurrentRender(t *testing.T) {
	if os.Getenv("INDRA_TEST_IMAGES") == "" {
		t.Skip("set INDRA_TEST_IMAGES=1")
	}
	data, err := os.ReadFile("testdata/multipage12.pdf")
	if err != nil {
		t.Fatal(err)
	}
	render := func() ([32]byte, error) {
		inst, err := pdfiumInsts.acquire()
		if err != nil {
			return [32]byte{}, err
		}
		defer pdfiumInsts.release(inst)
		doc, buf, err := inst.open(data)
		if err != nil {
			return [32]byte{}, err
		}
		defer inst.close(doc, buf)
		h := sha256.New()
		put := func(v any) { _ = binary.Write(h, binary.LittleEndian, v) }
		n := inst.pageCount(doc)
		put(int64(n))
		for i := 0; i < n; i++ {
			page, ok := inst.loadPage(doc, i)
			put(ok)
			if !ok {
				continue
			}
			g, ok := page.render(200.0 / 72.0)
			put(ok)
			if !ok {
				page.close()
				continue
			}
			put(int64(g.W))
			put(int64(g.H))
			h.Write(g.Pix)
			layer := page.extract(200.0/72.0, 0, 0)
			page.close()
			put(int64(layer.Chars))
			put(int64(layer.Stats.Rotated))
			put(layer.Invisible)
			put(int64(len(layer.Runs)))
			for _, r := range layer.Runs {
				h.Write([]byte(r.Text))
				for _, v := range r.Box {
					put(v)
				}
				put(int64(len(r.Chars)))
			}
		}
		var sum [32]byte
		copy(sum[:], h.Sum(nil))
		return sum, nil
	}

	want, err := render()
	if err != nil {
		t.Fatal(err)
	}
	const goroutines, rounds = 8, 2
	var wg sync.WaitGroup
	errs := make(chan string, goroutines)
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for r := 0; r < rounds; r++ {
				got, err := render()
				if err != nil {
					errs <- err.Error()
					return
				}
				if got != want {
					errs <- fmt.Sprintf("render %x != sequential %x", got, want)
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
}
