package indra

// CallAll pipelines a single handle's stages across pages; it must not change
// results, only overlap the work. Gated like the other model tests.

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func testPages(t *testing.T) []*Gray {
	t.Helper()
	names := []string{"rotated_letter", "table_pmc", "multi_column", "chinese_essay"}
	gs := make([]*Gray, len(names))
	for i, n := range names {
		gs[i] = loadTestGray(t, filepath.Join("testdata", n+".png"))
	}
	return gs
}

func TestCallAllMatchesCall(t *testing.T) {
	if os.Getenv("INDRA_TEST_IMAGES") == "" {
		t.Skip("INDRA_TEST_IMAGES not set")
	}
	if err := InitLib(); err != nil {
		t.Fatalf("InitLib: %v", err)
	}
	// Single-threaded sessions keep ORT reductions deterministic so the
	// comparison is exact.
	ocr, err := NewOCR(1)
	if err != nil {
		t.Fatalf("NewOCR: %v", err)
	}
	defer ocr.Close()

	gs := testPages(t)
	want := make([]*Page, len(gs))
	for i, g := range gs {
		want[i] = ocr.Call(g)
	}
	got := ocr.CallAll(gs)
	if len(got) != len(gs) {
		t.Fatalf("CallAll len %d, want %d", len(got), len(gs))
	}
	for i := range gs {
		if !reflect.DeepEqual(got[i], want[i]) {
			t.Fatalf("page %d: CallAll != Call\n got %+v\nwant %+v", i, got[i], want[i])
		}
	}
}

func TestCallAllEmpty(t *testing.T) {
	if err := InitLib(); err != nil {
		t.Fatalf("InitLib: %v", err)
	}
	ocr, err := NewOCR(1)
	if err != nil {
		t.Fatalf("NewOCR: %v", err)
	}
	defer ocr.Close()
	if got := ocr.CallAll(nil); len(got) != 0 {
		t.Fatalf("CallAll(nil) = %d pages, want 0", len(got))
	}
}

func TestPoolCallAllMatchesCall(t *testing.T) {
	if os.Getenv("INDRA_TEST_IMAGES") == "" {
		t.Skip("INDRA_TEST_IMAGES not set")
	}
	if err := InitLib(); err != nil {
		t.Fatalf("InitLib: %v", err)
	}
	ref, err := NewOCR(1)
	if err != nil {
		t.Fatalf("NewOCR: %v", err)
	}
	defer ref.Close()
	pool, err := NewPool(2, 1)
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	defer pool.Close()

	gs := testPages(t)
	got := pool.CallAll(gs)
	if len(got) != len(gs) {
		t.Fatalf("Pool.CallAll len %d, want %d", len(got), len(gs))
	}
	for i, g := range gs {
		want := ref.Call(g)
		if !reflect.DeepEqual(got[i], want) {
			t.Fatalf("page %d: Pool.CallAll != Call", i)
		}
	}
}
