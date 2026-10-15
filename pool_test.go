package indra

// Pure tests for the pool autotune defaults; no ONNX Runtime needed.

import (
	"runtime"
	"testing"
)

func TestAutoWorkers(t *testing.T) {
	if got := AutoWorkers(); got != max(1, runtime.NumCPU()/2) {
		t.Fatalf("AutoWorkers() = %d", got)
	}
}

func TestResolvePoolDefaults(t *testing.T) {
	gotW, gotT := resolvePool(0, 0, nil, 0)
	if gotW != AutoWorkers() {
		t.Errorf("workers = %d, want AutoWorkers=%d", gotW, AutoWorkers())
	}
	if want := DefaultThreads(1); gotT != want {
		t.Errorf("threads = %+v, want %+v", gotT, want)
	}

	if w, tc := resolvePool(4, 0, nil, 0); w != 4 || tc != DefaultThreads(1) {
		t.Errorf("explicit workers: got %d %+v", w, tc)
	}
	if w, tc := resolvePool(0, 4, nil, 0); w != AutoWorkers() || tc != DefaultThreads(4) {
		t.Errorf("explicit threads: got %d %+v", w, tc)
	}

	perModel := Threads{Det: 3, Rec: 7, Order: 2, Orient: 1}
	if _, tc := resolvePool(0, 4, &perModel, 0); tc != perModel {
		t.Errorf("ThreadsPerModel must win: got %+v", tc)
	}
}

func TestResolvePoolMemBudget(t *testing.T) {
	// Budget below one handle still yields a working pool.
	if w, _ := resolvePool(0, 0, nil, PerHandleBytes/2); w != 1 {
		t.Errorf("tiny budget: workers = %d, want 1", w)
	}
	if w, _ := resolvePool(0, 0, nil, 3*PerHandleBytes); w != 3 {
		t.Errorf("3-handle budget: workers = %d, want 3", w)
	}
	// A cap above the default does not raise the worker count.
	if w, _ := resolvePool(2, 0, nil, 64*PerHandleBytes); w != 2 {
		t.Errorf("generous budget: workers = %d, want 2 (unchanged)", w)
	}
	// No budget means no cap.
	if w, _ := resolvePool(5, 0, nil, 0); w != 5 {
		t.Errorf("no budget: workers = %d, want 5", w)
	}
}

func TestPoolBuilderChain(t *testing.T) {
	// Build without touching ORT is not possible, but the builder must be
	// chainable and store its fields.
	b := NewPoolBuilder().Workers(3).Threads(2).MemBudget(8 * PerHandleBytes)
	if b.workers != 3 || b.threads != 2 || b.memBudget != 8*PerHandleBytes {
		t.Fatalf("builder fields not stored: %+v", b)
	}
	w, tc := resolvePool(b.workers, b.threads, b.tc, b.memBudget)
	if w != 3 || tc != DefaultThreads(2) {
		t.Fatalf("resolve: %d %+v", w, tc)
	}
}
