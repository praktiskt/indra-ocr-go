// Worker pool over OCR sessions. ONNX Runtime sessions are not safe for
// concurrent Run, so parallelism comes from one session set per worker
// (upstream runs one single-threaded worker process per core).
package indra

import (
	"bytes"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"runtime"
	"sync"
)

// PerHandleBytes is a conservative estimate of the resident memory one pooled
// OCR handle adds (four sessions loaded and warmed). Measured marginal cost is
// ~120MB on small pages and up to ~260MB with large pages; the estimate errs
// high so MemBudget never overshoots. It covers handles only: in-flight input
// pages are the caller's memory.
const PerHandleBytes = 256 << 20

// AutoWorkers is the worker count pool defaults use: half the logical CPUs,
// one single-threaded handle each. On a 16-core box throughput plateaus from
// ~6 to ~16 workers at ~17.5 pages/s, so half the cores reaches the plateau
// while leaving headroom for the pipeline's detection/orientation stages and
// the Go runtime, and roughly halves the RAM of a one-per-core pool.
func AutoWorkers() int {
	if n := runtime.NumCPU() / 2; n > 0 {
		return n
	}
	return 1
}

// resolvePool applies the defaults and the memory cap to a builder's fields.
// Pure so it can be tested without loading ONNX Runtime.
func resolvePool(workers, threads int, tc *Threads, memBudget int64) (int, Threads) {
	var opts Threads
	switch {
	case tc != nil:
		opts = *tc
	case threads > 0:
		opts = DefaultThreads(threads)
	default:
		opts = DefaultThreads(1)
	}
	if workers <= 0 {
		workers = AutoWorkers()
	}
	if memBudget > 0 {
		if cap := int(memBudget / PerHandleBytes); cap < workers {
			if cap < 1 {
				cap = 1
			}
			workers = cap
		}
	}
	return workers, opts
}

// PoolBuilder configures a Pool. Zero-valued fields fall back to host-tuned
// defaults: NewPoolBuilder().Build() is equivalent to NewPoolAuto().
type PoolBuilder struct {
	workers   int
	threads   int
	tc        *Threads
	memBudget int64
	furniture bool
}

// NewPoolBuilder starts a builder with all defaults (see Build).
func NewPoolBuilder() *PoolBuilder { return &PoolBuilder{} }

// Workers sets the number of pooled handles. <=0 means auto (AutoWorkers).
func (b *PoolBuilder) Workers(n int) *PoolBuilder { b.workers = n; return b }

// Threads sets intra-op threads for every model (small models cap at 2).
// <=0 means the pool default of 1.
func (b *PoolBuilder) Threads(n int) *PoolBuilder { b.threads = n; return b }

// ThreadsPerModel sets per-model thread counts, overriding Threads.
func (b *PoolBuilder) ThreadsPerModel(tc Threads) *PoolBuilder { b.tc = &tc; return b }

// MemBudget caps workers so estimated resident memory stays within bytes
// (using PerHandleBytes, which covers pool handles only, not in-flight input
// pages). <=0 means no cap.
func (b *PoolBuilder) MemBudget(bytes int64) *PoolBuilder { b.memBudget = bytes; return b }

// Furniture keeps page headers and footers in each page's Markdown
// (upstream load(furniture=True)); default drops them.
func (b *PoolBuilder) Furniture(v bool) *PoolBuilder { b.furniture = v; return b }

// Build opens the pool.
func (b *PoolBuilder) Build() (*Pool, error) {
	workers, tc := resolvePool(b.workers, b.threads, b.tc, b.memBudget)
	return newPool(workers, tc, b.furniture)
}

// NewPoolAuto opens a pool sized for the host: AutoWorkers single-threaded
// handles. Equivalent to NewPoolBuilder().Build().
func NewPoolAuto() (*Pool, error) { return NewPoolBuilder().Build() }

// Pool is a fixed set of OCR handles shared by concurrent Callers.
type Pool struct {
	ch      chan *OCR
	Workers int
}

// NewPool opens workers OCR handles, threads intra-op threads each.
// threads=1 mirrors upstream (one single-threaded worker per core).
// For host-tuned defaults use NewPoolAuto, or NewPoolBuilder for full control.
func NewPool(workers, threads int) (*Pool, error) {
	return NewPoolWithThreads(workers, DefaultThreads(threads))
}

// NewPoolWithThreads opens workers OCR handles with per-model counts.
func NewPoolWithThreads(workers int, tc Threads) (*Pool, error) {
	return newPool(workers, tc, false)
}

// newPool opens workers handles with per-model thread counts and options.
func newPool(workers int, tc Threads, furniture bool) (*Pool, error) {
	if workers < 1 {
		workers = 1
	}
	p := &Pool{ch: make(chan *OCR, workers), Workers: workers}
	var mu sync.Mutex
	var firstErr error
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			o, err := NewOCRWithThreads(tc)
			if err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
				return
			}
			o.Furniture = furniture
			p.ch <- o
		}()
	}
	wg.Wait()
	if firstErr != nil {
		p.Close()
		return nil, firstErr
	}
	return p, nil
}

// Call OCRs one page image, blocking for a free worker.
func (p *Pool) Call(g *Gray) *Page {
	o := <-p.ch
	defer func() { p.ch <- o }()
	return o.Call(g)
}

// CallText OCRs one page with a caller-supplied PDF text layer (see
// OCR.CallText), blocking for a free worker.
func (p *Pool) CallText(g *Gray, tl *TextLayer) *Page {
	o := <-p.ch
	defer func() { p.ch <- o }()
	return o.CallText(g, tl)
}

// CallAll OCRs pages, spreading them over the pool's workers and pipelining
// stages within each worker (see OCR.CallAll). Results keep input order.
// For a fixed worker count this sustains more pages/s than looping Call,
// because the wide, idle-non-rec footprint of each worker is filled.
func (p *Pool) CallAll(gs []*Gray) []*Page {
	out := make([]*Page, len(gs))
	n := len(gs)
	if n == 0 {
		return out
	}
	// A few chunks per worker keeps every pipeline fed and balances load.
	chunk := (n + p.Workers*4 - 1) / (p.Workers * 4)
	if chunk < 1 {
		chunk = 1
	}
	var next int64
	var mu sync.Mutex
	var wg sync.WaitGroup
	for w := 0; w < p.Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			o := <-p.ch
			defer func() { p.ch <- o }()
			for {
				mu.Lock()
				lo := next
				next += int64(chunk)
				mu.Unlock()
				if int(lo) >= n {
					return
				}
				hi := min(int(lo)+chunk, n)
				copy(out[lo:hi], o.CallAll(gs[lo:hi]))
			}
		}()
	}
	wg.Wait()
	return out
}

// CallFiles reads and decodes image files and OCRs them over the pool, like
// the reference convert_many(paths): callers pass paths, not decoded pages.
// Chunks of 2*workers pages are decoded in parallel, so decoding overlaps OCR
// and peak memory stays at one chunk of pages. Results keep input order; work
// stops at the first error.
func (p *Pool) CallFiles(paths []string) ([]*Page, error) {
	out := make([]*Page, len(paths))
	n := len(paths)
	if n == 0 {
		return out, nil
	}
	chunk := 4 * p.Workers
	if chunk < 1 {
		chunk = 1
	}
	gs := make([]*Gray, n)
	var firstErr error
	var errMu sync.Mutex
	fail := func(err error) {
		errMu.Lock()
		if firstErr == nil {
			firstErr = err
		}
		errMu.Unlock()
	}
	// Decode chunk 0, then overlap: each chunk's OCR runs while chunk +1
	// decodes (bounded at two chunks of pages in memory). Set-order result.
	chunks := (n + chunk - 1) / chunk

	asyncDecode := func(k int, dst chan error) {
		lo, hi := k*chunk, min((k+1)*chunk, n)
		go func() {
			var wg sync.WaitGroup
			for i := lo; i < hi; i++ {
				wg.Add(1)
				go func(i int) {
					defer wg.Done()
					data, err := os.ReadFile(paths[i])
					var img image.Image
					if err == nil {
						img, _, err = image.Decode(bytes.NewReader(data))
					}
					if err != nil {
						fail(err)
						return
					}
					gs[i] = FromImage(img)
				}(i)
			}
			wg.Wait()
			errMu.Lock()
			dst <- firstErr
			errMu.Unlock()
		}()
	}
	cur := make(chan error, 1)
	asyncDecode(0, cur)
	for k := 0; k < chunks; k++ {
		if err := <-cur; err != nil {
			return nil, err
		}
		var next chan error
		if k+1 < chunks {
			next = make(chan error, 1)
			asyncDecode(k+1, next)
		}
		lo, hi := k*chunk, min((k+1)*chunk, n)
		copy(out[lo:hi], p.CallAll(gs[lo:hi]))
		cur = next
	}
	return out, nil
}

// Close releases all sessions.
func (p *Pool) Close() {
	for {
		select {
		case o := <-p.ch:
			o.Close()
		default:
			return
		}
	}
}
