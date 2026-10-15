// Worker pool over OCR sessions. ONNX Runtime sessions are not safe for
// concurrent Run, so parallelism comes from one session set per worker
// (upstream runs one single-threaded worker process per core).
package indra

import "sync"

// Pool is a fixed set of OCR handles shared by concurrent Callers.
type Pool struct {
	ch      chan *OCR
	Workers int
}

// NewPool opens workers OCR handles, threads intra-op threads each.
// threads=1 mirrors upstream (one single-threaded worker per core).
func NewPool(workers, threads int) (*Pool, error) {
	return NewPoolWithThreads(workers, DefaultThreads(threads))
}

// NewPoolWithThreads opens workers OCR handles with per-model counts.
func NewPoolWithThreads(workers int, tc Threads) (*Pool, error) {
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
