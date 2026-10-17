// ONNX Runtime sessions and per-model thread config.
package indra

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

type sessions struct {
	det    *ort.DynamicAdvancedSession
	rec    *ort.DynamicAdvancedSession
	recCJK *ort.DynamicAdvancedSession // CJK recognizer (routed by script)
	script *ort.DynamicAdvancedSession // per-line script classifier
	order  *ort.DynamicAdvancedSession
	orient *ort.DynamicAdvancedSession
}

func newOptions(threads int) (*ort.SessionOptions, error) {
	o, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	if err := o.SetIntraOpNumThreads(threads); err != nil {
		return nil, err
	}
	if err := o.SetInterOpNumThreads(1); err != nil {
		return nil, err
	}
	if err := o.SetGraphOptimizationLevel(ort.GraphOptimizationLevelEnableAll); err != nil {
		return nil, err
	}
	// CPU memory arena OFF. Matches upstream indra_ocr and keeps RSS within
	// PerHandleBytes: measured on the FUNSD corpus at 8 workers x 1 thread,
	// arena ON is ~3% faster (22.95 vs 22.12 pages/s) but ~245MB -> ~360MB
	// RSS per handle (2.88GB vs 1.96GB, and still growing with pages). The
	// 512px rec width batching bounds the per-shape growth, but the RSS cost
	// is real enough that the memory budget assumes arena off.
	if err := o.SetCpuMemArena(false); err != nil {
		return nil, err
	}
	if err := o.SetMemPattern(true); err != nil {
		return nil, err
	}
	if err := o.SetLogSeverityLevel(ort.LoggingLevelFatal); err != nil {
		return nil, err
	}
	return o, nil
}

// Threads sets per-model intra-op thread counts. Det/order/orient are
// small models where >2 threads adds sync overhead (measured det:
// 62ms@1, 42ms@2, 45ms@4, 48ms@8); rec scales to ~8 on a 16-core box.
type Threads struct {
	Det    int
	Rec    int
	Order  int
	Orient int
}

// DefaultThreads maps a single knob to per-model counts: rec honors it,
// the small models cap at 2.
func DefaultThreads(threads int) Threads {
	if threads < 1 {
		threads = 1
	}
	small := min(threads, 2)
	return Threads{Det: small, Rec: threads, Order: small, Orient: small}
}

// newSessions opens det/rec/order (+orient when present) from embedded models.
func newSessions(threads int) (*sessions, error) {
	return newSessionsWithThreads(DefaultThreads(threads))
}

// newSessionsWithThreads opens sessions with per-model thread counts.
func newSessionsWithThreads(tc Threads) (*sessions, error) {
	if err := InitLib(); err != nil {
		return nil, fmt.Errorf("indra: onnxruntime init: %w", err)
	}
	mkWith := func(file string, in, out []string, threads int) (*ort.DynamicAdvancedSession, error) {
		o, err := newOptions(threads)
		if err != nil {
			return nil, err
		}
		defer o.Destroy()
		data, err := modelFS.ReadFile("models/" + file)
		if err != nil {
			return nil, err
		}
		return ort.NewDynamicAdvancedSessionWithONNXData(data, in, out, o)
	}
	s := &sessions{}
	var err error
	if s.det, err = mkWith("det.onnx", []string{"x"}, []string{"out"}, tc.Det); err != nil {
		return nil, fmt.Errorf("indra: det: %w", err)
	}
	if s.rec, err = mkWith("rec.onnx", []string{"x", "widths"}, []string{"ids", "conf", "lengths"}, tc.Rec); err != nil {
		s.close()
		return nil, fmt.Errorf("indra: rec: %w", err)
	}
	if s.recCJK, err = mkWith("rec_cjk.onnx", []string{"x", "widths"}, []string{"ids", "conf", "lengths"}, tc.Rec); err != nil {
		s.close()
		return nil, fmt.Errorf("indra: rec_cjk: %w", err)
	}
	if s.script, err = mkWith("script.onnx", []string{"x", "widths"}, []string{"p"}, tc.Order); err != nil {
		s.close()
		return nil, fmt.Errorf("indra: script: %w", err)
	}
	if s.order, err = mkWith("order.onnx", []string{"feats", "mask"}, []string{"a", "cls", "same"}, tc.Order); err != nil {
		s.close()
		return nil, fmt.Errorf("indra: order: %w", err)
	}
	if s.orient, err = mkWith("orient.onnx", []string{"x"}, []string{"p"}, tc.Orient); err != nil {
		s.close()
		return nil, fmt.Errorf("indra: orient: %w", err)
	}
	return s, nil
}

// Close destroys sessions (process env stays alive).
func (s *sessions) close() {
	if s == nil {
		return
	}
	for _, d := range []*ort.DynamicAdvancedSession{s.det, s.rec, s.recCJK, s.script, s.order, s.orient} {
		if d != nil {
			d.Destroy()
		}
	}
}

// runDet runs det.onnx on ink [1,1,ph,pw], returns [3*oh*ow] + oh,ow.
func (s *sessions) runDet(ink []float32, ph, pw int) ([]float32, int, int, error) {
	x, err := ort.NewTensor(ort.Shape{1, 1, int64(ph), int64(pw)}, ink)
	if err != nil {
		return nil, 0, 0, err
	}
	defer x.Destroy()
	y, err := ort.NewEmptyTensor[float32](ort.Shape{1, 3, int64(ph / 2), int64(pw / 2)})
	if err != nil {
		return nil, 0, 0, err
	}
	defer y.Destroy()
	if err := s.det.Run([]ort.Value{x}, []ort.Value{y}); err != nil {
		return nil, 0, 0, err
	}
	sh := y.GetShape()
	return y.GetData(), int(sh[2]), int(sh[3]), nil
}

// recOut is one batch of rec.onnx outputs.
type recOut struct {
	IDs  []int64
	Conf []float32
	Lens []int64
	B, T int
}

// runRec runs rec.onnx on a padded batch [B,1,H,Wm].
func (s *sessions) runRec(batch []float32, b, h, wm int, widths []int64) (*recOut, error) {
	return recBatch(s.rec, batch, b, h, wm, widths)
}

// runRecCJK runs rec_cjk.onnx on a padded batch [B,1,H,Wm].
func (s *sessions) runRecCJK(batch []float32, b, h, wm int, widths []int64) (*recOut, error) {
	return recBatch(s.recCJK, batch, b, h, wm, widths)
}

func recBatch(sess *ort.DynamicAdvancedSession, batch []float32, b, h, wm int, widths []int64) (*recOut, error) {
	x, err := ort.NewTensor(ort.Shape{int64(b), 1, int64(h), int64(wm)}, batch)
	if err != nil {
		return nil, err
	}
	defer x.Destroy()
	w, err := ort.NewTensor(ort.Shape{int64(b)}, widths)
	if err != nil {
		return nil, err
	}
	defer w.Destroy()
	t := wm / 4
	if t < 1 {
		t = 1
	}
	ids, err := ort.NewEmptyTensor[int64](ort.Shape{int64(b), int64(t)})
	if err != nil {
		return nil, err
	}
	defer ids.Destroy()
	conf, err := ort.NewEmptyTensor[float32](ort.Shape{int64(b), int64(t)})
	if err != nil {
		return nil, err
	}
	defer conf.Destroy()
	lens, err := ort.NewEmptyTensor[int64](ort.Shape{int64(b)})
	if err != nil {
		return nil, err
	}
	defer lens.Destroy()
	if err := sess.Run([]ort.Value{x, w}, []ort.Value{ids, conf, lens}); err != nil {
		return nil, err
	}
	tt := int(ids.GetShape()[1])
	return &recOut{
		IDs:  ids.GetData(),
		Conf: conf.GetData(),
		Lens: lens.GetData(),
		B:    b, T: tt,
	}, nil
}

// runScript runs script.onnx on a padded batch [B,1,H,Wm], returning per-line
// P(the line holds Chinese characters).
func (s *sessions) runScript(batch []float32, b, h, wm int, widths []int64) ([]float32, error) {
	x, err := ort.NewTensor(ort.Shape{int64(b), 1, int64(h), int64(wm)}, batch)
	if err != nil {
		return nil, err
	}
	defer x.Destroy()
	w, err := ort.NewTensor(ort.Shape{int64(b)}, widths)
	if err != nil {
		return nil, err
	}
	defer w.Destroy()
	p, err := ort.NewEmptyTensor[float32](ort.Shape{int64(b)})
	if err != nil {
		return nil, err
	}
	defer p.Destroy()
	if err := s.script.Run([]ort.Value{x, w}, []ort.Value{p}); err != nil {
		return nil, err
	}
	return p.GetData(), nil
}

// orderOut holds order.onnx outputs for N lines.
type orderOut struct {
	A    []float32 // [N*N]
	Cls  []float32 // [N*10]
	Same []float32 // [N*N]
	N    int
}

// runOrder runs order.onnx for N lines (feats [1,N,16]).
func (s *sessions) runOrder(feats []float32, n int) (*orderOut, error) {
	f, err := ort.NewTensor(ort.Shape{1, int64(n), NFeats}, feats)
	if err != nil {
		return nil, err
	}
	defer f.Destroy()
	mask := make([]bool, n)
	for i := range mask {
		mask[i] = true
	}
	mk, err := ort.NewTensor(ort.Shape{1, int64(n)}, mask)
	if err != nil {
		return nil, err
	}
	defer mk.Destroy()
	a, err := ort.NewEmptyTensor[float32](ort.Shape{1, int64(n), int64(n)})
	if err != nil {
		return nil, err
	}
	defer a.Destroy()
	cls, err := ort.NewEmptyTensor[float32](ort.Shape{1, int64(n), int64(len(blockClasses))})
	if err != nil {
		return nil, err
	}
	defer cls.Destroy()
	same, err := ort.NewEmptyTensor[float32](ort.Shape{1, int64(n), int64(n)})
	if err != nil {
		return nil, err
	}
	defer same.Destroy()
	// NOTE: order.onnx also produces a "tab" [1,3,N,N] head, but the Go
	// port renders tables geometrically and never reads it. Only the
	// needed outputs are fetched so ORT can prune the tab head.
	if err := s.order.Run([]ort.Value{f, mk}, []ort.Value{a, cls, same}); err != nil {
		return nil, err
	}
	return &orderOut{
		A:    a.GetData(),
		Cls:  cls.GetData(),
		Same: same.GetData(),
		N:    n,
	}, nil
}

// runOrient runs orient.onnx on a 768x768 ink map, returns 4 class probs.
func (s *sessions) runOrient(ink []float32) ([4]float32, error) {
	var p [4]float32
	if s.orient == nil {
		return p, nil
	}
	x, err := ort.NewTensor(ort.Shape{1, 1, 768, 768}, ink)
	if err != nil {
		return p, err
	}
	defer x.Destroy()
	y, err := ort.NewEmptyTensor[float32](ort.Shape{1, 4})
	if err != nil {
		return p, err
	}
	defer y.Destroy()
	if err := s.orient.Run([]ort.Value{x}, []ort.Value{y}); err != nil {
		return p, err
	}
	copy(p[:], y.GetData())
	return p, nil
}
