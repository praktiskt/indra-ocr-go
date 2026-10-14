// ONNX Runtime sessions. Requires CGO and libonnxruntime (bundled on linux/amd64).
package indra

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	envOnce sync.Once
	envErr  error
)

// InitLib loads libonnxruntime from ORT_LIB_PATH, a standard location,
// or the bundled embedded copy (extracted once to the cache dir), then
// initializes the process-wide environment. Call once before NewSessions.
func InitLib() error {
	envOnce.Do(func() {
		if !ort.IsInitialized() {
			if p := os.Getenv("ORT_LIB_PATH"); p != "" {
				ort.SetSharedLibraryPath(p)
			} else if p := findStdLib(); p != "" {
				ort.SetSharedLibraryPath(p)
			} else {
				p, err := extractLib()
				if err != nil {
					envErr = err
					return
				}
				ort.SetSharedLibraryPath(p)
			}
			envErr = ort.InitializeEnvironment()
		}
	})
	return envErr
}

func findStdLib() string {
	for _, c := range []string{
		"/usr/lib/libonnxruntime.so",
		"/usr/local/lib/libonnxruntime.so",
		"./libonnxruntime.so",
	} {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// extractLib writes the bundled libonnxruntime.so to the cache dir and
// returns its path. Subsequent runs reuse the cached copy.
func extractLib() (string, error) {
	base, err := os.UserCacheDir()
	if err != nil {
		base = os.TempDir()
	}
	dir := filepath.Join(base, "indra-ocr", "lib")
	dst := filepath.Join(dir, "libonnxruntime.so")
	if _, err := os.Stat(dst); err != nil {
		data, err := libFS.ReadFile("lib/libonnxruntime.so")
		if err != nil {
			return "", fmt.Errorf("indra: embedded libonnxruntime: %w", err)
		}
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return "", err
		}
		tmp := dst + ".part"
		if err := os.WriteFile(tmp, data, 0o755); err != nil {
			return "", err
		}
		if err := os.Rename(tmp, dst); err != nil {
			return "", err
		}
	}
	return dst, nil
}

// Sessions holds the model sessions. Mirrors OrtOCR.__init__ model set.
type Sessions struct {
	Det    *ort.DynamicAdvancedSession
	Rec    *ort.DynamicAdvancedSession
	Order  *ort.DynamicAdvancedSession
	Orient *ort.DynamicAdvancedSession // nil when absent

	Threads int
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
	if err := o.SetCpuMemArena(false); err != nil {
		return nil, err
	}
	return o, nil
}

// NewSessions opens det/rec/order (+orient when present) from embedded models.
func NewSessions(threads int) (*Sessions, error) {
	if err := InitLib(); err != nil {
		return nil, fmt.Errorf("indra: onnxruntime init: %w", err)
	}
	o, err := newOptions(threads)
	if err != nil {
		return nil, err
	}
	defer o.Destroy()
	mk := func(file string, in, out []string) (*ort.DynamicAdvancedSession, error) {
		data, err := modelFS.ReadFile("models/" + file)
		if err != nil {
			return nil, err
		}
		return ort.NewDynamicAdvancedSessionWithONNXData(data, in, out, o)
	}
	s := &Sessions{Threads: threads}
	if s.Det, err = mk("det.onnx", []string{"x"}, []string{"out"}); err != nil {
		return nil, fmt.Errorf("indra: det: %w", err)
	}
	if s.Rec, err = mk("rec.onnx", []string{"x", "widths"}, []string{"ids", "conf", "lengths"}); err != nil {
		s.Close()
		return nil, fmt.Errorf("indra: rec: %w", err)
	}
	if s.Order, err = mk("order.onnx", []string{"feats", "mask"}, []string{"a", "cls", "same", "tab"}); err != nil {
		s.Close()
		return nil, fmt.Errorf("indra: order: %w", err)
	}
	if s.Orient, err = mk("orient.onnx", []string{"x"}, []string{"p"}); err != nil {
		s.Close()
		return nil, fmt.Errorf("indra: orient: %w", err)
	}
	return s, nil
}

// Close destroys sessions (process env stays alive).
func (s *Sessions) Close() {
	if s == nil {
		return
	}
	for _, d := range []*ort.DynamicAdvancedSession{s.Det, s.Rec, s.Order, s.Orient} {
		if d != nil {
			d.Destroy()
		}
	}
}

// RunDet runs det.onnx on ink [1,1,ph,pw], returns [3*oh*ow] + oh,ow.
func (s *Sessions) RunDet(ink []float32, ph, pw int) ([]float32, int, int, error) {
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
	if err := s.Det.Run([]ort.Value{x}, []ort.Value{y}); err != nil {
		return nil, 0, 0, err
	}
	data := append([]float32(nil), y.GetData()...)
	sh := y.GetShape()
	return data, int(sh[2]), int(sh[3]), nil
}

// RecOut is one batch of rec.onnx outputs.
type RecOut struct {
	IDs  []int64
	Conf []float32
	Lens []int64
	B, T int
}

// RunRec runs rec.onnx on a padded batch [B,1,H,Wm].
func (s *Sessions) RunRec(batch []float32, b, h, wm int, widths []int64) (*RecOut, error) {
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
	if err := s.Rec.Run([]ort.Value{x, w}, []ort.Value{ids, conf, lens}); err != nil {
		return nil, err
	}
	tt := int(ids.GetShape()[1])
	return &RecOut{
		IDs:  append([]int64(nil), ids.GetData()...),
		Conf: append([]float32(nil), conf.GetData()...),
		Lens: append([]int64(nil), lens.GetData()...),
		B:    b, T: tt,
	}, nil
}

// OrderOut holds order.onnx outputs for N lines.
type OrderOut struct {
	A    []float32 // [N*N]
	Cls  []float32 // [N*10]
	Same []float32 // [N*N]
	N    int
}

// RunOrder runs order.onnx for N lines (feats [1,N,16]).
func (s *Sessions) RunOrder(feats []float32, n int) (*OrderOut, error) {
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
	cls, err := ort.NewEmptyTensor[float32](ort.Shape{1, int64(n), int64(len(BlockClasses))})
	if err != nil {
		return nil, err
	}
	defer cls.Destroy()
	same, err := ort.NewEmptyTensor[float32](ort.Shape{1, int64(n), int64(n)})
	if err != nil {
		return nil, err
	}
	defer same.Destroy()
	tab, err := ort.NewEmptyTensor[float32](ort.Shape{1, 3, int64(n), int64(n)})
	if err != nil {
		return nil, err
	}
	defer tab.Destroy()
	if err := s.Order.Run([]ort.Value{f, mk}, []ort.Value{a, cls, same, tab}); err != nil {
		return nil, err
	}
	return &OrderOut{
		A:    append([]float32(nil), a.GetData()...),
		Cls:  append([]float32(nil), cls.GetData()...),
		Same: append([]float32(nil), same.GetData()...),
		N:    n,
	}, nil
}

// RunOrient runs orient.onnx on a 768x768 ink map, returns 4 class probs.
func (s *Sessions) RunOrient(ink []float32) ([4]float32, error) {
	var p [4]float32
	if s.Orient == nil {
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
	if err := s.Orient.Run([]ort.Value{x}, []ort.Value{y}); err != nil {
		return p, err
	}
	copy(p[:], y.GetData())
	return p, nil
}
