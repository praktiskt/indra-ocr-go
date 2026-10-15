// ONNX Runtime sessions. Requires CGO; native libs bundled on linux/amd64.
package indra

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"os"
	"sync"
	"syscall"
	"unsafe"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	envOnce sync.Once
	envErr  error
)

// InitLib loads libonnxruntime from ORT_LIB_PATH or the bundled
// embedded copy. Embedded libs load via anonymous memfd: nothing
// is written to disk. NewSessions calls it for you; call directly
// only to fail fast on load. Safe for concurrent use.
func InitLib() error {
	envOnce.Do(func() {
		if !ort.IsInitialized() {
			if p := os.Getenv("ORT_LIB_PATH"); p != "" {
				ort.SetSharedLibraryPath(p)
			} else if err := loadEmbeddedLibs(); err != nil {
				envErr = err
				return
			}
			envErr = ort.InitializeEnvironment()
		}
	})
	return envErr
}

// memfdFDs keeps the anonymous fds alive for the process
// lifetime; the loader may map from them lazily.
var memfdFDs []uintptr

func dlopenMemfd(f, path string) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	if C.dlopen(cpath, C.RTLD_LAZY|C.RTLD_GLOBAL) == nil {
		return fmt.Errorf("indra: load %s: %s", f, C.GoString(C.dlerror()))
	}
	return nil
}

// loadEmbeddedLibs dlopens the bundled libgcc_s, libstdc++
// and libonnxruntime from anonymous memory (memfd), so the
// package needs no writable filesystem. libonnxruntime's
// DT_NEEDED entries resolve against the preloaded runtimes.
func loadEmbeddedLibs() error {
	if sysMemfdCreate < 0 {
		return fmt.Errorf("indra: embedded libs need linux/amd64; set ORT_LIB_PATH")
	}
	var ortPath string
	for _, f := range []string{"libgcc_s.so.1", "libstdc++.so.6", "libonnxruntime.so"} {
		data, err := libFS.ReadFile("lib/" + f)
		if err != nil {
			return fmt.Errorf("indra: embedded %s: %w", f, err)
		}
		name := append([]byte(f), 0)
		fd, _, errno := syscall.Syscall(uintptr(sysMemfdCreate),
			uintptr(unsafe.Pointer(&name[0])), 0x0001 /* MFD_CLOEXEC */, 0)
		if errno != 0 {
			return fmt.Errorf("indra: memfd %s: %w", f, errno)
		}
		if err := writeAll(int(fd), data); err != nil {
			syscall.Close(int(fd))
			return fmt.Errorf("indra: memfd %s: %w", f, err)
		}
		path := "/proc/self/fd/" + itoa(int(fd))
		if err := dlopenMemfd(f, path); err != nil {
			syscall.Close(int(fd))
			return err
		}
		memfdFDs = append(memfdFDs, fd)
		if f == "libonnxruntime.so" {
			ortPath = path
		}
	}
	ort.SetSharedLibraryPath(ortPath)
	return nil
}

func writeAll(fd int, data []byte) error {
	for len(data) > 0 {
		n, err := syscall.Write(fd, data)
		if err != nil {
			return err
		}
		data = data[n:]
	}
	return nil
}

// Sessions holds the model sessions.
type Sessions struct {
	Det    *ort.DynamicAdvancedSession
	Rec    *ort.DynamicAdvancedSession
	Order  *ort.DynamicAdvancedSession
	Orient *ort.DynamicAdvancedSession
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
	if err := o.SetCpuMemArena(true); err != nil {
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

// NewSessions opens det/rec/order (+orient when present) from embedded models.
func NewSessions(threads int) (*Sessions, error) {
	return NewSessionsWithThreads(DefaultThreads(threads))
}

// NewSessionsWithThreads opens sessions with per-model thread counts.
func NewSessionsWithThreads(tc Threads) (*Sessions, error) {
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
	s := &Sessions{}
	var err error
	if s.Det, err = mkWith("det.onnx", []string{"x"}, []string{"out"}, tc.Det); err != nil {
		return nil, fmt.Errorf("indra: det: %w", err)
	}
	if s.Rec, err = mkWith("rec.onnx", []string{"x", "widths"}, []string{"ids", "conf", "lengths"}, tc.Rec); err != nil {
		s.Close()
		return nil, fmt.Errorf("indra: rec: %w", err)
	}
	if s.Order, err = mkWith("order.onnx", []string{"feats", "mask"}, []string{"a", "cls", "same", "tab"}, tc.Order); err != nil {
		s.Close()
		return nil, fmt.Errorf("indra: order: %w", err)
	}
	if s.Orient, err = mkWith("orient.onnx", []string{"x"}, []string{"p"}, tc.Orient); err != nil {
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
	sh := y.GetShape()
	return y.GetData(), int(sh[2]), int(sh[3]), nil
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
		IDs:  ids.GetData(),
		Conf: conf.GetData(),
		Lens: lens.GetData(),
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
		A:    a.GetData(),
		Cls:  cls.GetData(),
		Same: same.GetData(),
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
