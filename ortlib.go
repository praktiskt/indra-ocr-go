// ONNX Runtime library loading. Requires CGO; native libs bundled on
// linux/amd64, otherwise ORT_LIB_PATH. Embedded libs load via anonymous memfd:
// nothing is written to disk.
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
// is written to disk. newSessions calls it for you; call directly
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
