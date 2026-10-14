//go:build !linux || !amd64

package indra

// Embedded native libs are linux/amd64 only; other platforms
// must provide libonnxruntime via ORT_LIB_PATH.
const sysMemfdCreate = -1
