// Package indra is a native Go port of the Indra OCR pipeline
// (github.com/datalab-to/indra, Apache-2.0).
//
// Indra is a ~1M-parameter recognizer plus small detector / ordering /
// orientation models running on ONNX Runtime, CPU only. The weights use
// Datalab's modified OpenRAIL-M license (see LICENSE-MODELS), the code is
// Apache-2.0.
//
// Models are embedded via go:embed, no download required.
//
// Requires CGO and libonnxruntime (see session_ort.go).
package indra
