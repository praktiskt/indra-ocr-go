// Package indra is a native Go port of the Indra OCR pipeline
// (github.com/datalab-to/indra, Apache-2.0).
//
// Indra is a ~1M-parameter recognizer plus small detector / ordering /
// orientation models running on ONNX Runtime, CPU only. The weights use
// Datalab's modified OpenRAIL-M license (see LICENSE-MODELS), the code is
// Apache-2.0.
//
// Models (datalab-to/indra-ocr v3.1) and native libs are embedded:
// linux/amd64 needs no installs, no downloads, no disk writes.
// Other platforms must provide libonnxruntime via ORT_LIB_PATH.
//
// Requires CGO.
package indra
