package indra

import "embed"

// modelFS holds the embedded Indra OCR models (v3.1).
//
//go:embed models/det.onnx models/rec.onnx models/order.onnx models/meta.json models/orient.onnx models/script.onnx models/rec_cjk.onnx
var modelFS embed.FS

// libFS holds the embedded ONNX Runtime shared library (linux/amd64, v1.31.0).
//
//go:embed lib/libonnxruntime.so
var libFS embed.FS
