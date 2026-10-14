package indra

import "embed"

// modelFS holds the embedded Indra OCR models (v3.1).
//
//go:embed models/det.onnx models/rec.onnx models/order.onnx models/meta.json models/orient.onnx models/script.onnx models/rec_cjk.onnx
var modelFS embed.FS

// libFS holds the embedded native libraries (linux/amd64):
// ONNX Runtime 1.31.0 plus its C++ runtime dependencies.
//
//go:embed lib/libonnxruntime.so lib/libstdc++.so.6 lib/libgcc_s.so.1
var libFS embed.FS
