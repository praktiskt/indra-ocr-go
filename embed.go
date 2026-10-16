package indra

import "embed"

// modelFS holds the embedded Indra OCR models (datalab-to/indra-ocr v3.1):
// det, rec, rec_cjk (CJK recognizer), script (the line script classifier that
// routes to rec_cjk), order, orient + meta.json.
//
//go:embed models/det.onnx models/rec.onnx models/rec_cjk.onnx models/script.onnx models/order.onnx models/meta.json models/orient.onnx
var modelFS embed.FS

// libFS holds the embedded native libraries (linux/amd64):
// ONNX Runtime 1.31.0 plus its C++ runtime dependencies.
//
//go:embed lib/libonnxruntime.so lib/libstdc++.so.6 lib/libgcc_s.so.1
var libFS embed.FS
