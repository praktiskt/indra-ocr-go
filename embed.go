package indra

import "embed"

// modelFS holds the embedded Indra OCR models (datalab-to/indra-ocr v3.1).
// Only the models the pipeline loads: det, rec, order, orient + meta.json.
// script.onnx / rec_cjk.onnx exist upstream but CJK routing is not
// implemented, so they are deliberately not embedded.
//
//go:embed models/det.onnx models/rec.onnx models/order.onnx models/meta.json models/orient.onnx
var modelFS embed.FS

// libFS holds the embedded native libraries (linux/amd64):
// ONNX Runtime 1.31.0 plus its C++ runtime dependencies.
//
//go:embed lib/libonnxruntime.so lib/libstdc++.so.6 lib/libgcc_s.so.1
var libFS embed.FS
