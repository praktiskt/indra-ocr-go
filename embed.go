package indra

import "embed"

// modelFS holds the embedded Indra OCR models (datalab-to/indra-ocr v1.1.0):
// det, rec, rec_cjk, rec_rtl and rec_world (the script-group recognizers),
// script (the line script router), order, orient, the n-gram language models
// (lm.npz, lm_cjk.npz, lm_world.npz), the math models (mathcls, math_enc,
// math_dec_init, math_dec_step and math_meta.json) and meta.json.
//
//go:embed models/det.onnx models/rec.onnx models/rec_cjk.onnx models/rec_rtl.onnx models/rec_world.onnx models/script.onnx models/order.onnx models/meta.json models/orient.onnx models/lm.npz models/lm_cjk.npz models/lm_world.npz models/mathcls.onnx models/math_enc.onnx models/math_dec_init.onnx models/math_dec_step.onnx models/math_meta.json
var modelFS embed.FS

// libFS holds the embedded native libraries (linux/amd64):
// ONNX Runtime 1.31.0 + its C++ runtime dependencies, and PDFium (the PDF
// renderer + text-layer extractor used by CallPDF).
//
//go:embed lib/libonnxruntime.so lib/libstdc++.so.6 lib/libgcc_s.so.1 lib/libpdfium.so
var libFS embed.FS
