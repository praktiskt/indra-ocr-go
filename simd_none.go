//go:build !(goexperiment.simd && amd64)

package indra

// Non-SIMD fallback: both ink kernels run scalar.

func init() {
	fillInkRows = fillInkRowsScalar
	inkSet1Fn = inkSet1Scalar
}
