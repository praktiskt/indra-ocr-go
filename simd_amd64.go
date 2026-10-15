//go:build goexperiment.simd && amd64

package indra

// AMD64 SIMD fast paths for fillInk and the CropLine ink conversion,
// compiled only under GOEXPERIMENT=simd. Bit-identical to the scalar bodies:
// uint8 widened exactly, float32 division per IEEE, and the 1-x expression
// computed in the same order. Only enabled when the running CPU supports
// AVX-512 (checked at init time); otherwise the scalar path stays installed.

import "simd/archsimd"

func init() {
	// No vector instruction may execute before this CPU check:
	// BroadcastFloat32x16 compiles to an EVEX-encoded ZMM op, which SIGILLs
	// on a CPU without AVX-512. A vector type's Len() is a compile-time
	// constant on amd64 (always 16), so it cannot detect the CPU; gate on
	// the runtime feature check instead.
	if archsimd.X86.AVX512() {
		enableFillSimd()
	}
}

// enableFillSimd installs the AVX-512 kernels. It is kept out of init and
// marked noinline so the EVEX instructions that materialize the broadcast
// constants can only ever be reached once archsimd.X86.AVX512() is true;
// an inlined body would still sit behind the branch, but keeping it separate
// makes the guarantee independent of the inliner.
//
//go:noinline
func enableFillSimd() {
	c255fFill = archsimd.BroadcastFloat32x16(255)
	oneVecFill = archsimd.BroadcastFloat32x16(1)
	fillInkRows = fillInkRowsSimd
	inkSet1Fn = inkSet1Simd
}

var (
	c255fFill  archsimd.Float32x16
	oneVecFill archsimd.Float32x16
)

// fillInkRowsSimd vectorizes whole 16-lane groups in each row; the row tails
// (and the last group of the very last row, where backing memory ends) stay
// scalar.
func fillInkRowsSimd(ink []float32, pw int, src []uint8, sw, nw, y0, y1 int) {
	const g = 16
	for y := y0; y < y1; y++ {
		iu, ou := y*sw, y*pw
		limit := nw
		if iu+limit > len(src) {
			limit = len(src) - iu
		}
		x := 0
		for ; x+g <= limit; x += g {
			p := archsimd.LoadUint8x16(src[iu+x : iu+x+16])
			f := uint8ToFloat16(p)
			res := oneVecFill.Sub(f.Div(c255fFill))
			res.Store(ink[ou+x : ou+x+16])
		}
		fillInkRowsTailScalar(ink, pw, src, sw, nw, y, x)
	}
}

// inkSet1Simd fills a flat sink with 1 - p/255 elementwise (used by CropLine
// and any whole-slice ink conversion).
func inkSet1Simd(ink []float32, src []uint8) {
	const g = 16
	n := len(src)
	limit := min(n, len(ink))
	x := 0
	for ; x+g <= limit; x += g {
		p := archsimd.LoadUint8x16(src[x : x+16])
		f := uint8ToFloat16(p)
		res := oneVecFill.Sub(f.Div(c255fFill))
		res.Store(ink[x : x+16])
	}
	inkSet1Scalar(ink[x:limit], src[x:limit])
}

// uint8ToFloat16 widens 16 bytes to exact float32 values.
func uint8ToFloat16(p archsimd.Uint8x16) archsimd.Float32x16 {
	u32 := p.ExtendToUint32()
	return u32.AsInt32x16().ConvertToFloat32()
}

// fillInkRowsTailScalar covers the row tail from x for one row.
func fillInkRowsTailScalar(ink []float32, pw int, src []uint8, sw, nw, y, x int) {
	iu, ou := y*sw, y*pw
	for ; x < nw; x++ {
		ink[ou+x] = 1 - float32(src[iu+x])/255
	}
}
