package indra

import (
	"crypto/sha256"
	"encoding/hex"
	"testing"
)

// Pinned native libraries. Regenerate libpdfium.so with tools/vendor_pdfium.sh,
// which downloads this exact pdfium-binaries release and checks the sha256.
var pinnedLibs = map[string]string{
	"lib/libpdfium.so": "fd4de9f91f183476d04b1bc3fd17dacd95aba349c55d2c376512d95e082b441e", // PDFium 156.0.8076.0
}

// TestPinnedLibraries guards the embedded native libs against an accidental
// replacement: a swap must update this hash (and be re-validated — rendering
// feeds detect/rec/order).
func TestPinnedLibraries(t *testing.T) {
	for name, want := range pinnedLibs {
		data, err := libFS.ReadFile(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		sum := sha256.Sum256(data)
		if got := hex.EncodeToString(sum[:]); got != want {
			t.Errorf("%s sha256 = %s, want %s (update tools/vendor_pdfium.sh and re-validate)", name, got, want)
		}
	}
}
