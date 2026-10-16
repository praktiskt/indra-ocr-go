package indra

// cgo bridge to the embedded libpdfium.so. The library is loaded from an
// anonymous memfd at runtime and every entry point is resolved with dlsym, so
// nothing links against pdfium at build time and nothing is installed.

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

typedef void* FPDF_DOCUMENT;
typedef void* FPDF_PAGE;
typedef void* FPDF_TEXTPAGE;
typedef void* FPDF_BITMAP;
typedef void* FPDF_PAGEOBJECT;

static void* g_pdf = 0;

static int pdfium_open(const char* path) {
	g_pdf = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
	return g_pdf ? 0 : 1;
}
static int pdfium_loaded(void) { return g_pdf != 0; }

static void pdfium_initlib(void) {
	static void (*f)(void);
	if (!f) f = (void(*)(void))dlsym(g_pdf, "FPDF_InitLibrary");
	f();
}
static void pdfium_destroylib(void) {
	static void (*f)(void);
	if (!f) f = (void(*)(void))dlsym(g_pdf, "FPDF_DestroyLibrary");
	f();
}
static FPDF_DOCUMENT pdfium_loadmem(const void* p, size_t n, const char* pw) {
	static FPDF_DOCUMENT (*f)(const void*, size_t, const char*);
	if (!f) f = (void*(*)(const void*,size_t,const char*))dlsym(g_pdf, "FPDF_LoadMemDocument64");
	return f(p, n, pw);
}
static int pdfium_pagecount(FPDF_DOCUMENT d) {
	static int (*f)(FPDF_DOCUMENT);
	if (!f) f = (int(*)(void*))dlsym(g_pdf, "FPDF_GetPageCount");
	return f(d);
}
static FPDF_PAGE pdfium_loadpage(FPDF_DOCUMENT d, int i) {
	static FPDF_PAGE (*f)(FPDF_DOCUMENT, int);
	if (!f) f = (void*(*)(void*,int))dlsym(g_pdf, "FPDF_LoadPage");
	return f(d, i);
}
static float pdfium_pagewidth(FPDF_PAGE p) {
	static float (*f)(FPDF_PAGE);
	if (!f) f = (float(*)(void*))dlsym(g_pdf, "FPDF_GetPageWidthF");
	return f(p);
}
static float pdfium_pageheight(FPDF_PAGE p) {
	static float (*f)(FPDF_PAGE);
	if (!f) f = (float(*)(void*))dlsym(g_pdf, "FPDF_GetPageHeightF");
	return f(p);
}
static FPDF_BITMAP pdfium_bitmap(int w, int h, int fmt, void* buf, int stride) {
	static FPDF_BITMAP (*f)(int,int,int,void*,int);
	if (!f) f = (void*(*)(int,int,int,void*,int))dlsym(g_pdf, "FPDFBitmap_CreateEx");
	return f(w,h,fmt,buf,stride);
}
static void pdfium_bitmap_fill(FPDF_BITMAP b, int l, int t, int w, int h, unsigned long c) {
	static void (*f)(FPDF_BITMAP,int,int,int,int,unsigned long);
	if (!f) f = (void(*)(void*,int,int,int,int,unsigned long))dlsym(g_pdf, "FPDFBitmap_FillRect");
	f(b,l,t,w,h,c);
}
static void pdfium_render(FPDF_BITMAP b, FPDF_PAGE p, int sx, int sy, int w, int h, int rot, int flags) {
	static void (*f)(FPDF_BITMAP,FPDF_PAGE,int,int,int,int,int,int);
	if (!f) f = (void(*)(void*,void*,int,int,int,int,int,int))dlsym(g_pdf, "FPDF_RenderPageBitmap");
	f(b,p,sx,sy,w,h,rot,flags);
}
static void pdfium_bitmap_destroy(FPDF_BITMAP b) {
	static void (*f)(FPDF_BITMAP);
	if (!f) f = (void(*)(void*))dlsym(g_pdf, "FPDFBitmap_Destroy");
	f(b);
}
static void pdfium_closepage(FPDF_PAGE p) {
	static void (*f)(FPDF_PAGE);
	if (!f) f = (void(*)(void*))dlsym(g_pdf, "FPDF_ClosePage");
	f(p);
}
static void pdfium_closedoc(FPDF_DOCUMENT d) {
	static void (*f)(FPDF_DOCUMENT);
	if (!f) f = (void(*)(void*))dlsym(g_pdf, "FPDF_CloseDocument");
	f(d);
}
static int pdfium_pagetodevice(FPDF_PAGE p, int sx, int sy, int w, int h, int rot,
                               double px, double py, int* dx, int* dy) {
	static int (*f)(FPDF_PAGE,int,int,int,int,int,double,double,int*,int*);
	if (!f) f = (int(*)(void*,int,int,int,int,int,double,double,int*,int*))dlsym(g_pdf, "FPDF_PageToDevice");
	return f(p,sx,sy,w,h,rot,px,py,dx,dy);
}

// text page
static FPDF_TEXTPAGE pdfium_text_load(FPDF_PAGE p) {
	static FPDF_TEXTPAGE (*f)(FPDF_PAGE);
	if (!f) f = (void*(*)(void*))dlsym(g_pdf, "FPDFText_LoadPage");
	return f(p);
}
static int pdfium_text_count(FPDF_TEXTPAGE tp) {
	static int (*f)(FPDF_TEXTPAGE);
	if (!f) f = (int(*)(void*))dlsym(g_pdf, "FPDFText_CountChars");
	return f(tp);
}
static unsigned int pdfium_text_unicode(FPDF_TEXTPAGE tp, int i) {
	static unsigned int (*f)(FPDF_TEXTPAGE,int);
	if (!f) f = (unsigned int(*)(void*,int))dlsym(g_pdf, "FPDFText_GetUnicode");
	return f(tp,i);
}
static int pdfium_text_charbox(FPDF_TEXTPAGE tp, int i, double* l, double* r, double* b, double* t) {
	static int (*f)(FPDF_TEXTPAGE,int,double*,double*,double*,double*);
	if (!f) f = (int(*)(void*,int,double*,double*,double*,double*))dlsym(g_pdf, "FPDFText_GetCharBox");
	return f(tp,i,l,r,b,t);
}
static float pdfium_text_charangle(FPDF_TEXTPAGE tp, int i) {
	static float (*f)(FPDF_TEXTPAGE,int);
	if (!f) f = (float(*)(void*,int))dlsym(g_pdf, "FPDFText_GetCharAngle");
	return f(tp,i);
}
static void pdfium_text_close(FPDF_TEXTPAGE tp) {
	static void (*f)(FPDF_TEXTPAGE);
	if (!f) f = (void(*)(void*))dlsym(g_pdf, "FPDFText_ClosePage");
	f(tp);
}

// page objects (invisible-text check)
static int pdfium_page_objects(FPDF_PAGE p) {
	static int (*f)(FPDF_PAGE);
	if (!f) f = (int(*)(void*))dlsym(g_pdf, "FPDFPage_CountObjects");
	return f(p);
}
static FPDF_PAGEOBJECT pdfium_page_object(FPDF_PAGE p, int i) {
	static FPDF_PAGEOBJECT (*f)(FPDF_PAGE,int);
	if (!f) f = (void*(*)(void*,int))dlsym(g_pdf, "FPDFPage_GetObject");
	return f(p,i);
}
static int pdfium_objtype(FPDF_PAGEOBJECT o) {
	static int (*f)(FPDF_PAGEOBJECT);
	if (!f) f = (int(*)(void*))dlsym(g_pdf, "FPDFPageObj_GetType");
	return f(o);
}
static int pdfium_textobj_rendermode(FPDF_PAGEOBJECT o) {
	static int (*f)(FPDF_PAGEOBJECT);
	if (!f) f = (int(*)(void*))dlsym(g_pdf, "FPDFTextObj_GetTextRenderMode");
	return f(o);
}
*/
import "C"

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"syscall"
	"unicode"
	"unsafe"
)

// pdfPageObjText is FPDF_PAGEOBJ_TEXT.
const pdfPageObjText = 1
const pdfTextRenderInvisible = 3

var (
	pdfiumOnce sync.Once
	pdfiumErr  error
)

// loadPDFium loads the embedded libpdfium.so via memfd and opens it. Safe to
// call repeatedly.
func loadPDFium() error {
	pdfiumOnce.Do(func() {
		if sysMemfdCreate < 0 {
			pdfiumErr = fmt.Errorf("indra: pdfium needs linux/amd64; not available")
			return
		}
		data, err := libFS.ReadFile("lib/libpdfium.so")
		if err != nil {
			pdfiumErr = fmt.Errorf("indra: embedded libpdfium: %w", err)
			return
		}
		name := append([]byte("libpdfium.so"), 0)
		fd, _, errno := syscall.Syscall(uintptr(sysMemfdCreate),
			uintptr(unsafe.Pointer(&name[0])), 0x0001 /* MFD_CLOEXEC */, 0)
		if errno != 0 {
			pdfiumErr = fmt.Errorf("indra: memfd libpdfium: %w", errno)
			return
		}
		if err := writeAll(int(fd), data); err != nil {
			pdfiumErr = err
			return
		}
		memfdFDs = append(memfdFDs, fd)
		cpath := C.CString("/proc/self/fd/" + itoa(int(fd)))
		defer C.free(unsafe.Pointer(cpath))
		if C.pdfium_open(cpath) != 0 {
			pdfiumErr = fmt.Errorf("indra: dlopen libpdfium failed")
			return
		}
		C.pdfium_initlib()
	})
	return pdfiumErr
}

// --- Go wrappers over the C shims (only this cgo file can see them). ---

// pdfOpen loads a PDF from memory and returns (document, buffer) — the caller
// must pdfClose both. FPDF_LoadMemDocument64 does NOT copy the bytes: pdfium
// reads them for the document's lifetime, and handing it a Go heap pointer
// violates the cgo pointer rules (the GC may free or move it while pages are
// still being rendered). So the bytes are copied into C memory here and freed by
// pdfClose.
func pdfOpen(data []byte) (unsafe.Pointer, unsafe.Pointer, error) {
	if len(data) == 0 {
		return nil, nil, fmt.Errorf("indra: empty pdf")
	}
	buf := C.CBytes(data)
	cs := C.CString("")
	defer C.free(unsafe.Pointer(cs))
	doc := unsafe.Pointer(C.pdfium_loadmem(buf, C.size_t(len(data)), cs))
	if doc == nil {
		C.free(buf)
		return nil, nil, fmt.Errorf("indra: pdfium could not open the document")
	}
	return doc, buf, nil
}

func pdfClose(doc, buf unsafe.Pointer) {
	if doc != nil {
		C.pdfium_closedoc(C.FPDF_DOCUMENT(doc))
	}
	if buf != nil {
		C.free(buf)
	}
}
func pdfPageCount(doc unsafe.Pointer) int {
	return int(C.pdfium_pagecount(C.FPDF_DOCUMENT(doc)))
}

// pdfRenderPage renders page index at scale (px/pt) to grayscale, returning the
// image and the page size in points.
func pdfRenderPage(doc unsafe.Pointer, index int, scale float64) (*Gray, float64, float64, bool) {
	page := C.pdfium_loadpage(C.FPDF_DOCUMENT(doc), C.int(index))
	if page == nil {
		return nil, 0, 0, false
	}
	defer C.pdfium_closepage(page)
	pw := float64(C.pdfium_pagewidth(page))
	ph := float64(C.pdfium_pageheight(page))
	w := int(pw*scale + 0.5)
	h := int(ph*scale + 0.5)
	if w < 1 || h < 1 {
		return nil, pw, ph, false
	}
	buf := make([]byte, w*h*4)
	bmp := C.pdfium_bitmap(C.int(w), C.int(h), 4 /*BGRA*/, unsafe.Pointer(&buf[0]), C.int(w*4))
	if bmp == nil {
		return nil, pw, ph, false
	}
	defer C.pdfium_bitmap_destroy(bmp)
	C.pdfium_bitmap_fill(bmp, 0, 0, C.int(w), C.int(h), 0xFFFFFFFF)
	C.pdfium_render(bmp, page, 0, 0, C.int(w), C.int(h), 0, 0)
	px := make([]uint8, w*h)
	for i := 0; i < w*h; i++ {
		px[i] = uint8((299*int(buf[i*4+2]) + 587*int(buf[i*4+1]) + 114*int(buf[i*4])) / 1000)
	}
	return &Gray{Pix: px, W: w, H: h}, pw, ph, true
}

// pdfDeviceAffine maps PDF page space (points) to the rendered frame (top-left
// origin, points), mirroring pypdfium2's FPDF_PageToDevice use in textlayer.py.
// Returns [A00 A01 A02 A10 A11 A12]: xf = A00*x + A01*y + A02, yf = A10*x + A11*y + A12.
func pdfDeviceAffine(page C.FPDF_PAGE, w, h float64) [6]float64 {
	const s = 1000.0
	dev := func(x, y float64) (float64, float64) {
		var dx, dy C.int
		C.pdfium_pagetodevice(page, 0, 0, C.int(w*s+0.5), C.int(h*s+0.5), 0,
			C.double(x), C.double(y), &dx, &dy)
		return float64(dx) / s, float64(dy) / s
	}
	ox, oy := dev(0, 0)
	exx, exy := dev(1, 0)
	eyx, eyy := dev(0, 1)
	return [6]float64{exx - ox, eyx - ox, ox, exy - oy, eyy - oy, oy}
}

func pdfInvisibleFrac(page C.FPDF_PAGE) float64 {
	n := int(C.pdfium_page_objects(page))
	if n == 0 {
		return 0
	}
	inv := 0
	for i := 0; i < n; i++ {
		obj := C.pdfium_page_object(page, C.int(i))
		if obj == nil {
			continue
		}
		if int(C.pdfium_objtype(obj)) == pdfPageObjText &&
			int(C.pdfium_textobj_rendermode(obj)) == pdfTextRenderInvisible {
			inv++
		}
	}
	return float64(inv) / float64(n)
}

// pdfProcess renders page index at scale and extracts its text-layer runs (boxes
// in rendered pixels), the character stats and the invisible-text fraction. It
// mirrors textlayer.extract_runs / read_text_layer.
func pdfProcess(doc unsafe.Pointer, index int, scale float64) (*Gray, []TextRun, int, int, float64, bool) {
	page := C.pdfium_loadpage(C.FPDF_DOCUMENT(doc), C.int(index))
	if page == nil {
		return nil, nil, 0, 0, 0, false
	}
	defer C.pdfium_closepage(page)
	pw := float64(C.pdfium_pagewidth(page))
	ph := float64(C.pdfium_pageheight(page))
	w := int(pw*scale + 0.5)
	h := int(ph*scale + 0.5)
	if w < 1 || h < 1 {
		return nil, nil, 0, 0, 0, false
	}
	buf := make([]byte, w*h*4)
	bmp := C.pdfium_bitmap(C.int(w), C.int(h), 4, unsafe.Pointer(&buf[0]), C.int(w*4))
	if bmp == nil {
		return nil, nil, 0, 0, 0, false
	}
	C.pdfium_bitmap_fill(bmp, 0, 0, C.int(w), C.int(h), 0xFFFFFFFF)
	C.pdfium_render(bmp, page, 0, 0, C.int(w), C.int(h), 0, 0)
	C.pdfium_bitmap_destroy(bmp)
	px := make([]uint8, w*h)
	for i := 0; i < w*h; i++ {
		px[i] = uint8((299*int(buf[i*4+2]) + 587*int(buf[i*4+1]) + 114*int(buf[i*4])) / 1000)
	}
	g := &Gray{Pix: px, W: w, H: h}

	invisible := pdfInvisibleFrac(page)
	A := pdfDeviceAffine(page, pw, ph)
	tp := C.pdfium_text_load(page)
	if tp == nil {
		return g, nil, 0, 0, invisible, true
	}
	defer C.pdfium_text_close(tp)
	nc := int(C.pdfium_text_count(tp))
	var cs []TextChar
	var breaks []int
	nChars, nRot, nBad := 0, 0, 0
	for i := 0; i < nc; i++ {
		c := rune(C.pdfium_text_unicode(tp, C.int(i)))
		if c == '\r' || c == '\n' {
			breaks = append(breaks, len(cs))
			continue
		}
		if c == 0x02 {
			c = '-'
		}
		if !unicode.IsSpace(c) {
			nChars++
			nBad += badCharCount(string(c))
			a := float64(C.pdfium_text_charangle(tp, C.int(i)))
			d0 := A[0]*math.Cos(a) + A[1]*math.Sin(a)
			d1 := A[3]*math.Cos(a) + A[4]*math.Sin(a)
			if !(d0 > 0 && math.Abs(d1) <= 1e-3*math.Max(math.Abs(d0), 1e-9)) {
				nRot++
				breaks = append(breaks, len(cs))
				continue
			}
		}
		var bl, br, bb, bt C.double
		C.pdfium_text_charbox(tp, C.int(i), &bl, &br, &bb, &bt)
		// map the 4 corners of (l,b,r,t) through the affine, then to pixels
		var xs = [4]float64{float64(bl), float64(br), float64(bl), float64(br)}
		var ys = [4]float64{float64(bb), float64(bb), float64(bt), float64(bt)}
		minx, miny := math.Inf(1), math.Inf(1)
		maxx, maxy := math.Inf(-1), math.Inf(-1)
		for k := 0; k < 4; k++ {
			fx := A[0]*xs[k] + A[1]*ys[k] + A[2]
			fy := A[3]*xs[k] + A[4]*ys[k] + A[5]
			minx, maxx = math.Min(minx, fx), math.Max(maxx, fx)
			miny, maxy = math.Min(miny, fy), math.Max(maxy, fy)
		}
		box := [4]float64{minx * scale, miny * scale, maxx * scale, maxy * scale}
		// off-page (beyond the crop box) characters are dropped
		cx, cy := (minx+maxx)/2, (miny+maxy)/2
		if cx < 0 || cx > pw || cy < 0 || cy > ph {
			continue
		}
		cs = append(cs, TextChar{Ch: c, Box: box})
	}
	// cut into runs at the breaks
	cuts := append(append([]int(nil), breaks...), len(cs))
	sort.Ints(cuts)
	var runs []TextRun
	start := 0
	for _, end := range cuts {
		if end > len(cs) {
			end = len(cs)
		}
		if end < start {
			continue
		}
		seg := cs[start:end]
		start = end
		seg = foldSpacingAcute(seg)
		var ink []TextChar
		for _, c := range seg {
			if !unicode.IsSpace(c.Ch) {
				ink = append(ink, c)
			}
		}
		if len(ink) == 0 {
			continue
		}
		box := [4]float64{math.Inf(1), math.Inf(1), math.Inf(-1), math.Inf(-1)}
		var sb strings.Builder
		var chars []TextChar
		for _, c := range seg {
			sb.WriteRune(c.Ch)
			chars = append(chars, c)
		}
		for _, c := range ink {
			box[0] = math.Min(box[0], c.Box[0])
			box[1] = math.Min(box[1], c.Box[1])
			box[2] = math.Max(box[2], c.Box[2])
			box[3] = math.Max(box[3], c.Box[3])
		}
		runs = append(runs, TextRun{Text: strings.Join(strings.Fields(sb.String()), " "), Box: box, Chars: chars})
	}
	return g, runs, nChars, nRot, invisible, true
}

// acuteCompose maps a base letter to its precomposed acute form.
var acuteCompose = map[rune]rune{
	'a': 'á', 'A': 'Á', 'c': 'ć', 'C': 'Ć', 'e': 'é', 'E': 'É', 'i': 'í', 'I': 'Í',
	'l': 'ĺ', 'L': 'Ĺ', 'n': 'ń', 'N': 'Ń', 'o': 'ó', 'O': 'Ó', 'r': 'ŕ', 'R': 'Ŕ',
	's': 'ś', 'S': 'Ś', 'u': 'ú', 'U': 'Ú', 'y': 'ý', 'Y': 'Ý', 'z': 'ź', 'Z': 'Ź',
	'g': 'ǵ', 'G': 'Ǵ', 'w': 'ẃ', 'W': 'Ẃ', 'æ': 'ǽ', 'Æ': 'Ǽ',
	'ı': 'í', // dotless i (Turkish/legacy) before an acute
}

// foldSpacingAcute composes a spacing acute (U+00B4) that PDFs place just before
// its base letter: pdfium emits the accent glyph in content order, so "Cicho´n"
// comes out where the recognizer (whose vocab is NFC) gives "Cichoń". This
// deviates from upstream indra, which does not normalize the text layer — see
// docs/DEVIATIONS.md.
//
// The acute is composed only when it sits between two letters (prev letter,
// next composable letter), which is every real accent case here; a U+00B4 after
// a non-letter (or before punctuation) is a prime/symbol and is left as-is, e.g.
// "SplExtp´," or "r´, ´s".
func foldSpacingAcute(seg []TextChar) []TextChar {
	out := make([]TextChar, 0, len(seg))
	for i := 0; i < len(seg); i++ {
		if seg[i].Ch == '\u00b4' && i > 0 && i+1 < len(seg) &&
			unicode.IsLetter(seg[i-1].Ch) {
			if p, ok := acuteCompose[seg[i+1].Ch]; ok {
				nx := seg[i+1]
				nx.Ch = p
				out = append(out, nx)
				i++
				continue
			}
		}
		out = append(out, seg[i])
	}
	return out
}
