package indra

// cgo bridge to the embedded libpdfium.so. The library is loaded from an
// anonymous memfd at runtime and every entry point is resolved with dlsym, so
// nothing links against pdfium at build time and nothing is installed.
//
// PDFium's C API is not thread-safe (the API header allows only one call in
// flight), so the pool hands each concurrent CallPDF its own loaded copy of the
// library: separate copies have separate globals and render in parallel.

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

// pdfium_api is one loaded copy of libpdfium: the dlopen handle plus every
// entry point this file calls. PDFium's C API is not thread-safe — the API
// header states that only one call may be in flight at a time — but its
// process-wide globals live in the *loaded library image*, so two copies of the
// same .so are two independent PDFium instances. The Go side lends one instance
// per concurrent document (pdfiumInst), which is what makes concurrent
// rendering safe without serializing it.
typedef struct pdfium_api {
	void* lib;
	void (*InitLibrary)(void);
	FPDF_DOCUMENT (*LoadMemDocument64)(const void*, size_t, const char*);
	int (*GetPageCount)(FPDF_DOCUMENT);
	FPDF_PAGE (*LoadPage)(FPDF_DOCUMENT, int);
	float (*GetPageWidthF)(FPDF_PAGE);
	float (*GetPageHeightF)(FPDF_PAGE);
	FPDF_BITMAP (*Bitmap_CreateEx)(int, int, int, void*, int);
	void (*Bitmap_FillRect)(FPDF_BITMAP, int, int, int, int, unsigned long);
	void (*RenderPageBitmap)(FPDF_BITMAP, FPDF_PAGE, int, int, int, int, int, int);
	void (*Bitmap_Destroy)(FPDF_BITMAP);
	void (*ClosePage)(FPDF_PAGE);
	void (*CloseDocument)(FPDF_DOCUMENT);
	int (*PageToDevice)(FPDF_PAGE, int, int, int, int, int, double, double, int*, int*);
	FPDF_TEXTPAGE (*Text_LoadPage)(FPDF_PAGE);
	int (*Text_CountChars)(FPDF_TEXTPAGE);
	unsigned int (*Text_GetUnicode)(FPDF_TEXTPAGE, int);
	int (*Text_GetCharBox)(FPDF_TEXTPAGE, int, double*, double*, double*, double*);
	float (*Text_GetCharAngle)(FPDF_TEXTPAGE, int);
	void (*Text_ClosePage)(FPDF_TEXTPAGE);
	int (*Page_CountObjects)(FPDF_PAGE);
	FPDF_PAGEOBJECT (*Page_GetObject)(FPDF_PAGE, int);
	int (*PageObj_GetType)(FPDF_PAGEOBJECT);
	int (*TextObj_GetTextRenderMode)(FPDF_PAGEOBJECT);
} pdfium_api;

#define PDFIUM_SYM(api, field, name)                    \
	do {                                                \
		*(void**)(&(api)->field) = dlsym((api)->lib, name); \
		if (!(api)->field) {                            \
			dlclose((api)->lib);                        \
			free(api);                                  \
			return 0;                                   \
		}                                               \
	} while (0)

// pdfium_api_open opens one copy of libpdfium at path, resolves every entry
// point and initializes it, returning NULL on any failure. RTLD_LOCAL keeps
// each copy's symbols out of the global namespace, so several copies coexist
// as separate instances. dlsym resolves against the copy itself, so calls
// through one instance never reach another's globals.
static pdfium_api* pdfium_api_open(const char* path) {
	void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	if (!lib) return 0;
	pdfium_api* api = (pdfium_api*)calloc(1, sizeof(pdfium_api));
	if (!api) { dlclose(lib); return 0; }
	api->lib = lib;
	PDFIUM_SYM(api, InitLibrary, "FPDF_InitLibrary");
	PDFIUM_SYM(api, LoadMemDocument64, "FPDF_LoadMemDocument64");
	PDFIUM_SYM(api, GetPageCount, "FPDF_GetPageCount");
	PDFIUM_SYM(api, LoadPage, "FPDF_LoadPage");
	PDFIUM_SYM(api, GetPageWidthF, "FPDF_GetPageWidthF");
	PDFIUM_SYM(api, GetPageHeightF, "FPDF_GetPageHeightF");
	PDFIUM_SYM(api, Bitmap_CreateEx, "FPDFBitmap_CreateEx");
	PDFIUM_SYM(api, Bitmap_FillRect, "FPDFBitmap_FillRect");
	PDFIUM_SYM(api, RenderPageBitmap, "FPDF_RenderPageBitmap");
	PDFIUM_SYM(api, Bitmap_Destroy, "FPDFBitmap_Destroy");
	PDFIUM_SYM(api, ClosePage, "FPDF_ClosePage");
	PDFIUM_SYM(api, CloseDocument, "FPDF_CloseDocument");
	PDFIUM_SYM(api, PageToDevice, "FPDF_PageToDevice");
	PDFIUM_SYM(api, Text_LoadPage, "FPDFText_LoadPage");
	PDFIUM_SYM(api, Text_CountChars, "FPDFText_CountChars");
	PDFIUM_SYM(api, Text_GetUnicode, "FPDFText_GetUnicode");
	PDFIUM_SYM(api, Text_GetCharBox, "FPDFText_GetCharBox");
	PDFIUM_SYM(api, Text_GetCharAngle, "FPDFText_GetCharAngle");
	PDFIUM_SYM(api, Text_ClosePage, "FPDFText_ClosePage");
	PDFIUM_SYM(api, Page_CountObjects, "FPDFPage_CountObjects");
	PDFIUM_SYM(api, Page_GetObject, "FPDFPage_GetObject");
	PDFIUM_SYM(api, PageObj_GetType, "FPDFPageObj_GetType");
	PDFIUM_SYM(api, TextObj_GetTextRenderMode, "FPDFTextObj_GetTextRenderMode");
	api->InitLibrary();
	return api;
}

static FPDF_DOCUMENT pdfium_loadmem(pdfium_api* api, const void* p, size_t n, const char* pw) {
	return api->LoadMemDocument64(p, n, pw);
}
static int pdfium_pagecount(pdfium_api* api, FPDF_DOCUMENT d) {
	return api->GetPageCount(d);
}
static FPDF_PAGE pdfium_loadpage(pdfium_api* api, FPDF_DOCUMENT d, int i) {
	return api->LoadPage(d, i);
}
static float pdfium_pagewidth(pdfium_api* api, FPDF_PAGE p) {
	return api->GetPageWidthF(p);
}
static float pdfium_pageheight(pdfium_api* api, FPDF_PAGE p) {
	return api->GetPageHeightF(p);
}
static FPDF_BITMAP pdfium_bitmap(pdfium_api* api, int w, int h, int fmt, void* buf, int stride) {
	return api->Bitmap_CreateEx(w, h, fmt, buf, stride);
}
static void pdfium_bitmap_fill(pdfium_api* api, FPDF_BITMAP b, int l, int t, int w, int h, unsigned long c) {
	api->Bitmap_FillRect(b, l, t, w, h, c);
}
static void pdfium_render(pdfium_api* api, FPDF_BITMAP b, FPDF_PAGE p, int sx, int sy, int w, int h, int rot, int flags) {
	api->RenderPageBitmap(b, p, sx, sy, w, h, rot, flags);
}
static void pdfium_bitmap_destroy(pdfium_api* api, FPDF_BITMAP b) {
	api->Bitmap_Destroy(b);
}
static void pdfium_closepage(pdfium_api* api, FPDF_PAGE p) {
	api->ClosePage(p);
}
static void pdfium_closedoc(pdfium_api* api, FPDF_DOCUMENT d) {
	api->CloseDocument(d);
}
static int pdfium_pagetodevice(pdfium_api* api, FPDF_PAGE p, int sx, int sy, int w, int h, int rot,
                               double px, double py, int* dx, int* dy) {
	return api->PageToDevice(p, sx, sy, w, h, rot, px, py, dx, dy);
}

// text page
static FPDF_TEXTPAGE pdfium_text_load(pdfium_api* api, FPDF_PAGE p) {
	return api->Text_LoadPage(p);
}
static int pdfium_text_count(pdfium_api* api, FPDF_TEXTPAGE tp) {
	return api->Text_CountChars(tp);
}
static unsigned int pdfium_text_unicode(pdfium_api* api, FPDF_TEXTPAGE tp, int i) {
	return api->Text_GetUnicode(tp, i);
}
static int pdfium_text_charbox(pdfium_api* api, FPDF_TEXTPAGE tp, int i, double* l, double* r, double* b, double* t) {
	return api->Text_GetCharBox(tp, i, l, r, b, t);
}
static float pdfium_text_charangle(pdfium_api* api, FPDF_TEXTPAGE tp, int i) {
	return api->Text_GetCharAngle(tp, i);
}
static void pdfium_text_close(pdfium_api* api, FPDF_TEXTPAGE tp) {
	api->Text_ClosePage(tp);
}

// page objects (invisible-text check)
static int pdfium_page_objects(pdfium_api* api, FPDF_PAGE p) {
	return api->Page_CountObjects(p);
}
static FPDF_PAGEOBJECT pdfium_page_object(pdfium_api* api, FPDF_PAGE p, int i) {
	return api->Page_GetObject(p, i);
}
static int pdfium_objtype(pdfium_api* api, FPDF_PAGEOBJECT o) {
	return api->PageObj_GetType(o);
}
static int pdfium_textobj_rendermode(pdfium_api* api, FPDF_PAGEOBJECT o) {
	return api->TextObj_GetTextRenderMode(o);
}
*/
import "C"

import (
	"fmt"
	"math"
	"runtime"
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
	pdfiumBytesOnce sync.Once
	pdfiumBytes     []byte
	pdfiumBytesErr  error
)

// pdfiumLibBytes reads the embedded libpdfium.so once.
func pdfiumLibBytes() ([]byte, error) {
	pdfiumBytesOnce.Do(func() {
		pdfiumBytes, pdfiumBytesErr = libFS.ReadFile("lib/libpdfium.so")
	})
	if pdfiumBytesErr != nil {
		return nil, fmt.Errorf("indra: embedded libpdfium: %w", pdfiumBytesErr)
	}
	return pdfiumBytes, nil
}

// pdfiumInst is one loaded copy of libpdfium, with exclusive use until it is
// released back to the pool. PDFium is not thread-safe: no entry point may run
// concurrently with another on the same instance. Its globals belong to the
// loaded library image, so separate copies — a separate memfd and dlopen each —
// are independent PDFium instances, and renditions on distinct instances run in
// parallel without sharing state.
type pdfiumInst struct {
	api *C.pdfium_api
}

// pdfiumMaxInstances caps how many copies coexist. Rendering is only a small
// share of a PDF page's cost (the recognizer dominates), so a handful of
// renderers already feed every worker; each copy also costs a library mapping
// and its own caches, so one per worker would only add memory.
func pdfiumMaxInstances() int {
	n := runtime.NumCPU()
	if n > 8 {
		n = 8
	}
	if n < 1 {
		n = 1
	}
	return n
}

// pdfiumPool hands each concurrent CallPDF its own pdfiumInst. Instances are
// opened lazily and only while fewer than pdfiumMaxInstances exist and none is
// idle, so a single-threaded caller opens just one.
type pdfiumPool struct {
	mu   sync.Mutex
	cond *sync.Cond
	idle []*pdfiumInst
	n    int
}

var pdfiumInsts = newPDFiumPool()

func newPDFiumPool() *pdfiumPool {
	p := &pdfiumPool{}
	p.cond = sync.NewCond(&p.mu)
	return p
}

// acquire returns an instance with exclusive use until release.
func (p *pdfiumPool) acquire() (*pdfiumInst, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for {
		if n := len(p.idle); n > 0 {
			inst := p.idle[n-1]
			p.idle = p.idle[:n-1]
			return inst, nil
		}
		if p.n < pdfiumMaxInstances() {
			p.n++ // reserve the slot while the copy is opened
			p.mu.Unlock()
			inst, err := newPDFiumInst()
			p.mu.Lock()
			if err != nil {
				p.n--
				p.cond.Broadcast()
				return nil, err
			}
			return inst, nil
		}
		p.cond.Wait()
	}
}

func (p *pdfiumPool) release(inst *pdfiumInst) {
	p.mu.Lock()
	p.idle = append(p.idle, inst)
	p.mu.Unlock()
	p.cond.Signal()
}

// newPDFiumInst opens one independent copy of libpdfium from an anonymous
// memfd. dlopen deduplicates a library by its inode, so every instance gets its
// own copy of the bytes — that private copy is what isolates the instances'
// globals. The fd stays open for the process lifetime (like memfdFDs): the
// mapping keeps the bytes alive.
func newPDFiumInst() (*pdfiumInst, error) {
	if sysMemfdCreate < 0 {
		return nil, fmt.Errorf("indra: pdfium needs linux/amd64; not available")
	}
	data, err := pdfiumLibBytes()
	if err != nil {
		return nil, err
	}
	name := append([]byte("libpdfium.so"), 0)
	fd, _, errno := syscall.Syscall(uintptr(sysMemfdCreate),
		uintptr(unsafe.Pointer(&name[0])), 0x0001 /* MFD_CLOEXEC */, 0)
	if errno != 0 {
		return nil, fmt.Errorf("indra: memfd libpdfium: %w", errno)
	}
	if err := writeAll(int(fd), data); err != nil {
		syscall.Close(int(fd))
		return nil, err
	}
	cpath := C.CString("/proc/self/fd/" + itoa(int(fd)))
	defer C.free(unsafe.Pointer(cpath))
	api := C.pdfium_api_open(cpath)
	if api == nil {
		syscall.Close(int(fd))
		return nil, fmt.Errorf("indra: dlopen libpdfium failed")
	}
	return &pdfiumInst{api: api}, nil
}

// --- Go wrappers over the C shims (only this cgo file can see them). ---

// open loads a PDF from memory and returns (document, buffer) — the caller
// must close both. FPDF_LoadMemDocument64 does NOT copy the bytes: pdfium
// reads them for the document's lifetime, and handing it a Go heap pointer
// violates the cgo pointer rules (the GC may free or move it while pages are
// still being rendered). So the bytes are copied into C memory here and freed by
// close.
func (inst *pdfiumInst) open(data []byte) (unsafe.Pointer, unsafe.Pointer, error) {
	if len(data) == 0 {
		return nil, nil, fmt.Errorf("indra: empty pdf")
	}
	buf := C.CBytes(data)
	cs := C.CString("")
	defer C.free(unsafe.Pointer(cs))
	doc := unsafe.Pointer(C.pdfium_loadmem(inst.api, buf, C.size_t(len(data)), cs))
	if doc == nil {
		C.free(buf)
		return nil, nil, fmt.Errorf("indra: pdfium could not open the document")
	}
	return doc, buf, nil
}

func (inst *pdfiumInst) close(doc, buf unsafe.Pointer) {
	if doc != nil {
		C.pdfium_closedoc(inst.api, C.FPDF_DOCUMENT(doc))
	}
	if buf != nil {
		C.free(buf)
	}
}

func (inst *pdfiumInst) pageCount(doc unsafe.Pointer) int {
	return int(C.pdfium_pagecount(inst.api, C.FPDF_DOCUMENT(doc)))
}

// renderPage renders page index at scale (px/pt) to grayscale, returning the
// image and the page size in points.
func (inst *pdfiumInst) renderPage(doc unsafe.Pointer, index int, scale float64) (*Gray, float64, float64, bool) {
	page := C.pdfium_loadpage(inst.api, C.FPDF_DOCUMENT(doc), C.int(index))
	if page == nil {
		return nil, 0, 0, false
	}
	defer C.pdfium_closepage(inst.api, page)
	pw := float64(C.pdfium_pagewidth(inst.api, page))
	ph := float64(C.pdfium_pageheight(inst.api, page))
	w := int(pw*scale + 0.5)
	h := int(ph*scale + 0.5)
	if w < 1 || h < 1 {
		return nil, pw, ph, false
	}
	buf := make([]byte, w*h*4)
	bmp := C.pdfium_bitmap(inst.api, C.int(w), C.int(h), 4 /*BGRA*/, unsafe.Pointer(&buf[0]), C.int(w*4))
	if bmp == nil {
		return nil, pw, ph, false
	}
	defer C.pdfium_bitmap_destroy(inst.api, bmp)
	C.pdfium_bitmap_fill(inst.api, bmp, 0, 0, C.int(w), C.int(h), 0xFFFFFFFF)
	C.pdfium_render(inst.api, bmp, page, 0, 0, C.int(w), C.int(h), 0, 0)
	px := make([]uint8, w*h)
	for i := 0; i < w*h; i++ {
		px[i] = uint8((299*int(buf[i*4+2]) + 587*int(buf[i*4+1]) + 114*int(buf[i*4])) / 1000)
	}
	return &Gray{Pix: px, W: w, H: h}, pw, ph, true
}

// deviceAffine maps PDF page space (points) to the rendered frame (top-left
// origin, points), mirroring pypdfium2's FPDF_PageToDevice use in textlayer.py.
// Returns [A00 A01 A02 A10 A11 A12]: xf = A00*x + A01*y + A02, yf = A10*x + A11*y + A12.
func (inst *pdfiumInst) deviceAffine(page C.FPDF_PAGE, w, h float64) [6]float64 {
	const s = 1000.0
	dev := func(x, y float64) (float64, float64) {
		var dx, dy C.int
		C.pdfium_pagetodevice(inst.api, page, 0, 0, C.int(w*s+0.5), C.int(h*s+0.5), 0,
			C.double(x), C.double(y), &dx, &dy)
		return float64(dx) / s, float64(dy) / s
	}
	ox, oy := dev(0, 0)
	exx, exy := dev(1, 0)
	eyx, eyy := dev(0, 1)
	return [6]float64{exx - ox, eyx - ox, ox, exy - oy, eyy - oy, oy}
}

func (inst *pdfiumInst) invisibleFrac(page C.FPDF_PAGE) float64 {
	n := int(C.pdfium_page_objects(inst.api, page))
	if n == 0 {
		return 0
	}
	inv := 0
	for i := 0; i < n; i++ {
		obj := C.pdfium_page_object(inst.api, page, C.int(i))
		if obj == nil {
			continue
		}
		if int(C.pdfium_objtype(inst.api, obj)) == pdfPageObjText &&
			int(C.pdfium_textobj_rendermode(inst.api, obj)) == pdfTextRenderInvisible {
			inv++
		}
	}
	return float64(inv) / float64(n)
}

// process renders page index at scale and extracts its text-layer runs (boxes
// in rendered pixels), the character stats and the invisible-text fraction. It
// mirrors textlayer.extract_runs / read_text_layer.
func (inst *pdfiumInst) process(doc unsafe.Pointer, index int, scale float64) (*Gray, []TextRun, int, int, float64, bool) {
	page := C.pdfium_loadpage(inst.api, C.FPDF_DOCUMENT(doc), C.int(index))
	if page == nil {
		return nil, nil, 0, 0, 0, false
	}
	defer C.pdfium_closepage(inst.api, page)
	pw := float64(C.pdfium_pagewidth(inst.api, page))
	ph := float64(C.pdfium_pageheight(inst.api, page))
	w := int(pw*scale + 0.5)
	h := int(ph*scale + 0.5)
	if w < 1 || h < 1 {
		return nil, nil, 0, 0, 0, false
	}
	buf := make([]byte, w*h*4)
	bmp := C.pdfium_bitmap(inst.api, C.int(w), C.int(h), 4, unsafe.Pointer(&buf[0]), C.int(w*4))
	if bmp == nil {
		return nil, nil, 0, 0, 0, false
	}
	C.pdfium_bitmap_fill(inst.api, bmp, 0, 0, C.int(w), C.int(h), 0xFFFFFFFF)
	C.pdfium_render(inst.api, bmp, page, 0, 0, C.int(w), C.int(h), 0, 0)
	C.pdfium_bitmap_destroy(inst.api, bmp)
	px := make([]uint8, w*h)
	for i := 0; i < w*h; i++ {
		px[i] = uint8((299*int(buf[i*4+2]) + 587*int(buf[i*4+1]) + 114*int(buf[i*4])) / 1000)
	}
	g := &Gray{Pix: px, W: w, H: h}

	invisible := inst.invisibleFrac(page)
	A := inst.deviceAffine(page, pw, ph)
	tp := C.pdfium_text_load(inst.api, page)
	if tp == nil {
		return g, nil, 0, 0, invisible, true
	}
	defer C.pdfium_text_close(inst.api, tp)
	nc := int(C.pdfium_text_count(inst.api, tp))
	var cs []TextChar
	var breaks []int
	nChars, nRot, nBad := 0, 0, 0
	for i := 0; i < nc; i++ {
		c := rune(C.pdfium_text_unicode(inst.api, tp, C.int(i)))
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
			a := float64(C.pdfium_text_charangle(inst.api, tp, C.int(i)))
			d0 := A[0]*math.Cos(a) + A[1]*math.Sin(a)
			d1 := A[3]*math.Cos(a) + A[4]*math.Sin(a)
			if !(d0 > 0 && math.Abs(d1) <= 1e-3*math.Max(math.Abs(d0), 1e-9)) {
				nRot++
				breaks = append(breaks, len(cs))
				continue
			}
		}
		var bl, br, bb, bt C.double
		C.pdfium_text_charbox(inst.api, tp, C.int(i), &bl, &br, &bb, &bt)
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
		seg = foldSpacingAccents(seg)
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

// foldSpacingAccents composes a spacing diacritic that a PDF text layer places
// next to its base letter. pdfium emits the accent glyph as a separate character
// in content order, before or after the base depending on the PDF producer, so
// "Cicho´n" (before) and "Schu¨tz" (after) both come out where the recognizer —
// whose vocab is NFC — gives "Cichoń" / "Schütz". It is a deliberate deviation
// from upstream indra, which does not normalize the text layer; see
// docs/DEVIATIONS.md.
//
// The spacing glyph is looked up in spacingMarks and composed through the
// generated composeMark table (Unicode canonical composition), so every base a
// mark can join is handled, and a double-marked letter (Vietnamese) builds up by
// successive composition.
//
// Orientation: a prefix mark is drawn over/next to the *next* glyph, a postfix
// mark over the *previous* one, and the text layer carries per-character boxes,
// so the mark is joined to whichever neighbour its box overlaps more. A mark
// whose neighbour has no precomposed form — a math operator, a unit prime after
// a number, an unrecomposable base — is left as a spacing character. ASCII `^`
// and `~` are not spacing marks, so equations are safe.
func foldSpacingAccents(seg []TextChar) []TextChar {
	out := make([]TextChar, 0, len(seg))
	for i := 0; i < len(seg); i++ {
		mark, ok := spacingMarks[seg[i].Ch]
		if !ok {
			out = append(out, seg[i])
			continue
		}
		pi := len(out) - 1
		hasPrev := pi >= 0
		hasNext := i+1 < len(seg)
		fwd := hasNext && composable(mark, seg[i+1].Ch)
		if fwd && hasPrev && unicode.IsNumber(out[pi].Ch) {
			fwd = false // a prime attached to a number ("5´") is a unit symbol
		}
		bwd := hasPrev && composable(mark, out[pi].Ch)
		// A postfix accent is only folded onto the previous character when its
		// glyph actually sits over it: a prime or other trailing symbol is offset
		// (or has no box), so "SplExtp´," keeps its prime.
		postfix := false
		if hasPrev && bwd {
			if hasNext {
				postfix = overlapX(seg[i].Box, out[pi].Box) > overlapX(seg[i].Box, seg[i+1].Box)
			} else {
				postfix = overlapX(seg[i].Box, out[pi].Box) > 0
			}
		}
		switch {
		case postfix:
			out[pi].Ch = composeMark[mark][out[pi].Ch]
		case fwd:
			nx := seg[i+1]
			nx.Ch = composeMark[mark][nx.Ch]
			out = append(out, nx)
			i++
		case hasPrev && bwd && hasNext && isSpacingMark(seg[i+1].Ch):
			// a run of marks over one base (stacked Vietnamese): the next glyph is
			// another accent, so this one belongs to the base before it
			out[pi].Ch = composeMark[mark][out[pi].Ch]
		default:
			out = append(out, seg[i])
		}
	}
	return out
}

func composable(mark, base rune) bool {
	_, ok := composeMark[mark][base]
	return ok
}

func isSpacingMark(r rune) bool {
	_, ok := spacingMarks[r]
	return ok
}

func overlapX(a, b [4]float64) float64 {
	lo := math.Max(a[0], b[0])
	hi := math.Min(a[2], b[2])
	if hi > lo {
		return hi - lo
	}
	return 0
}
