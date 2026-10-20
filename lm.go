// Beam search with a token n-gram language model, for lines the recognizer
// is unsure of. Port of lm.py.
//
// The recognizer reads greedily: one best class per step. On a line whose
// confidence is under the gate, a CTC prefix beam search keeps several
// readings and the language model breaks close calls between them:
// score = log P_ctc + alpha * log P_lm + beta * tokens. The recognizer's own
// probabilities still decide, so a clearly written odd word survives.
//
// lm.npz holds the model as a backoff table (no pickle): per history length
// n, sorted keys of the (history, token) pairs seen in training with their
// log-probabilities, and sorted history keys with their backoff weights. A
// key packs tokens `bits` bits each (0 = line start, t + 1 = token t).
package indra

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"sync"
)

const (
	lmDefaultBits = 11 // per token, unless lm.npz holds "bits"
	lmTagShift    = 61  // the history length goes in the top 3 bits
	lmBOS         = -1
)

// lmPack packs tokens into one key: 0 for BOS, t + 1 for token t.
func lmPack(tokens []int, bits int) uint64 {
	var k uint64
	for _, t := range tokens {
		v := uint64(0)
		if t != lmBOS {
			v = uint64(t + 1)
		}
		k = (k << uint(bits)) | v
	}
	return k
}

// npyFile is one decoded .npy member of an .npz (zip) archive.
type npyFile struct {
	descr string
	shape []int
	data  []byte
}

// readNpz reads the .npy members of an .npz archive (a zip; stored or
// deflated members). Port of np.load on an npz file.
func readNpz(data []byte) (map[string]*npyFile, error) {
	zr, err := openZip(data)
	if err != nil {
		return nil, err
	}
	out := map[string]*npyFile{}
	for _, f := range zr {
		npy, err := readNpy(f.data)
		if err != nil {
			return nil, fmt.Errorf("indra: %s in npz: %w", f.name, err)
		}
		out[strings.TrimSuffix(f.name, ".npy")] = npy
	}
	return out, nil
}

type zipEntry struct {
	name string
	data []byte
}

// openZip parses zip archives without archive/zip (which needs io.ReaderAt
// and seek): the npz members are read back to front through the central
// directory, deflated members are inflated whole.
func openZip(data []byte) ([]zipEntry, error) {
	// End of central directory: signature 0x06054b50, last 22 bytes + comment.
	eocd := -1
	for i := len(data) - 22; i >= 0 && i > len(data)-22-65536; i-- {
		if binary.LittleEndian.Uint32(data[i:]) == 0x06054b50 {
			eocd = i
			break
		}
	}
	if eocd < 0 {
		return nil, fmt.Errorf("indra: npz: no zip end-of-central-directory")
	}
	n := int(binary.LittleEndian.Uint16(data[eocd+10:]))
	cdOff := int(binary.LittleEndian.Uint32(data[eocd+16:]))
	var out []zipEntry
	for i := 0; i < n; i++ {
		p := cdOff + i*46
		if p+46 > len(data) || binary.LittleEndian.Uint32(data[p:]) != 0x02014b50 {
			return nil, fmt.Errorf("indra: npz: bad central directory")
		}
		method := binary.LittleEndian.Uint16(data[p+10:])
		csize := int(binary.LittleEndian.Uint32(data[p+20:]))
		nameLen := int(binary.LittleEndian.Uint16(data[p+28:]))
		lho := int(binary.LittleEndian.Uint32(data[p+42:]))
		name := string(data[p+46 : p+46+nameLen])
		lp := lho
		if binary.LittleEndian.Uint32(data[lp:]) != 0x04034b50 {
			return nil, fmt.Errorf("indra: npz: bad local header for %s", name)
		}
		nl := int(binary.LittleEndian.Uint16(data[lp+26:]))
		el := int(binary.LittleEndian.Uint16(data[lp+28:]))
		start := lp + 30 + nl + el
		raw := data[start : start+csize]
		var b []byte
		switch method {
		case 0:
			b = raw
		case 8:
			fr := flate.NewReader(bytes.NewReader(raw))
			var err error
			b, err = readAllFlate(fr, csize)
			if err != nil {
				return nil, fmt.Errorf("indra: npz: inflate %s: %w", name, err)
			}
		default:
			return nil, fmt.Errorf("indra: npz: unsupported zip method %d for %s", method, name)
		}
		out = append(out, zipEntry{name: name, data: b})
	}
	return out, nil
}

func readAllFlate(fr io.Reader, hint int) ([]byte, error) {
	buf := make([]byte, 0, max(hint, 1<<16))
	p := make([]byte, 1<<16)
	for {
		n, err := fr.Read(p)
		buf = append(buf, p[:n]...)
		if err != nil {
			if err == io.EOF {
				return buf, nil
			}
			return nil, err
		}
	}
}

// readNpy parses one .npy blob: magic, header, raw data.
func readNpy(b []byte) (*npyFile, error) {
	if len(b) < 10 || !bytes.HasPrefix(b, []byte("\x93NUMPY")) {
		return nil, fmt.Errorf("not an npy file")
	}
	major := b[6]
	var hlen int
	if major >= 3 {
		hlen = int(binary.LittleEndian.Uint32(b[8:12]))
	} else {
		hlen = int(binary.LittleEndian.Uint16(b[8:10]))
	}
	hoff := 10
	if major >= 3 {
		hoff = 12
	}
	header := string(b[hoff : hoff+hlen])
	f := &npyFile{}
	if i := strings.Index(header, "'descr'"); i >= 0 {
		f.descr = pyStr(header[i+8:])
	}
	shape := header[strings.Index(header, "'shape'"):]
	la := strings.Index(shape, "(")
	lb := strings.Index(shape, ")")
	for _, part := range strings.Split(shape[la+1:lb], ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			var d int
			if _, err := fmt.Sscanf(part, "%d", &d); err == nil {
				f.shape = append(f.shape, d)
			}
		}
	}
	f.data = b[hoff+hlen:]
	return f, nil
}

// pyStr reads a quoted Python string at the start of s.
func pyStr(s string) string {
	q := s[0]
	s = s[1:]
	if i := strings.IndexByte(s, q); i >= 0 {
		return s[:i]
	}
	return s
}

// npyU64 reads a uint64 array ('<u8' or '|u8').
func npyU64(f *npyFile) []uint64 {
	n := len(f.data) / 8
	out := make([]uint64, n)
	for i := range out {
		out[i] = binary.LittleEndian.Uint64(f.data[i*8:])
	}
	return out
}

// npyF64 reads a float array ('<f8' or '<f4') as float64.
func npyF64(f *npyFile) []float64 {
	switch f.descr {
	case "<f8", "|f8", "=f8":
		n := len(f.data) / 8
		out := make([]float64, n)
		for i := range out {
			out[i] = math.Float64frombits(binary.LittleEndian.Uint64(f.data[i*8:]))
		}
		return out
	default: // "<f4" and friends
		n := len(f.data) / 4
		out := make([]float64, n)
		for i := range out {
			out[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(f.data[i*4:])))
		}
		return out
	}
}

// ngramLM is one recognizer's n-gram language model: sorted (history, token)
// tables per order with log-probabilities, and sorted history keys with
// backoff weights. All history lengths share one sorted table, the length in
// the top bits (keys use at most order x bits), so a step needs one search
// per table. Port of NgramLM in lm.py.
type ngramLM struct {
	order     int
	vocabSize int
	bits      int
	pKeys     []uint64
	pVals     []float64
	hKeys     []uint64
	hVals     []float64

	cacheMu sync.Mutex
	cache   map[[2]uint64]float64 // (packed history+token, token) -> logp
	maxp    float64
	maxpOK  bool
}

// tagKey packs a packed key with its history-length tag in the top bits.
func tagKey(k uint64, n int) uint64 {
	return k | uint64(n)<<uint(lmTagShift)
}

func newNgramLM(z map[string]*npyFile) (*ngramLM, error) {
	lm := &ngramLM{cache: map[[2]uint64]float64{}}
	lm.order = int(npyU64(z["order"])[0])
	lm.vocabSize = int(npyU64(z["vocab_size"])[0])
	lm.bits = lmDefaultBits
	if b, ok := z["bits"]; ok {
		lm.bits = int(npyU64(b)[0])
	}
	if lm.order*lm.bits > lmTagShift {
		return nil, fmt.Errorf("indra: lm: order %d x %d bits do not fit a key", lm.order, lm.bits)
	}
	type table struct {
		keys []uint64
		vals []float64
	}
	pt := make([]table, lm.order)
	ht := make([]table, lm.order)
	for n := 0; n < lm.order; n++ {
		pt[n] = table{npyU64(z[fmt.Sprintf("k%d", n)]), npyF64(z[fmt.Sprintf("v%d", n)])}
		ht[n] = table{npyU64(z[fmt.Sprintf("hk%d", n)]), npyF64(z[fmt.Sprintf("hb%d", n)])}
	}
	// All history lengths in one sorted table, the length in the top bits.
	type kv struct {
		k uint64
		v float64
	}
	pAll := make([]kv, 0, len(pt[0].keys)*lm.order)
	hAll := make([]kv, 0, len(ht[0].keys)*lm.order)
	for n := 0; n < lm.order; n++ {
		for i, k := range pt[n].keys {
			pAll = append(pAll, kv{tagKey(k, n), pt[n].vals[i]})
		}
		for i, k := range ht[n].keys {
			hAll = append(hAll, kv{tagKey(k, n), ht[n].vals[i]})
		}
	}
	sort.Slice(pAll, func(i, j int) bool { return pAll[i].k < pAll[j].k })
	sort.Slice(hAll, func(i, j int) bool { return hAll[i].k < hAll[j].k })
	lm.pKeys = make([]uint64, len(pAll))
	lm.pVals = make([]float64, len(pAll))
	for i, e := range pAll {
		lm.pKeys[i], lm.pVals[i] = e.k, e.v
	}
	lm.hKeys = make([]uint64, len(hAll))
	lm.hVals = make([]float64, len(hAll))
	for i, e := range hAll {
		lm.hKeys[i], lm.hVals[i] = e.k, e.v
	}
	return lm, nil
}

// find returns the index of key in a sorted table, or -1.
func (lm *ngramLM) find(keys []uint64, key uint64) int {
	i := sort.Search(len(keys), func(j int) bool { return keys[j] >= key })
	if i < len(keys) && keys[i] == key {
		return i
	}
	return -1
}

// logp returns the (backed-off) log-probability of token t after hist.
// hist is newest-last, packed per call. Port of NgramLM.logp.
func (lm *ngramLM) logp(hist []int, t int) float64 {
	hk := lmPack(hist, lm.bits)
	ck := [2]uint64{hk, uint64(t + 1)}
	lm.cacheMu.Lock()
	v, ok := lm.cache[ck]
	lm.cacheMu.Unlock()
	if ok {
		return v
	}
	top := min(len(hist), lm.order-1)
	// hs[n]: the last n tokens of hist.
	hs := make([][]int, top+1)
	for n := 1; n <= top; n++ {
		hs[n] = hist[len(hist)-n:]
	}
	m := 0 // the longest history the model has seen
	for n := 1; n <= top; n++ {
		if lm.find(lm.hKeys, tagKey(lmPack(hs[n], lm.bits), n)) < 0 {
			break
		}
		m = n
	}
	acc, val := 0.0, math.NaN()
	for n := m; n >= 0; n-- {
		pk := lmPack(append(append([]int{}, hs[n]...), t), lm.bits)
		if i := lm.find(lm.pKeys, tagKey(pk, n)); i >= 0 {
			val = acc + lm.pVals[i]
			break
		}
		acc += lm.hVals[lm.find(lm.hKeys, tagKey(lmPack(hs[n], lm.bits), n))]
	}
	if math.IsNaN(val) {
		val = acc - math.Log(float64(lm.vocabSize))
	}
	lm.cacheMu.Lock()
	if len(lm.cache) < 1_000_000 {
		lm.cache[ck] = val
	}
	lm.cacheMu.Unlock()
	return val
}

// maxLogp bounds every value logp returns: the largest log-probability plus
// every positive backoff weight a backoff can add. Port of max_logp.
func (lm *ngramLM) maxLogp() float64 {
	if lm.maxpOK {
		return lm.maxp
	}
	top := -math.Log(float64(lm.vocabSize))
	for _, v := range lm.pVals {
		if v > top {
			top = v
		}
	}
	bo := 0.0
	for _, v := range lm.hVals {
		if v > 0 {
			bo += v
		}
	}
	lm.maxp = top + bo
	lm.maxpOK = true
	return lm.maxp
}

// lmLSE is log-sum-exp of two floats (-inf handled). Port of _lse.
func lmLSE(a, b float64) float64 {
	if math.IsInf(a, -1) {
		return b
	}
	if math.IsInf(b, -1) {
		return a
	}
	m := math.Max(a, b)
	return m + math.Log(math.Exp(a-m)+math.Exp(b-m))
}

// lmBeam is one CTC prefix beam: (log p ending in blank, log p ending in
// non-blank, lm score, steps, the prefix's last order-1 tokens packed).
type lmBeam struct {
	b, nb, s float64
	st       []int
	hk       uint64
}

// lmBeamSearch runs the CTC prefix beam search over a line's steps.
// blank: [T] blank probabilities; topv/topi: [T*K] the k best classes per
// step (blank included) and their probabilities; a class under candFloor
// probability at a step is not tried there. Returns (token ids, the step
// where each token was first emitted). Port of beam_search in lm.py.
func lmBeamSearch(blank, topv []float32, topi []int64, t, k int, lm *ngramLM,
	alpha, beta float64, beam int, candFloor float64) (ids []int, steps []int) {
	bits := lm.bits
	hmask := uint64(1)<<uint(bits*(lm.order-1)) - 1
	beams := map[string]*lmBeam{"": {b: 0, nb: math.Inf(-1), s: 0, st: nil, hk: 0}}
	logSkip := math.Log(0.999)
	prefixKey := func(p []int) string {
		return string(packPrefix(p))
	}
	for tt := 0; tt < t; tt++ {
		b0 := float64(blank[tt])
		if b0 > logSkip {
			nxt := map[string]*lmBeam{}
			for p, e := range beams {
				nxt[p] = &lmBeam{b: lmLSE(e.b, e.nb) + b0, nb: math.Inf(-1), s: e.s, st: e.st, hk: e.hk}
			}
			beams = nxt
			continue
		}
		nxt := map[string]*lmBeam{}
		add := func(prefix string, b, nb, s float64, st []int, hk uint64) {
			if old, ok := nxt[prefix]; ok {
				old.b, old.nb = lmLSE(old.b, b), lmLSE(old.nb, nb)
				return
			}
			nxt[prefix] = &lmBeam{b: b, nb: nb, s: s, st: st, hk: hk}
		}
		var cands []struct {
			c  int
			lp float64
		}
		for q := 0; q < k; q++ {
			c := int(topi[tt*k+q])
			if v := float64(topv[tt*k+q]); c != 0 && v >= candFloor {
				cands = append(cands, struct {
					c  int
					lp float64
				}{c, math.Log(v)})
			}
		}
		// Every beam's own entries first (the blank, and a repeat that
		// collapses): their scores only grow below, so with a full beam the
		// lowest of them is a floor; a new prefix that cannot reach it never
		// enters the beam, and its LM score is not looked up.
		for p, e := range beams {
			add(p, lmLSE(e.b, e.nb)+b0, math.Inf(-1), e.s, e.st, e.hk)
			last := -1
			if pl := unprefix(p); len(pl) > 0 {
				last = pl[len(pl)-1]
			}
			for _, cd := range cands {
				if cd.c == last {
					add(p, math.Inf(-1), e.nb+cd.lp, e.s, e.st, e.hk)
				}
			}
		}
		floor := math.Inf(-1)
		if len(beams) >= int(beam) {
			floor = math.Inf(1)
			for _, e := range nxt {
				floor = math.Min(floor, lmLSE(e.b, e.nb)+e.s)
			}
		}
		reach := beta + alpha*lm.maxLogp()
		type ext_t struct {
			prefix       []int
			pk           string
			c            int
			newFrom, s   float64
			st           []int
			hk           uint64
		}
		var exts []ext_t
		for p, e := range beams {
			pl := unprefix(p)
			tot := lmLSE(e.b, e.nb)
			last := -1
			if len(pl) > 0 {
				last = pl[len(pl)-1]
			}
			for _, cd := range cands {
				// after a blank, the same class is a new token
				newFrom := tot + cd.lp
				if cd.c == last {
					newFrom = e.b + cd.lp
				}
				if math.IsInf(newFrom, -1) {
					continue
				}
				q := append(append([]int{}, pl...), cd.c)
				qk := prefixKey(q)
				if _, exists := nxt[qk]; exists {
					// an existing beam: merges, its LM score stays
					add(qk, math.Inf(-1), newFrom, e.s, e.st, e.hk)
				} else if newFrom+e.s+reach >= floor-1e-9 { // float slack
					exts = append(exts, ext_t{prefix: q, pk: qk, c: cd.c, newFrom: newFrom, s: e.s, st: e.st, hk: e.hk})
				}
			}
		}
		for _, e := range exts {
			lp := lm.logp(unpackHist(e.hk, bits, lm.order-1), e.c)
			st := append(append([]int{}, e.st...), tt)
			add(e.pk, math.Inf(-1), e.newFrom, e.s+alpha*lp+beta, st, (e.hk<<uint(bits)|uint64(e.c+1))&hmask)
		}
		// keep the top `beam` by total score
		type scored struct {
			k string
			v float64
			e *lmBeam
		}
		all := make([]scored, 0, len(nxt))
		for p, e := range nxt {
			all = append(all, scored{p, lmLSE(e.b, e.nb) + e.s, e})
		}
		sort.Slice(all, func(i, j int) bool {
			if all[i].v != all[j].v {
				return all[i].v > all[j].v
			}
			return all[i].k < all[j].k
		})
		if len(all) > int(beam) {
			all = all[:beam]
		}
		beams = map[string]*lmBeam{}
		for _, e := range all {
			beams[e.k] = e.e
		}
	}
	// best by total score
	var bestP string
	var bestE *lmBeam
	bestV := math.Inf(-1)
	for p, e := range beams {
		v := lmLSE(e.b, e.nb) + e.s
		if v > bestV || bestE == nil {
			bestV, bestE, bestP = v, e, p
		}
	}
	pl := unprefix(bestP)
	return append([]int{}, pl...), append([]int{}, bestE.st...)
}

// packPrefix packs token ids into a map key: 2 bytes each (vocabularies
// stay under 65536 classes).
func packPrefix(p []int) []byte {
	b := make([]byte, 2*len(p))
	for i, v := range p {
		binary.LittleEndian.PutUint16(b[i*2:], uint16(v))
	}
	return b
}

// unprefix unpacks a key made by unsafeBytes.
func unprefix(p string) []int {
	if p == "" {
		return nil
	}
	out := make([]int, len(p)/2)
	for i := range out {
		out[i] = int(binary.LittleEndian.Uint16([]byte(p[i*2:])))
	}
	return out
}

// unpackHist unpacks the last order-1 tokens from a packed history key.
func unpackHist(hk uint64, bits, order1 int) []int {
	out := make([]int, order1)
	for i := order1 - 1; i >= 0; i-- {
		v := hk & ((1 << uint(bits)) - 1)
		hk >>= uint(bits)
		if v == 0 {
			out[i] = lmBOS
		} else {
			out[i] = int(v) - 1
		}
	}
	return out
}

// lmStore holds the process-shared language models, loaded lazily from the
// embedded npz files (they are large: ~110 MB decompressed in total, one
// copy shared by every OCR handle).
var lmStore = struct {
	sync.Mutex
	m map[string]*ngramLM
}{m: map[string]*ngramLM{}}

// lmFor loads the named model's LM ("base" = lm.npz, "cjk" = lm_cjk.npz, ...)
// once per process; nil when the model has no LM file.
func lmFor(group string) *ngramLM {
	file := "lm.npz"
	if group != "base" {
		file = "lm_" + group + ".npz"
	}
	lmStore.Lock()
	defer lmStore.Unlock()
	if lm, ok := lmStore.m[group]; ok {
		return lm
	}
	lmStore.m[group] = nil // sentinel: failed loads are not retried
	data, err := modelFS.ReadFile("models/" + file)
	if err != nil {
		return nil
	}
	z, err := readNpz(data)
	if err != nil {
		return nil
	}
	lm, err := newNgramLM(z)
	if err != nil {
		return nil
	}
	lmStore.m[group] = lm
	return lm
}
