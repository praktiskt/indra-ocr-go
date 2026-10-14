// Page markdown + layout blocks. Port of layout_post.py.
package indra

import (
	"html"
	"math"
	"sort"
	"strings"
)

const (
	clsHeader  = 2 // Page-Header
	clsFooter  = 3 // Page-Footer
	clsSection = 1 // Section-Header
	clsList    = 4 // List-Group
	clsTable   = 5 // Table
)

// Block is a layout block in reading order.
type Block struct {
	Label string
	BBox  [4]float64
	Text  string
}

// JoinLines joins paragraph lines; 'tumori-'+'genesis' -> 'tumorigenesis';
// CJK-CJK break takes no space. Port of join_lines.
func JoinLines(texts []string) string {
	var out string
	for _, t := range texts {
		t = strings.TrimSpace(t)
		if t == "" {
			continue
		}
		ro := []rune(out)
		rt := []rune(t)
		switch {
		case len(ro) > 1 && ro[len(ro)-1] == '-' && isAlpha(ro[len(ro)-2]) && isAlpha(rt[0]) && isLower(rt[0]):
			out = string(ro[:len(ro)-1]) + t
		case out != "" && isCJK(ro[len(ro)-1]) && isCJK(rt[0]):
			out += t
		default:
			if out == "" {
				out = t
			} else {
				out += " " + t
			}
		}
	}
	return out
}

func isAlpha(r rune) bool {
	return (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || r > 127
}

func isLower(r rune) bool { return (r >= 'a' && r <= 'z') || (r > 127 && isLowerUnicode(r)) }

func isLowerUnicode(r rune) bool {
	// Lowercase test for non-ASCII without full unicode tables cost: use simple range check.
	return r >= 'a' && r <= 'z'
}

func isCJK(r rune) bool {
	o := int(r)
	return (o >= 0x3000 && o <= 0x30FF) || (o >= 0x3400 && o <= 0x9FFF) ||
		(o >= 0xF900 && o <= 0xFAFF) || (o >= 0xFF00 && o <= 0xFFEF) ||
		(o >= 0x20000 && o <= 0x2FA1F)
}

// PageBlocks builds the layout blocks behind PageMarkdown.
func PageBlocks(boxes [][4]float64, texts []string, order []int, cls [][]float64, same [][]float64) []Block {
	var out []Block
	for _, blk := range SplitBlocks(order, same) {
		x0, y0 := math.Inf(1), math.Inf(1)
		x1, y1 := math.Inf(-1), math.Inf(-1)
		var ts []string
		for _, i := range blk {
			b := boxes[i]
			x0 = math.Min(x0, b[0])
			y0 = math.Min(y0, b[1])
			x1 = math.Max(x1, b[2])
			y1 = math.Max(y1, b[3])
			ts = append(ts, texts[i])
		}
		label := "Text"
		if id := argmaxRowSum(cls, blk); id >= 0 && id < len(BlockClasses) {
			label = BlockClasses[id]
		}
		out = append(out, Block{
			Label: label,
			BBox:  [4]float64{r1(x0), r1(y0), r1(x1), r1(y1)},
			Text:  strings.Join(ts, "\n"),
		})
	}
	return out
}

// PageMarkdown renders one page. cells maps a line to its table cells when a
// line holds several; tab holds same-row/col/cell logits when available.
func PageMarkdown(boxes [][4]float64, texts []string, order []int, cls [][]float64, same [][]float64, keepFurniture bool, cells map[int][][2]string, cellBoxes map[int][][4]float64) string {
	var parts []string
	for _, blk := range SplitBlocks(order, same) {
		id := argmaxRowSum(cls, blk)
		if (id == clsHeader || id == clsFooter) && !keepFurniture {
			continue
		}
		tb := make([]string, len(blk))
		for k, i := range blk {
			tb[k] = texts[i]
		}
		switch {
		case id == clsSection:
			parts = append(parts, "## "+JoinLines(tb))
		case id == clsTable && len(blk) > 1:
			var cb [][4]float64
			var ct []string
			for _, i := range blk {
				if cells != nil {
					if cs, ok := cells[i]; ok && len(cs) > 0 {
						for k, c := range cs {
							ct = append(ct, c[1])
							if cellBoxes != nil {
								if bx, ok := cellBoxes[i]; ok && k < len(bx) {
									cb = append(cb, bx[k])
									continue
								}
							}
							cb = append(cb, boxes[i])
						}
						continue
					}
				}
				cb = append(cb, boxes[i])
				ct = append(ct, texts[i])
			}
			parts = append(parts, TableHTML(cb, ct))
		case id == clsList:
			var items [][]string
			for _, t := range tb {
				if len(items) == 0 || isListItem(t) {
					items = append(items, []string{t})
				} else {
					items[len(items)-1] = append(items[len(items)-1], t)
				}
			}
			var lines []string
			for _, it := range items {
				lines = append(lines, JoinLines(it))
			}
			parts = append(parts, strings.Join(lines, "\n"))
		default:
			parts = append(parts, JoinLines(tb))
		}
	}
	var kept []string
	for _, p := range parts {
		if p != "" {
			kept = append(kept, p)
		}
	}
	return strings.Join(kept, "\n\n")
}

func isListItem(t string) bool {
	s := strings.TrimSpace(t)
	for _, p := range []string{"•", "-", "–", "▪", "◦", "*", "·", "→"} {
		if strings.HasPrefix(s, p) {
			rest := s[len(p):]
			if rest == "" || rest[0] == ' ' || rest[0] == '\t' {
				return true
			}
		}
	}
	// (1) / 1. / 1) / a. prefixes
	i := 0
	if i < len(s) && s[i] == '(' {
		i++
	}
	d := 0
	for i+d < len(s) && d < 3 && s[i+d] >= '0' && s[i+d] <= '9' {
		d++
	}
	if d > 0 && i+d < len(s) && (s[i+d] == '.' || s[i+d] == ')') {
		return true
	}
	if len(s) >= 2 && isASCIILetter(rune(s[0])) && (s[1] == '.' || s[1] == ')') {
		return true
	}
	return false
}

// TableHTML builds an HTML table from cell lines. Port of table_html.
func TableHTML(boxes [][4]float64, texts []string) string {
	if len(boxes) == 0 {
		return ""
	}
	rows := tableRows(boxes)
	spanX := maxBoxX1(boxes) - minBoxX0(boxes)
	if len(rows) > 2 && len(rows[0]) == 1 && maxRowLen(rows) > 1 &&
		boxes[rows[0][0]][2]-boxes[rows[0][0]][0] > 0.6*spanX {
		// Title row: emit as text, recurse on the rest.
		keep := []int{}
		seen := map[int]bool{rows[0][0]: true}
		for i := range boxes {
			if !seen[i] {
				keep = append(keep, i)
			}
		}
		nb := make([][4]float64, len(keep))
		nt := make([]string, len(keep))
		for i, k := range keep {
			nb[i], nt[i] = boxes[k], texts[k]
		}
		return texts[rows[0][0]] + "\n\n" + TableHTML(nb, nt)
	}
	width := maxRowLen(rows)
	full := [][]int{}
	for _, r := range rows {
		if len(r) == width {
			full = append(full, r)
		}
	}
	rows = mergeWrapped(rows, boxes)
	// Column bands from fullest rows.
	bands := make([][2]float64, width)
	for c := 0; c < width; c++ {
		var l, r []float64
		for _, fr := range full {
			l = append(l, boxes[fr[c]][0])
			r = append(r, boxes[fr[c]][2])
		}
		bands[c] = [2]float64{median(l), median(r)}
	}
	edges := make([]float64, width+1)
	edges[0] = math.Inf(-1)
	for c := 0; c < width-1; c++ {
		edges[c+1] = (bands[c][1] + bands[c+1][0]) / 2
	}
	edges[width] = math.Inf(1)
	var sb strings.Builder
	sb.WriteString("<table>")
	for ri, r := range rows {
		tag := "td"
		if ri == 0 && len(rows) > 1 {
			tag = "th"
		}
		type cell struct {
			span int
			text string
		}
		got := map[int]cell{}
		ord := append([]int(nil), r...)
		sort.Slice(ord, func(p, q int) bool {
			if boxes[ord[p]][1] != boxes[ord[q]][1] {
				return boxes[ord[p]][1] < boxes[ord[q]][1]
			}
			return boxes[ord[p]][0] < boxes[ord[q]][0]
		})
		for _, k := range ord {
			x0, x1 := boxes[k][0], boxes[k][1]
			_ = x1
			first := searchEdges(edges, boxes[k][0]) - 1
			last := searchEdges(edges, boxes[k][2]) - 1
			lo, hi := max(0, first), min(width-1, last)
			var span []int
			for c := lo; c <= hi; c++ {
				if math.Min(boxes[k][2], bands[c][1])-math.Max(x0, bands[c][0]) > 0.5*(bands[c][1]-bands[c][0]) {
					span = append(span, c)
				}
			}
			if len(span) == 0 {
				span = []int{min(max(first, 0), width-1)}
			}
			c0 := span[0]
			if prev, ok := got[c0]; ok {
				if len(span) > prev.span {
					prev.span = len(span)
				}
				prev.text += " " + texts[k]
				got[c0] = prev
			} else {
				got[c0] = cell{len(span), texts[k]}
			}
		}
		sb.WriteString("<tr>")
		for c := 0; c < width; {
			if cl, ok := got[c]; ok {
				if cl.span > 1 {
					sb.WriteString("<" + tag + " colspan=\"" + itoa(cl.span) + "\">" + html.EscapeString(cl.text) + "</" + tag + ">")
				} else {
					sb.WriteString("<" + tag + ">" + html.EscapeString(cl.text) + "</" + tag + ">")
				}
				c += cl.span
			} else {
				sb.WriteString("<" + tag + "></" + tag + ">")
				c++
			}
		}
		sb.WriteString("</tr>")
	}
	sb.WriteString("</table>")
	return sb.String()
}

func searchEdges(edges []float64, x float64) int {
	// first edge > x (like np.searchsorted right)
	lo, hi := 0, len(edges)
	for lo < hi {
		m := (lo + hi) / 2
		if edges[m] <= x {
			lo = m + 1
		} else {
			hi = m
		}
	}
	return lo
}

func tableRows(boxes [][4]float64) [][]int {
	n := len(boxes)
	cy := make([]float64, n)
	for i, b := range boxes {
		cy[i] = (b[1] + b[3]) / 2
	}
	hs := make([]float64, n)
	for i, b := range boxes {
		hs[i] = b[3] - b[1]
	}
	h := median(hs)
	ord := make([]int, n)
	for i := range ord {
		ord[i] = i
	}
	sort.Slice(ord, func(p, q int) bool { return cy[ord[p]] < cy[ord[q]] })
	var rows [][]int
	for _, i := range ord {
		if len(rows) > 0 {
			mean := 0.0
			for _, k := range rows[len(rows)-1] {
				mean += cy[k]
			}
			mean /= float64(len(rows[len(rows)-1]))
			if math.Abs(cy[i]-mean) < 0.5*h {
				rows[len(rows)-1] = append(rows[len(rows)-1], i)
				continue
			}
		}
		rows = append(rows, []int{i})
	}
	for _, r := range rows {
		sort.Slice(r, func(p, q int) bool { return boxes[r[p]][0] < boxes[r[q]][0] })
	}
	return rows
}

func mergeWrapped(rows [][]int, boxes [][4]float64) [][]int {
	if len(rows) < 3 {
		return rows
	}
	hs := make([]float64, len(boxes))
	for i, b := range boxes {
		hs[i] = b[3] - b[1]
	}
	h := median(hs)
	var gaps []float64
	for i := 0; i+1 < len(rows); i++ {
		mx := -math.MaxFloat64
		for _, k := range rows[i] {
			if boxes[k][3] > mx {
				mx = boxes[k][3]
			}
		}
		mn := math.MaxFloat64
		for _, k := range rows[i+1] {
			if boxes[k][1] < mn {
				mn = boxes[k][1]
			}
		}
		gaps = append(gaps, mn-mx)
	}
	med := median(gaps)
	out := [][]int{append([]int(nil), rows[0]...)}
	prev := rows[0]
	for i, g := range gaps {
		r := rows[i+1]
		if g < 0.5*med && g < 0.6*h && len(r) < len(prev) {
			out[len(out)-1] = append(out[len(out)-1], r...)
		} else {
			out = append(out, append([]int(nil), r...))
		}
		prev = r
	}
	return out
}

func median(v []float64) float64 {
	return npMedian(sortedCopy(v))
}

func sortedCopy(v []float64) []float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s
}

// npMedian matches numpy.median (averages the two middle values for even n).
func npMedian(s []float64) float64 {
	n := len(s)
	if n == 0 {
		return 0
	}
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func minBoxX0(b [][4]float64) float64 {
	m := math.Inf(1)
	for _, x := range b {
		m = math.Min(m, x[0])
	}
	return m
}

func maxBoxX1(b [][4]float64) float64 {
	m := math.Inf(-1)
	for _, x := range b {
		m = math.Max(m, x[2])
	}
	return m
}

func maxRowLen(rows [][]int) int {
	m := 0
	for _, r := range rows {
		m = max(m, len(r))
	}
	return m
}

func r1(v float64) float64 { return math.Round(v*10) / 10 }

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [8]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
