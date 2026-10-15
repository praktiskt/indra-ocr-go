package indra

// Broad quality guard over a document corpus (default: FUNSD, 199 diverse
// forms/faxes/ads). Catches perf changes that silently degrade text quality
// on pages outside testdata/.
//
// Layout: INDRA_CORPUS_DIR/img/*.png + gt.json mapping image basename to its
// ground-truth word list, e.g. {"train_0.png": ["Monthly", ...]}.
//
// FUNSD fetch (needs uv + ~15MB):
//
//	uv run --with pyarrow python - <<'EOF'
//	import pyarrow.parquet as pq, json, os
//	os.makedirs("img", exist_ok=True)
//	gt = {}
//	for split, f in [("test", "test.parquet"), ("train", "train.parquet")]:
//	    # from https://huggingface.co/datasets/nielsr/funsd data/
//	    for row in pq.read_table(f).to_pylist():
//	        name = f"{split}_{row['id']}.png"
//	        open(f"img/{name}", "wb").write(row["image"]["bytes"])
//	        gt[name] = row["words"]
//	json.dump(gt, open("gt.json", "w"))
//	EOF
//
// Run with INDRA_CORPUS_DIR=/path/to/corpus. Needs models + lib like the
// other image tests. Measured reference (v0.8.0, threads=2): word recall
// ~0.85; threshold 0.80 leaves margin for refactor noise while catching
// real regressions (e.g. det side 768 shifts 178/199 pages' output).

import (
	"bytes"
	"encoding/json"
	"image/png"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

var (
	corpusTokRe  = regexp.MustCompile(`[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?`)
	corpusWordRe = regexp.MustCompile(`^[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?$`)
)

func corpusWords(lines []string) map[string]bool {
	set := map[string]bool{}
	for _, ln := range lines {
		for _, w := range corpusTokRe.FindAllString(strings.ToLower(ln), -1) {
			set[w] = true
		}
	}
	return set
}

func TestCorpusRecall(t *testing.T) {
	dir := os.Getenv("INDRA_CORPUS_DIR")
	if dir == "" {
		t.Skip("INDRA_CORPUS_DIR not set")
	}
	if err := InitLib(); err != nil {
		t.Fatalf("InitLib: %v", err)
	}
	gtRaw, err := os.ReadFile(filepath.Join(dir, "gt.json"))
	if err != nil {
		t.Fatalf("read gt.json: %v", err)
	}
	var gt map[string][]string
	if err := json.Unmarshal(gtRaw, &gt); err != nil {
		t.Fatalf("parse gt.json: %v", err)
	}
	files, err := filepath.Glob(filepath.Join(dir, "img", "*.png"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no images in %s/img", dir)
	}
	sort.Strings(files)

	type result struct {
		hit, total int
		ms         int64
	}
	results := make([]result, len(files))
	ch := make(chan int, len(files))
	for i := range files {
		ch <- i
	}
	close(ch)
	t0 := time.Now()
	var wg sync.WaitGroup
	for wi := 0; wi < 8; wi++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ocr, err := NewOCR(2)
			if err != nil {
				panic(err)
			}
			defer ocr.Close()
			for i := range ch {
				data, err := os.ReadFile(files[i])
				if err != nil {
					panic(err)
				}
				img, err := png.Decode(bytes.NewReader(data))
				if err != nil {
					panic(err)
				}
				s := time.Now()
				p := ocr.Call(FromImage(img))
				pred := corpusWords(lineTexts(p))
				base := filepath.Base(files[i])
				words, ok := gt[base]
				if !ok {
					// accept keys with or without extension
					words = gt[strings.TrimSuffix(base, filepath.Ext(base))]
				}
				var g []string
				for _, w := range words {
					if w = strings.ToLower(w); corpusWordRe.MatchString(w) {
						g = append(g, w)
					}
				}
				hit := 0
				for _, w := range g {
					if pred[w] {
						hit++
					}
				}
				results[i] = result{hit: hit, total: len(g), ms: time.Since(s).Milliseconds()}
			}
		}()
	}
	wg.Wait()
	hit, total, ms := 0, 0, int64(0)
	pages := 0
	for _, r := range results {
		if r.total == 0 {
			continue
		}
		hit += r.hit
		total += r.total
		ms += r.ms
		pages++
	}
	recall := 0.0
	if total > 0 {
		recall = float64(hit) / float64(total)
	}
	t.Logf("corpus: pages=%d word_recall=%.4f avg_ms=%d total=%v", pages, recall, ms/int64(max(pages, 1)), time.Since(t0))
	if pages == 0 {
		t.Fatalf("corpus: no pages with ground truth")
	}
	if recall < 0.80 {
		t.Fatalf("corpus word recall %.4f < 0.80", recall)
	}
}

func lineTexts(p *Page) []string {
	out := make([]string, len(p.Lines))
	for i, l := range p.Lines {
		out[i] = l.Text
	}
	return out
}
