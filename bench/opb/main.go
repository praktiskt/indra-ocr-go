// Command opb runs the Go port over an Omni Parse Bench manifest and emits
// markdown + labeled blocks as JSONL for omni_parse_bench.
//
// One page at a time (matches the upstream benchmark's methodology), through
// the pool. Build + run:
//
//	go run ./bench/opb -manifest /tmp/opb/manifest.json -out /tmp/opb/go.jsonl -workers 8
//
// then score with bench/opb/score.py. See bench/opb/README.md.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"image/png"
	"log"
	"os"
	"time"

	indra "github.com/praktiskt/indra-ocr-go"
)

type item struct {
	PageID string `json:"page_id"`
	PNG    string `json:"png"`
	Suite  string `json:"suite"`
}

type block struct {
	Label string     `json:"label"`
	BBox  [4]float64 `json:"bbox"`
	Text  string     `json:"text"`
}

type row struct {
	PageID   string  `json:"page_id"`
	Suite    string  `json:"suite"`
	Markdown string  `json:"markdown"`
	Blocks   []block `json:"blocks"`
	MS       float64 `json:"ms"`
}

func main() {
	manifest := flag.String("manifest", "", "manifest.json from prep.py")
	out := flag.String("out", "", "output JSONL")
	workers := flag.Int("workers", 8, "pool workers")
	threads := flag.Int("threads", 1, "intra-op threads per worker")
	furniture := flag.Bool("furniture", false, "keep page headers/footers in markdown")
	mathOn := flag.Bool("math", false, "read math lines as LaTeX")
	flag.Parse()
	if *manifest == "" || *out == "" {
		log.Fatal("need -manifest and -out")
	}

	raw, err := os.ReadFile(*manifest)
	check(err)
	var items []item
	check(json.Unmarshal(raw, &items))

	p, err := indra.NewPoolBuilder().Workers(*workers).Threads(*threads).Furniture(*furniture).Math(*mathOn).Build()
	check(err)
	defer p.Close()

	f, err := os.Create(*out)
	check(err)
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()
	enc := json.NewEncoder(w)

	t0 := time.Now()
	for i, it := range items {
		data, err := os.ReadFile(it.PNG)
		if err != nil {
			continue
		}
		img, err := png.Decode(bytes.NewReader(data))
		if err != nil {
			continue
		}
		s := time.Now()
		page := p.Call(indra.FromImage(img))
		ms := float64(time.Since(s).Microseconds()) / 1000
		r := row{PageID: it.PageID, Suite: it.Suite, Markdown: page.Markdown, MS: ms}
		for _, b := range page.Blocks {
			r.Blocks = append(r.Blocks, block{Label: b.Label, BBox: b.BBox, Text: b.Text})
		}
		check(enc.Encode(r))
		if (i+1)%200 == 0 {
			log.Printf("%d/%d pages", i+1, len(items))
		}
	}
	log.Printf("done %d pages in %v", len(items), time.Since(t0))
}

func check(err error) {
	if err != nil {
		log.Fatal(err)
	}
}
