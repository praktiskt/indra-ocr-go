package indra

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

type orderRef struct {
	Boxes       [][4]float64 `json:"boxes"`
	Texts       []string     `json:"texts"`
	Feats       [][]float64  `json:"feats"`
	A           [][]float64  `json:"A"`
	Same        [][]float64  `json:"same"`
	Cls         [][]float64  `json:"cls"`
	Order       []int        `json:"order"`
	Clusters    [][]int      `json:"clusters"`
	Xycut       []int        `json:"xycut"`
	BlocksOrder []int        `json:"blocks_order"`
}

func loadOrderRef(t *testing.T) orderRef {
	t.Helper()
	p := os.Getenv("INDRA_REF_JSON")
	if p == "" {
		p = "/tmp/opencode/order_ref.json"
	}
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Skipf("no ref json: %v", err)
	}
	var r orderRef
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func to2(a [][4]float64) [][]float64 {
	m := make([][]float64, len(a))
	for i, b := range a {
		m[i] = []float64{b[0], b[1], b[2], b[3]}
	}
	return m
}

func TestDiffLineFeatures(t *testing.T) {
	r := loadOrderRef(t)
	got := LineFeatures(r.Boxes, 900, 1200, r.Texts)
	flat := []float64{}
	for _, row := range r.Feats {
		flat = append(flat, row...)
	}
	if len(got) != len(flat) {
		t.Fatalf("len %d != %d", len(got), len(flat))
	}
	maxD := 0.0
	for i := range got {
		if d := math.Abs(float64(got[i]) - flat[i]); d > maxD {
			maxD = d
		}
	}
	t.Logf("max feats abs diff = %g", maxD)
	if maxD > 1e-5 {
		t.Fatalf("features diverge")
	}
}

func TestDiffDecodeOrderFull(t *testing.T) {
	r := loadOrderRef(t)
	if got := DecodeOrder(r.A, 4); !reflect.DeepEqual(got, r.Order) {
		t.Fatalf("order differs:\n got %v\nwant %v", got, r.Order)
	}
}

func TestDiffCluster(t *testing.T) {
	r := loadOrderRef(t)
	got := cluster(r.Same)
	want := r.Clusters
	if len(got) != len(want) {
		t.Fatalf("cluster count %d != %d\ngot %v\nwant %v", len(got), len(want), got, want)
	}
	// compare as sets of sets
	key := func(g []int) string {
		s := ""
		for _, v := range g {
			s += string(rune('0'+v/10)) + string(rune('0'+v%10)) + ","
		}
		return s
	}
	_ = key
	gs := map[string]bool{}
	for _, g := range got {
		gs[intSetKey(g)] = true
	}
	for _, w := range want {
		if !gs[intSetKey(w)] {
			t.Fatalf("cluster mismatch:\n got %v\nwant %v", got, want)
		}
	}
}

func intSetKey(g []int) string {
	s := ""
	for _, v := range g {
		if v < 10 {
			s += string(rune('0' + v))
		} else {
			s += string(rune('0'+v/10)) + string(rune('0'+v%10))
		}
		s += ","
	}
	return s
}

func TestDiffBlocksOrder(t *testing.T) {
	r := loadOrderRef(t)
	if got := DecodeBlocksOrder(r.A, r.Same, r.Boxes, r.Cls); !reflect.DeepEqual(got, r.BlocksOrder) {
		t.Fatalf("blocks_order differs:\n got %v\nwant %v", got, r.BlocksOrder)
	}
}

func TestDiffDecodeOrder(t *testing.T) {
	// python: decode_order([[0,5],[-5,0]]) == [0,1]
	a := [][]float64{{0, 5}, {-5, 0}}
	if o := DecodeOrder(a, 4); !reflect.DeepEqual(o, []int{0, 1}) {
		t.Fatalf("order = %v", o)
	}
}

func TestDiffXYCut(t *testing.T) {
	r := loadOrderRef(t)
	if got := OrderXYCut(r.Boxes); !reflect.DeepEqual(got, r.Xycut) {
		t.Fatalf("xycut differs:\n got %v\nwant %v", got, r.Xycut)
	}
}
