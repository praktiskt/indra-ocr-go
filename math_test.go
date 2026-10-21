package indra

import "testing"

// TestMathGarbled: a math reading that spells prose out letter by letter is
// worse than the text it would replace (the PDF text layer's own characters
// keep the diacritics and the spacing).
func TestMathGarbled(t *testing.T) {
	garbled := []string{
		// prose read as math: single letters in a row inside a span, no
		// LaTeX command
		"Francesca Castaldo,^{5,\\dagger}$ M a r c o T u c c i o , ^ { 1 , 3 }$",
		"Manlio de Domenico,7,8,9,10 D a v i d C . K r a k a u e r , $^{4}$",
		"$M a r c o$",
	}
	for _, s := range garbled {
		if !mathGarbled(s) {
			t.Errorf("mathGarbled(%q) = false, want true", s)
		}
	}
	clean := []string{
		// genuine math: LaTeX commands, operators, single letters apart
		"$\\frac { d V } { d t } ~ = ~ - \\lambda U C + \\rho C$",
		"$\\mathrm { ~ ( 1 ) }$",
		"$\\mathbf { L e m m a }$ 7.3.1.",
		"$U + C + D \\ = \\ 1 .$",
		"$-$",
		"the quadratic dependence on $U$ represents",
		"",
	}
	for _, s := range clean {
		if mathGarbled(s) {
			t.Errorf("mathGarbled(%q) = true, want false", s)
		}
	}
}
