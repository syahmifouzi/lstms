package main

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
)

// example1 := "Doug saw Jane."
// example2 := "Jane saw Spot."
// example3 := "Spot saw Doug."

// there should be 5 vectors here "Jane0 saw1 Doug2 Spot3 .4"

// X is inputs
var X = [][]float64{
	{0, 0, 1, 0, 0},
	{0, 1, 0, 0, 0},
	{1, 0, 0, 0, 0},
	{0, 0, 0, 0, 1},
	{1, 0, 0, 0, 0},
	{0, 1, 0, 0, 0},
	{0, 0, 0, 1, 0},
	{0, 0, 0, 0, 1},
	{0, 0, 0, 1, 0},
	{0, 1, 0, 0, 0},
	{0, 0, 1, 0, 0},
	{0, 0, 0, 0, 1},
}

var wi [][]float64
var bi [][]float64
var wf [][]float64
var bf [][]float64
var wo [][]float64
var bo [][]float64
var wc [][]float64
var bc [][]float64

func main() {
	// declare n set init variables
	dumm1 := [][]float64{{0, 0, 0, 0, 0}}
	// var checkErr [][]float64
	// checkErr = zeros(1, 5)

	qcPrev2 := [][]float64{{0, 0, 0, 0, 0}}
	qhPrev2 := [][]float64{{0, 0, 0, 0, 0}}

	initWeight(X, dumm1)

	// train the module
	for n := 0; n < 10000; n++ {

		// state := [][]float64{{0, 0, 0, 0, 0}}
		cPrev := [][]float64{{0, 0, 0, 0, 0}}
		hPrev := [][]float64{{0, 0, 0, 0, 0}}
		dhNext := [][]float64{{0, 0, 0, 0, 0}}
		dcNext := [][]float64{{0, 0, 0, 0, 0}}
		hfs := [][]float64{}
		his := [][]float64{}
		hos := [][]float64{}
		hcs := [][]float64{}
		hs := [][]float64{}
		cs := [][]float64{}
		probs := [][]float64{}
		xhs := [][]float64{}
		cPrevs := [][]float64{}

		// (forward propagation)
		for i, v := range X {
			x := [][]float64{}
			x = append(x, v)
			xh := tambah(x, hPrev)
			xhs = append(xhs, xh[0])
			h, c, hf, hi, ho, hc := lstmForward(xh, cPrev)
			hfs = append(hfs, hf[0])
			his = append(his, hi[0])
			hos = append(hos, ho[0])
			hcs = append(hcs, hc[0])
			hs = append(hs, h[0])
			cs = append(cs, c[0])
			cPrevs = append(cPrevs, cPrev[0])

			cPrev = c
			hPrev = h
			// qcPrev2 = cPrev

			p := [][]float64{}
			if i+1 != len(X) {
				p = append(p, X[i+1])
			} else {
				p = append(p, X[0])
			}
			y := tolak(h, p)
			// prob is = dh
			// prob := darab(y, sigmoidDeriv(h))
			probs = append(probs, y[0])

			// y := dot(h, wy)
			// y = tambah(y, by)
			// prob := softmax(y)
			// probs = append(probs, prob[0])

			// if n%100 == 0 {
			// 	fmt.Println("input: ", x)
			// 	fmt.Println("output: ", h)
			// 	// fmt.Println("Err: ", checkErr)
			// }
		}

		// reverse everything first
		// fmt.Println("hos: ", hos)
		for i := len(hfs)/2 - 1; i >= 0; i-- {
			opp := len(hfs) - 1 - i
			hfs[i], hfs[opp] = hfs[opp], hfs[i]
			his[i], his[opp] = his[opp], his[i]
			hos[i], hos[opp] = hos[opp], hos[i]
			hcs[i], hcs[opp] = hcs[opp], hcs[i]
			hs[i], hs[opp] = hs[opp], hs[i]
			cs[i], cs[opp] = cs[opp], cs[i]
			probs[i], probs[opp] = probs[opp], probs[i]
			xhs[i], xhs[opp] = xhs[opp], xhs[i]
			cPrevs[i], cPrevs[opp] = cPrevs[opp], cPrevs[i]
		}

		// fmt.Println("Hello world!")
		// fmt.Print("Press 'Enter' to continue...")
		// bufio.NewReader(os.Stdin).ReadBytes('\n')
		// return

		// (Back propragation)
		for i := range X {

			hf := [][]float64{}
			hi := [][]float64{}
			ho := [][]float64{}
			hc := [][]float64{}
			h := [][]float64{}
			c := [][]float64{}
			prob := [][]float64{}
			xh := [][]float64{}
			cPrev2 := [][]float64{}

			hf = append(hf, hfs[i])
			hi = append(hi, his[i])
			ho = append(ho, hos[i])
			hc = append(hc, hcs[i])
			h = append(h, hs[i])
			c = append(c, cs[i])
			prob = append(prob, probs[i])
			xh = append(xh, xhs[i])
			// if i+1 != len(X) {
			// 	cPrev2 = append(cPrev2, cPrevs[i+1])
			// } else {
			// 	cPrev2 = append(cPrev2, cPrevs[i])
			// }
			cPrev2 = append(cPrev2, cPrevs[i])

			dhNext, dcNext = lstmBackward(hf, hi, ho, hc, h, c, prob, dhNext, dcNext, cPrev2, xh)
		}

	}

	// dumm2 := [][]float64{{1, 0, 0, 0, 0}}
	// outDumm, _ := lstmCell(dumm2, dumm1, dumm1)
	// fmt.Println("Err: ", checkErr)
	for i := 0; i < 10; i++ {
		testPower(qcPrev2, qhPrev2)
	}

}

func testPower(cPrev, hPrev [][]float64) {
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	if scanner.Err() != nil {
		log.Fatalln(scanner.Err())
	}
	txt := scanner.Text()
	x := [][]float64{{0, 0, 0, 0, 0}}
	// there should be 5 vectors here "Jane0 saw1 Doug2 Spot3 .4"
	switch txt {
	case "jane":
		x[0][0] = 1
		x[0][1] = 0
		x[0][2] = 0
		x[0][3] = 0
		x[0][4] = 0
	case "saw":
		x[0][0] = 0
		x[0][1] = 1
		x[0][2] = 0
		x[0][3] = 0
		x[0][4] = 0
	case "doug":
		x[0][0] = 0
		x[0][1] = 0
		x[0][2] = 1
		x[0][3] = 0
		x[0][4] = 0
	case "spot":
		x[0][0] = 0
		x[0][1] = 0
		x[0][2] = 0
		x[0][3] = 1
		x[0][4] = 0
	case ".":
		x[0][0] = 0
		x[0][1] = 0
		x[0][2] = 0
		x[0][3] = 0
		x[0][4] = 1
	default:
		x[0][0] = 0
		x[0][1] = 0
		x[0][2] = 0
		x[0][3] = 0
		x[0][4] = 1
	}

	fmt.Println("txt:", x)

	xh := tambah(x, hPrev)
	h, _, _, _, _, _ := lstmForward(xh, cPrev)

	max := 0.00
	index := 0

	for _, v := range h {
		for i, v2 := range v {
			if i == 0 {
				max = v2
			} else {
				if max < v2 {
					max = v2
					index = i
				}
			}
			// fmt.Println("txt:", v2)
			// fmt.Println("max:", max)
		}
		// fmt.Println("txt:", txt)
		// fmt.Println("iterate h:", v)
		// fmt.Println("iterate h0:", v[0])
		// fmt.Println("iterate x0:", x[0][0])
	}

	switch index {
	case 0:
		fmt.Println("jane")
	case 1:
		fmt.Println("saw")
	case 2:
		fmt.Println("doug")
	case 3:
		fmt.Println("spot")
	case 4:
		fmt.Println(".")
	default:
		fmt.Println("lol")
	}

}

func initWeight(in [][]float64, out [][]float64) {
	wi = randomizeValue(len(in[0]), len(out[0]))
	bi = zeros(len(out), len(out[0]))

	// get the dimensions of matrix
	// init neural network weights
	wf = randomizeValue(len(in[0]), len(out[0]))
	bf = zeros(len(out), len(out[0]))

	wo = randomizeValue(len(in[0]), len(out[0]))
	bo = zeros(len(out), len(out[0]))

	wc = randomizeValue(len(in[0]), len(out[0]))
	bc = zeros(len(out), len(out[0]))
}

func lstmForward(xh, cPrev [][]float64) ([][]float64, [][]float64, [][]float64, [][]float64, [][]float64, [][]float64) {
	hf := forgetGate(xh)
	hi := inputGate(xh)
	ho := outputGate(xh)
	hc := candidateGate(xh)
	c1 := darab(hf, cPrev)
	c2 := darab(hi, hc)
	c := tambah(c1, c2)
	h := darab(ho, tanH(c))

	// fmt.Println("h: ", h)

	return h, c, hf, hi, ho, hc
}

func lstmBackward(hf, hi, ho, hc, h, c, prob, dhNext, dcNext, cPrev, xh [][]float64) ([][]float64, [][]float64) {

	// # Note we're adding dh_next here
	dh := prob
	dh = tambah(dh, dhNext)

	// # Gradient for ho in h = ho * tanh(c)
	dho := darab(tanH(c), dh)
	dho = darab(sigmoidDeriv(ho), dho)

	// # Gradient for c in h = ho * tanh(c), note we're adding dc_next here
	dc := darab(ho, dh)
	dc = darab(dc, tanHDeriv(c))
	dc = tambah(dc, dcNext)

	// # Gradient for hf in c = hf * c_old + hi * hc
	dhf := darab(cPrev, dc)
	dhf = darab(sigmoidDeriv(hf), dhf)

	// # Gradient for hi in c = hf * c_old + hi * hc
	dhi := darab(hc, dc)
	dhi = darab(sigmoidDeriv(hi), dhi)

	// # Gradient for hc in c = hf * c_old + hi * hc
	dhc := darab(hi, dc)
	dhc = darab(tanHDeriv(hc), dhc)

	//  Gate gradients, just a normal fully connected layer gradient
	dWf := dot(transpose(xh), dhf)
	dbf := dhf
	// dXf := dot(dhf, transpose(wf))

	dWi := dot(transpose(xh), dhi)
	dbi := dhi
	// dXi := dot(dhi, transpose(wi))

	dWo := dot(transpose(xh), dho)
	dbo := dho
	// dXo := dot(dho, transpose(wo))

	dWc := dot(transpose(xh), dhc)
	dbc := dhc
	// dXc := dot(dhc, transpose(wc))

	// # As X was used in multiple gates, the gradient must be accumulated here
	// dX = dXo + dXc + dXi + dXf
	// # Split the concatenated X, so that we get our gradient of h_old
	// dh_next = dX[:, :H]
	// dhNext = tambah(dXo, dXc)
	// dhNext = tambah(dhNext, dXi)
	dhNext = dh
	dcNext = darab(hf, dc)

	wf = dWf
	wi = dWi
	wc = dWc
	wo = dWo
	bf = dbf
	bi = dbi
	bc = dbc
	bo = dbo

	return dhNext, dcNext
}

func forgetGate(xh [][]float64) [][]float64 {
	hf := dot(xh, wf)
	hf = tambah(hf, bf)
	return sigmoid(hf)
}

func inputGate(xh [][]float64) [][]float64 {
	hi := dot(xh, wi)
	hi = tambah(hi, bi)

	return sigmoid(hi)
}

func outputGate(xh [][]float64) [][]float64 {
	ho := dot(xh, wo)
	ho = tambah(ho, bo)

	return sigmoid(ho)
}

func candidateGate(xh [][]float64) [][]float64 {
	hc := dot(xh, wc)
	hc = tambah(hc, bc)

	return tanH(hc)
}

func randomizeValue(r int, c int) [][]float64 {
	//we are seeding the rand variable with present time
	//so that we would get different output each time
	// rand.Seed(time.Now().UnixNano())
	// OR WE CAN JUST CONSTANT IT FOR NOW!
	rand.Seed(0)

	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v[i][j] = 2*rand.Float64() - 1
		}
	}

	return v
}

func zeros(r int, c int) [][]float64 {
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v[i][j] = 0
		}
	}

	return v
}

func zerosAlpha(r int, c int) [][]float64 {
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v[i][j] = 0.5
		}
	}

	return v
}

func sigmoidDeriv(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = v2 * (1 - v2)
		}
	}

	return output
}

func sigmoid(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			var nX float64
			nX = 0 - v2
			output[i][i2] = 1 / (1 + math.Exp(nX))
		}
	}

	return output
}

func tanH(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = math.Tanh(v2)
		}
	}

	return output
}

func tanHDeriv(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = 1 - math.Pow(math.Tanh(v2), 2)
		}
	}

	return output
}

func dot(m1 [][]float64, m2 [][]float64) [][]float64 {

	// Ref 2d slice
	// https://gobyexample.com/slices
	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m2[0]))
	}

	// outR := len(m1)
	// outC := len(m2[0])

	for outR, v := range m1 {
		for outC := range m2[0] {
			output[outR][outC] = 0
			for i2, v2 := range v {
				output[outR][outC] += v2 * m2[i2][outC]
			}
		}
	}

	return output
}

func darab(m1 [][]float64, m2 [][]float64) [][]float64 {

	product := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		product[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			product[i][i2] = v2 * m2[i][i2]
		}
	}

	return product
}

func tambah(m1 [][]float64, m2 [][]float64) [][]float64 {

	sum := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		sum[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			sum[i][i2] = v2 + m2[i][i2]
		}
	}

	return sum
}

func tolak(m1 [][]float64, m2 [][]float64) [][]float64 {

	difference := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		difference[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			difference[i][i2] = v2 - m2[i][i2]
		}
	}

	return difference
}

func transpose(m1 [][]float64) [][]float64 {

	mT := make([][]float64, len(m1[0]))
	for i := 0; i < len(m1[0]); i++ {
		mT[i] = make([]float64, len(m1))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			mT[i2][i] = v2
		}
	}

	return mT
}

func softmax(m1 [][]float64) [][]float64 {

	sumZExp := 0.00
	s := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		s[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			s[i][i2] = math.Exp(v2)
		}
	}

	for _, v := range s {
		for _, v2 := range v {
			sumZExp += v2
		}
	}

	for i, v := range s {
		for i2, v2 := range v {
			s[i][i2] = v2 / sumZExp
		}
	}

	return s
}

/*
https://en.wikipedia.org/wiki/Softmax_function
>>> import math
>>> z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
>>> z_exp = [math.exp(i) for i in z]
>>> print([round(i, 2) for i in z_exp])
[2.72, 7.39, 20.09, 54.6, 2.72, 7.39, 20.09]
>>> sum_z_exp = sum(z_exp)
>>> print(round(sum_z_exp, 2))
114.98
>>> softmax = [i / sum_z_exp for i in z_exp]
>>> print([round(i, 3) for i in softmax])
[0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]

*/
