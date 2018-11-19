package main

import (
	"fmt"
	"math"
	"math/rand"
)

// X is inputs
var X = [][]float64{
	{1, 2},
	{0.5, 3},
}

var Y = [][]float64{
	{0.5},
	{1.25},
}

var Wa = [][]float64{
	{0.45},
	{0.25},
}

var Wi = [][]float64{
	{0.95},
	{0.8},
}

var Wf = [][]float64{
	{0.7},
	{0.45},
}

var Wo = [][]float64{
	{0.6},
	{0.4},
}

var Ua = [][]float64{
	{0.15},
}

var Ui = [][]float64{
	{0.8},
}

var Uf = [][]float64{
	{0.1},
}

var Uo = [][]float64{
	{0.25},
}

var ba = [][]float64{
	{0.2},
}

var bi = [][]float64{
	{0.65},
}

var bf = [][]float64{
	{0.15},
}

var bo = [][]float64{
	{0.1},
}

var alpha = 0.1
var layers = 2
var trains = 1

func main() {

	alphaW := zerosAlpha(2, 1)
	alphaU := zerosAlpha(1, 1)
	alphaB := zerosAlpha(1, 1)

	for n := 0; n < trains; n++ {

		outPrevs := [][]float64{{0}}
		statePrevs := [][]float64{{0}}
		states := [][]float64{{0}}
		outs := [][]float64{}
		as := [][]float64{}
		is := [][]float64{}
		fs := [][]float64{}
		os := [][]float64{}

		for i, v := range X {
			x := [][]float64{}
			x = append(x, v)
			outPrevsI := [][]float64{}
			outPrevsI = append(outPrevsI, outPrevs[i])
			a := candidateGate(x, outPrevsI)
			ig := inputGate(x, outPrevsI)
			f := forgetGate(x, outPrevsI)
			o := outputGate(x, outPrevsI)
			fmt.Println("a: ", a[0])
			fmt.Println("i: ", ig[0])
			fmt.Println("f: ", f[0])
			fmt.Println("o: ", o[0])

			statePrevsI := [][]float64{}
			statePrevsI = append(statePrevsI, statePrevs[i])
			c1 := darab(a, ig)
			c2 := darab(f, statePrevsI)
			c := tambah(c1, c2)
			out := darab(o, tanH(c))
			fmt.Println("state: ", c[0])
			fmt.Println("out: ", out[0])

			outPrevs = append(outPrevs, out[0])
			statePrevs = append(statePrevs, c[0])
			outs = append(outs, out[0])
			as = append(as, a[0])
			is = append(is, ig[0])
			fs = append(fs, f[0])
			os = append(os, o[0])
			states = append(states, c[0])
		}

		triOutAtts := [][]float64{{0}}
		BPDeltaStates := [][]float64{{0}}
		// Append 0s to future f
		fs = append(fs, BPDeltaStates[0])

		gateCompilations := zeros(layers, 4)

		i2 := 0
		// OR
		// newI = layers - i - 1

		dWa := zeros(1, 2)
		dWi := zeros(1, 2)
		dWf := zeros(1, 2)
		dWo := zeros(1, 2)

		dUa := zeros(1, 1)
		dUi := zeros(1, 1)
		dUf := zeros(1, 1)
		dUo := zeros(1, 1)

		dBa := zeros(1, 1)
		dBi := zeros(1, 1)
		dBf := zeros(1, 1)
		dBo := zeros(1, 1)

		for i := layers - 1; i >= 0; i-- {

			fmt.Println("######################################################################################")

			out := [][]float64{}
			out = append(out, outs[i])
			y := [][]float64{}
			y = append(y, Y[i])
			// triangle delta @ t
			triAtt := tolak(out, y)
			fmt.Println("triangle@t: ", triAtt[0])

			// triangle out @ t
			triOutAtt := [][]float64{}
			triOutAtt = append(triOutAtt, triOutAtts[i2])
			fmt.Println("triangleOut@t: ", triOutAtt[0])
			dOut := tambah(triAtt, triOutAtt)
			fmt.Println("dOut: ", dOut[0])

			o := [][]float64{}
			o = append(o, os[i])

			state := [][]float64{}
			state = append(state, states[i+1])

			BPDeltaStatesI := [][]float64{}
			BPDeltaStatesI = append(BPDeltaStatesI, BPDeltaStates[i2])

			fFuture := [][]float64{}
			fFuture = append(fFuture, fs[i+1])

			dState := darab(dOut, o)
			dState = darab(dState, tanHDeriv(state))
			dState = tambah(dState, darab(BPDeltaStatesI, fFuture))
			fmt.Println("dState: ", dState[0])

			ig := [][]float64{}
			ig = append(ig, is[i])

			a := [][]float64{}
			a = append(a, as[i])

			dA := darab(dState, ig)
			dA = darab(dA, oneMinusSquare(a))
			fmt.Println("dA: ", dA[0])

			dI := darab(dState, a)
			dI = darab(dI, sigmoidDeriv(ig))
			fmt.Println("dI: ", dI[0])

			statePrev := [][]float64{}
			statePrev = append(statePrev, states[i])

			f := [][]float64{}
			f = append(f, fs[i])

			fmt.Println("states: ", states)
			fmt.Println("i-1: ", i-1)
			fmt.Println("state-1: ", statePrev)
			dF := darab(dState, statePrev)
			dF = darab(dF, sigmoidDeriv(f))
			fmt.Println("dF: ", dF[0])

			dO := darab(dOut, tanH(state))
			dO = darab(dO, sigmoidDeriv(o))
			fmt.Println("dO: ", dO[0])

			cWs := compileWs()
			cUs := compileUs()
			cGs := compileGates(dA, dI, dF, dO)

			dX := dot(cWs, cGs)
			fmt.Println("dX: ", dX)

			dOutNext := dot(cUs, cGs)
			fmt.Println("dOutNext: ", dOutNext)

			BPDeltaStates = append(BPDeltaStates, dState[0])
			triOutAtts = append(triOutAtts, dOutNext[0])

			for i3, v := range cGs {
				gateCompilations[i2][i3] = v[0]
			}

			x := [][]float64{}
			x = append(x, X[i])
			dWa = tambah(dWa, dot(dA, x))
			dWi = tambah(dWi, dot(dI, x))
			dWf = tambah(dWf, dot(dF, x))
			dWo = tambah(dWo, dot(dO, x))

			if i != 0 {
				outPrev := [][]float64{}
				outPrev = append(outPrev, outs[i-1])
				dUa = tambah(dUa, dot(dA, outPrev))
				dUi = tambah(dUi, dot(dI, outPrev))
				dUf = tambah(dUf, dot(dF, outPrev))
				dUo = tambah(dUo, dot(dO, outPrev))
			}

			dBa = tambah(dBa, dA)
			dBi = tambah(dBi, dI)
			dBf = tambah(dBf, dF)
			dBo = tambah(dBo, dO)

			i2++
		}

		Wa = tolak(Wa, darab(alphaW, transpose(dWa)))
		Wi = tolak(Wi, darab(alphaW, transpose(dWi)))
		Wf = tolak(Wf, darab(alphaW, transpose(dWf)))
		Wo = tolak(Wo, darab(alphaW, transpose(dWo)))

		Ua = tolak(Ua, darab(alphaU, transpose(dUa)))
		Ui = tolak(Ui, darab(alphaU, transpose(dUi)))
		Uf = tolak(Uf, darab(alphaU, transpose(dUf)))
		Uo = tolak(Uo, darab(alphaU, transpose(dUo)))

		ba = tolak(ba, darab(alphaB, transpose(dBa)))
		bi = tolak(bi, darab(alphaB, transpose(dBi)))
		bf = tolak(bf, darab(alphaB, transpose(dBf)))
		bo = tolak(bo, darab(alphaB, transpose(dBo)))

		fmt.Println("Wa: ", Wa)
		fmt.Println("Wi: ", Wi)
		fmt.Println("Wf: ", Wf)
		fmt.Println("Wo: ", Wo)
		fmt.Println("Ua: ", Ua)
		fmt.Println("Ui: ", Ui)
		fmt.Println("Uf: ", Uf)
		fmt.Println("Uo: ", Uo)
		fmt.Println("Ba: ", ba)
		fmt.Println("Bi: ", bi)
		fmt.Println("Bf: ", bf)
		fmt.Println("Bo: ", bo)

	}

}

func candidateGate(x, out [][]float64) [][]float64 {
	hc1 := dot(x, Wa)
	hc2 := dot(out, Ua)
	hc := tambah(hc1, hc2)
	hc = tambah(hc, ba)

	return tanH(hc)
}

func inputGate(x, out [][]float64) [][]float64 {
	hi1 := dot(x, Wi)
	hi2 := dot(out, Ui)
	hi := tambah(hi1, hi2)
	hi = tambah(hi, bi)

	return sigmoid(hi)
}

func forgetGate(x, out [][]float64) [][]float64 {
	hf1 := dot(x, Wf)
	hf2 := dot(out, Uf)
	hf := tambah(hf1, hf2)
	hf = tambah(hf, bf)

	return sigmoid(hf)
}

func outputGate(x, out [][]float64) [][]float64 {
	ho1 := dot(x, Wo)
	ho2 := dot(out, Uo)
	ho := tambah(ho1, ho2)
	ho = tambah(ho, bo)

	return sigmoid(ho)
}

func compileWs() [][]float64 {
	r := 2
	c := 4
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			// v[i][j] = 0
			switch j {
			case 0:
				v[i][j] = Wa[i][0]
			case 1:
				v[i][j] = Wi[i][0]
			case 2:
				v[i][j] = Wf[i][0]
			case 3:
				v[i][j] = Wo[i][0]
			}
		}
	}

	fmt.Println("compileWs: ", v)

	return v
}

func compileUs() [][]float64 {
	r := 1
	c := 4
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			// v[i][j] = 0
			switch j {
			case 0:
				v[i][j] = Ua[i][0]
			case 1:
				v[i][j] = Ui[i][0]
			case 2:
				v[i][j] = Uf[i][0]
			case 3:
				v[i][j] = Uo[i][0]
			}
		}
	}

	fmt.Println("compileUs: ", v)

	return v
}

func compileGates(a, is, f, o [][]float64) [][]float64 {
	r := 4
	c := 1
	v := make([][]float64, r)
	for i := 0; i < r; i++ {
		v[i] = make([]float64, c)
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			// v[i][j] = 0
			switch i {
			case 0:
				v[i][j] = a[0][0]
			case 1:
				v[i][j] = is[0][0]
			case 2:
				v[i][j] = f[0][0]
			case 3:
				v[i][j] = o[0][0]
			}
		}
	}

	fmt.Println("compileGates: ", v)

	return v
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
			v[i][j] = alpha
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

func oneMinusSquare(m1 [][]float64) [][]float64 {

	output := make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		output[i] = make([]float64, len(m1[0]))
	}

	for i, v := range m1 {
		for i2, v2 := range v {
			output[i][i2] = 1 - math.Pow(v2, 2)
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
