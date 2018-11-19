package main

import (
	"fmt"
	"math"
	"math/rand"
)

// example1 := "Doug saw Jane."
// example2 := "Jane saw Spot."
// example3 := "Spot saw Doug."

// there should be 5 vectors here "Jane0 saw1 Doug2 Spot3 .4"

// X is inputs
var X = [][]float64{
	{1},
	{2},
}

var Wf = [][]float64{
	{0.7, 0.45},
}

var Y = [][]float64{
	{0},
	{1},
}

var Uf = [][]float64{
	{0.1},
}

var bf = [][]float64{
	{0.15},
}

func main() {
	a1 := tambah(X, Y)
	a1 = dot(Wf, a1)
	a1 = tambah(a1, bf)
	// s1 := dot(Wf, X)
	// s2 := dot(Wf, Y)
	// s3 := tambah(s1, s2)
	// s3 = tambah(s3, bf)
	fmt.Println("result: ", sigmoid(a1))
}

// func forgetGate(xh [][]float64) [][]float64 {
// 	hf := dot(xh, wf)
// 	hf = tambah(hf, bf)
// 	return sigmoid(hf)
// }

// func inputGate(xh [][]float64) [][]float64 {
// 	hi := dot(xh, wi)
// 	hi = tambah(hi, bi)

// 	return sigmoid(hi)
// }

// func outputGate(xh [][]float64) [][]float64 {
// 	ho := dot(xh, wo)
// 	ho = tambah(ho, bo)

// 	return sigmoid(ho)
// }

// func candidateGate(xh [][]float64) [][]float64 {
// 	hc := dot(xh, wc)
// 	hc = tambah(hc, bc)

// 	return tanH(hc)
// }

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
