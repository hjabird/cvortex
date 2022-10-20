#include "libcvtx.h"
/*============================================================================
VortFunc.c

Fast good-enough maths functions.

Copyright(c) HJA Bird

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
============================================================================*/
#ifndef CVTX_FAST_MATHS_H
#define CVTX_FAST_MATHS_H

namespace cvtx {
namespace maths {

template<int ExpNumerator, int ExpDenominator = 1>
constexpr float pow(float x);

template<>
constexpr float pow<1, 1>(float x) {
  return x;
}

template<>
constexpr float pow<2, 1>(float x) {
  return x * x;
}

template <>
constexpr float pow<3, 1>(float x) {
  return x * x * x;
}

template <>
constexpr float pow<4, 1>(float x) {
  return pow<2>(x) * pow<2>(x);
}

template <>
constexpr float pow<5, 1>(float x) {
  return pow<4>(x) * x;
}

template <>
constexpr float pow<6, 1>(float x) {
  return pow<3>(x) * pow<3>(x);
}

template <>
constexpr float pow<7, 1>(float x) {
  return pow<6>(x) * x;
}

template <>
constexpr float pow<8, 1>(float x) {
  return pow<4>(x) * pow<4>(x);
}

template <>
constexpr float pow<9, 1>(float x) {
  return pow<3>(x) * pow<3>(x) * pow<3>(x);
}

}	// namespace maths
}	// namespace cvtx
#endif  // CVTX_FAST_MATHS_H