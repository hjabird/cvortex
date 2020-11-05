#include "libcvtx.h"
/*============================================================================
RedistributionFunc.cpp

Common functions used to redistribute vortex particles.

Copyright(c) 2019-2020 HJA Bird

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

/* 
Reference Winckelmans et al. 2005, 
Vortex_methods_and_their_applications_to_trailing_wake_vortex_sims,
C.R. Physique 6
*/

#include <cassert>

static float lambda0(float U) {
	assert(U >= 0.f);
	return U < 0.5f ? 1.f : 0.f;
}

CVTX_EXPORT const cvtx_RedistFunc cvtx_RedistFunc_lambda0(void) {
	cvtx_RedistFunc g0;
	g0.func = lambda0;
	g0.radius = 0.5f;
	return g0;
}

static float lambda1(float U) {
	assert(U >= 0.f);
	return U <= 1.f ? 1.f - U : 0.f;
}

CVTX_EXPORT const cvtx_RedistFunc cvtx_RedistFunc_lambda1(void) {
	cvtx_RedistFunc g1;
	g1.func = lambda1;
	g1.radius = 1.0f;
	return g1;
}

static float lambda2(float U) {
	assert(U >= 0.f);
	return U < 0.5f ? 1.f - U*U : (U<1.5f ? 0.5f*(1.f-U)*(2.f-U) : 0.f);
}

CVTX_EXPORT const cvtx_RedistFunc cvtx_RedistFunc_lambda2(void) {
	cvtx_RedistFunc g2;
	g2.func = lambda2;
	g2.radius = 1.5f;
	return g2;
}

static float lambda3(float U) {
	assert(U >= 0);
	return U < 1.f ? 0.5f*(1.f-U*U)*(2.f-U) : 
		(U < 2.f ? (1.f/6.f)*(1.f - U)*(2.f - U)*(3.f - U) : 0.f);
}

CVTX_EXPORT const cvtx_RedistFunc cvtx_RedistFunc_lambda3(void) {
	cvtx_RedistFunc g3;
	g3.func = lambda3;
	g3.radius = 2.0f;
	return g3;
}

static float m4p(float U) {
	assert(U >= 0.f);
	return U < 1.f ? 1.f - 2.5f * U * U + 1.5f * U * U * U :
		(U < 2.f ? 0.5f * (1.f - U) * (2.f - U) * (2.f - U) : 0.f);
}

CVTX_EXPORT const cvtx_RedistFunc cvtx_RedistFunc_m4p(void) {
	cvtx_RedistFunc m4;
	m4.func = m4p;
	m4.radius = 2.0f;
	return m4;
}
