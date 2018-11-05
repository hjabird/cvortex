#include "VortFunc.h"
/*============================================================================
VortFunc.c

Common functions used to regularise vortex particles.

Copyright(c) 2018 HJA Bird

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

#include <math.h>

static float g_singular(float rho) {
	return 1;
}

static float zeta_singular(float rho) {
	return 0;
}

static void combined_singular(float rho, float* reduction, float* vort_frac) {
	*reduction = 1;
	*vort_frac = 0;
	return;
}

static float g_winckel(float rho) {
	float a, b, c, d;
	a = rho * rho + (float)2.5;
	b = a * rho * rho * rho;
	c = rho * rho + 1;
	d = (b / powf(c, 2.5));
	return d;
}

static float zeta_winckel(float rho) {
	float a, b, c;
	a = rho * rho + 1;
	b = powf(a, 3.5);
	c = 7.5 /b;
	return c;
}

static void combined_winckel(float rho, float* g, float* zeta) {
	*g = g_winckel(rho);
	*zeta = zeta_winckel(rho);
	return;
}


const cvtx_VortFunc cvtx_VortFunc_singular(void)
{
	cvtx_VortFunc ret;
	ret.g_fn = &g_singular;
	ret.zeta_fn = &zeta_singular;
	ret.combined_fn = &combined_singular;
	return ret;
}

const cvtx_VortFunc cvtx_VortFunc_winckelmans(void)
{
	cvtx_VortFunc ret;
	ret.g_fn = &g_winckel;
	ret.zeta_fn = &zeta_winckel;
	ret.combined_fn = &combined_winckel;
	return ret;
}
