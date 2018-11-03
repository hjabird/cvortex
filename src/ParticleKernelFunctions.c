#include "ParticleKernelFunctions.h"
/*============================================================================
ParticleKernelFunctions.c

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

static float vort_red_singular(float rho) {
	return 1;
}

static float vort_frac_singular(float rho) {
	return 0;
}

static void combined_static(float rho, float* reduction, float* vort_frac) {
	*reduction = 1;
	*vort_frac = 0;
	return;
}

static float vort_red_winckel(float rho) {
	float a, b, c, d;
	a = rho * rho + (float)2.5;
	b = a * rho * rho * rho;
	c = rho + 1;
	d = (b / powf(c, 2.5));
	return d;
}

static float vort_frac_winckel(float rho) {
	float a, b, c;
	a = rho * rho + 1;
	b = powf(a, 3.5);
	c = a / b;
	return c;
}

static void combined_winckel(float rho, float* reduction, float* vort_frac) {
	*reduction = vort_red_winckel(rho);
	*vort_frac = vort_frac_winckel(rho);
	return;
}


const cvtx_VortFunc cvtx_ParticleKernalFunctions_singular(void)
{
	cvtx_VortFunc ret;
	ret.reduction_factor_fn = &vort_red_singular;
	ret.vorticity_fraction_fn = &vort_frac_singular;
	return ret;
}

const cvtx_VortFunc cvtx_ParticleKernalFunctions_winckelmans(void)
{
	cvtx_VortFunc ret;
	ret.reduction_factor_fn = &vort_red_winckel;
	ret.vorticity_fraction_fn = &vort_frac_winckel;
	return ret;
}
