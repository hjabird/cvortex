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

#include <assert.h>
#include <math.h>
#include <stdio.h>

static float warn_bad_eta_fn(float rho){
	static int warned = 0;
	assert(0 && "Vortex regularisation function had no viscous method!");
	if(warned == 0){
		fprintf(stderr, "Tried to calculated perform viscous calculations "
			"with inapproriate vortex regularisation function!\n"
			"Modelling as invicid.\n\n");
		warned = 1;
	}
	return 0;
}

static float g_singular(float rho) {
	return 1;
}

static float zeta_singular(float rho) {
	return 0;
}

static void combined_singular(float rho, float* g, float* zeta) {
	*g = 1;
	*zeta = 0;
	return;
}

static float g_winckel(float rho) {
	float a, b, c, d;
	assert(rho >= 0 && "Rho should not be -ve");
	a = rho * rho + (float)2.5;
	b = a * rho * rho * rho;
	c = rho * rho + 1;
	d = (b / powf(c, 2.5));
	return d;
}

static float zeta_winckel(float rho) {
	float a, b, c;
	assert(rho >= 0 && "Rho should not be -ve");
	a = rho * rho + 1;
	b = powf(a, 3.5);
	c = (float)7.5 / b;
	return c;
}

static float eta_winckel(float rho) {
	float a, b, c;
	assert(rho >= 0 && "Rho should not be -ve");
	a = (float) 52.5;
	b = rho * rho + 1;
	c = powf(b, -4.5);
	return a * c;
}

static void combined_winckel(float rho, float* g, float* zeta) {
	assert(rho >= 0 && "Rho should not be -ve");
	*g = g_winckel(rho);
	*zeta = zeta_winckel(rho);
	return;
}

static float g_planetary(float rho) {
	assert(rho >= 0 && "Rho should not be -ve");
	return rho < (float)1. ? rho * rho * rho : (float)1.;
}

static float zeta_planetary(float rho){
	assert(rho >= 0 && "Rho should not be -ve");
	return rho < (float)1. ? (float)3 : (float)0;
}

static void combined_planetary(float rho, float* g, float* zeta) {
	assert(rho >= 0 && "Rho should not be -ve");
	*g = g_planetary(rho);
	*zeta = zeta_planetary(rho);
	return;
}

static float g_gaussian(float rho){
	/* = 1 to 8sf for rho ~>6. Taylor expansion otherwise */
	assert(rho >= 0 && "Rho should not be -ve");
	if(rho > (float)6.){
		return (float)1.;
	} else {
		const float pi = 3.14159265359f;
		/* Approximate erf using Abramowitz and Stegan 1.7.26 */
		float a1 = 0.3480242f, a2 = -0.0958798f, a3 = 0.7478556f, p = 0.47047f;
		float rho_sr2 = rho / sqrtf(2);
		float t = (float) 1. / (1 + p * rho_sr2);
		float erf = 1.f-t * (a1 + t * (a2 + t * a3)) * expf(-rho_sr2 * rho_sr2);
		float term2 = rho * sqrtf(2 / pi) * expf(-rho_sr2 * rho_sr2);
		return erf - term2;
	}
}

static float zeta_gaussian(float rho){
	assert(rho >= 0 && "Rho should not be -ve");
	const float pi = 3.14159265359f;
	return sqrtf(2 / pi) * expf(-rho * rho / 2);
}

static void combined_gaussian(float rho, float* g, float* zeta) {
	assert(rho >= 0 && "Rho should not be -ve");
	*g = g_gaussian(rho);
	*zeta = zeta_gaussian(rho);
	return;
}

const cvtx_VortFunc EXPORT cvtx_VortFunc_singular(void)
{
	cvtx_VortFunc ret;
	ret.g_fn = &g_singular;
	ret.zeta_fn = &zeta_singular;
	ret.eta_fn = &warn_bad_eta_fn;	/* Not possible for singular vortex */
	ret.combined_fn = &combined_singular;
	return ret;
}

const cvtx_VortFunc EXPORT cvtx_VortFunc_winckelmans(void)
{
	cvtx_VortFunc ret;
	ret.g_fn = &g_winckel;
	ret.zeta_fn = &zeta_winckel;
	ret.eta_fn = eta_winckel;
	ret.combined_fn = &combined_winckel;
	return ret;
}

const cvtx_VortFunc EXPORT cvtx_VortFunc_planetary(void)
{
	cvtx_VortFunc ret;
	ret.g_fn = &g_planetary;
	ret.zeta_fn = &zeta_planetary;
	ret.eta_fn = &warn_bad_eta_fn; /* Not possible for planetary vortex */
	ret.combined_fn = &combined_winckel;
	return ret;
}

const cvtx_VortFunc EXPORT cvtx_VortFunc_gaussian(void){
	cvtx_VortFunc ret;
	ret.g_fn = &g_gaussian;
	ret.zeta_fn = &zeta_gaussian;
	ret.eta_fn = &zeta_gaussian; 
	/* See Winckelmans et al., C. R. Physique 6 (2005), around eq (28) */
	ret.combined_fn = &combined_winckel;
	return ret;
}