#include "redistribution_helper_funcs.h"
/*============================================================================
redistribution_helper_funcs.cpp

Functions that are common to redistributing vortex particles in both 2D
and 3D.

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

#include <cassert>
#include <cmath>
#include <cstdlib>

float get_strength_threshold(
	float* strs, int n_inpt_particles, int n_desired_particles) {

	double minv, maxv, range;
	float fminv, fmaxv, tol = 0.01f, * guesses;
	const int n_guesses = 1024;
	int* g_counts, i, k = 0, interp;
	farray_info(strs, n_inpt_particles, NULL, &fminv, &fmaxv);
	minv = fminv; maxv = fmaxv;
	range = maxv - minv;

	if (n_inpt_particles < n_desired_particles) {
		return (float)(maxv * 1.05);
	}

	/* Guess vorticity thresholds and count the kept particles.*/
	guesses = (float*) malloc(sizeof(float) * n_guesses);
	g_counts = (int*) malloc(sizeof(float) * n_guesses);
	while (1) {
		range = (maxv - minv) * 1.05;
		for (i = 0; i < n_guesses; ++i) {
			g_counts[i] = 0;
			guesses[i] = (float)(minv + i * range / (float)(n_guesses - 1));
		}
		for (i = 0; i < n_inpt_particles; ++i) {
			interp = (int)floor((double)(n_guesses - 1) * (strs[i] - minv) / range);
			if (interp < 0) { g_counts[0]++; }
			else if (interp >= n_guesses) { g_counts[n_guesses - 1]++; }
			else { g_counts[interp]++; }
		}
		interp = g_counts[n_guesses - 1];
		for (i = n_guesses - 2; i >= 0; --i) {
			maxv = guesses[i + 1];
			minv = guesses[i];
			interp += g_counts[i];
			g_counts[i] = interp;
			if (interp > n_desired_particles) {
				k = i + 1;
				break;
			}
		}
		/* Termination condition. */
		if (minv == maxv || g_counts[k] == g_counts[k - 1] ||
			fabs((float)(n_desired_particles - g_counts[k])
				/ ((float)n_desired_particles)) < tol * 0.6) {
			break;
		}
	}
	minv = guesses[k];
	free(guesses);
	free(g_counts);
	return (float)minv;
}

void farray_info(
	float* strs, int n_inpt_partices,
	float* mean, float* min, float* max)
{
	assert(strs != NULL);
	float ave, mi, ma;
	int i;

	ave = 0.f;
	mi = ma = n_inpt_partices > 0 ? strs[0] : 0.f;
	if (mean != NULL) {
#pragma omp parallel for reduction(+: ave)
		for (i = 0; i < n_inpt_partices; ++i) {
			ave += strs[i];
		}
	}
	if (min != NULL || max != NULL) {
		for (i = 0; i < n_inpt_partices; ++i) {
			mi = mi < strs[i] ? mi : strs[i];
			ma = ma > strs[i] ? ma : strs[i];
		}
	}
	ave /= n_inpt_partices;
	if (mean != NULL) { *mean = ave; }
	if (min != NULL) { *min = mi; }
	if (max != NULL) { *max = ma; }
	return;
}
