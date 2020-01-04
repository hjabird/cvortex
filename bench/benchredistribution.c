#include "benchredistribution.h"
/*============================================================================
benchredistribution.c

Benchmark particle redistribution for cvortex.

Copyright(c) 2018-2019 HJA Bird

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

#include "libcvtx.h"
#include "bencharraysetup.h"
#include "benchtools.h"

static float meshsize;

void redistribution_3D_m4(int);
void redistribution_3D_lambda0(int);
void redistribution_3D_lambda1(int);
void redistribution_3D_lambda2(int);
void redistribution_3D_lambda3(int);

void redistribution_2D_m4(int);
void redistribution_2D_lambda0(int);
void redistribution_2D_lambda1(int);
void redistribution_2D_lambda2(int);
void redistribution_2D_lambda3(int);

void run_redistribution_tests(void) {
	create_particles_3D(1000000, 1., 8e-6);
	create_particles_3D_outarr(8 * 1000000);
	meshsize = cbrt(2. / 1000.);
	BENCH("P3D redistribute-m4p vsmall", redistribution_3D_m4, test_repeats(), 1000);
	BENCH("P3D redistribute-lambda0 vsmall", redistribution_3D_lambda0, test_repeats(), 1000);
	BENCH("P3D redistribute-lambda1 vsmall", redistribution_3D_lambda1, test_repeats(), 1000);
	BENCH("P3D redistribute-lambda2 vsmall", redistribution_3D_lambda2, test_repeats(), 1000);
	BENCH("P3D redistribute-lambda3 vsmall", redistribution_3D_lambda3, test_repeats(), 1000);
	meshsize = cbrt(2. / 10000.);
	BENCH("P3D redistribute-m4p small", redistribution_3D_m4, test_repeats(), 10000);
	BENCH("P3D redistribute-lambda0 small", redistribution_3D_lambda0, test_repeats(), 10000);
	BENCH("P3D redistribute-lambda1 small", redistribution_3D_lambda1, test_repeats(), 10000);
	BENCH("P3D redistribute-lambda2 small", redistribution_3D_lambda2, test_repeats(), 10000);
	BENCH("P3D redistribute-lambda3 small", redistribution_3D_lambda3, test_repeats(), 10000);
	meshsize = cbrt(2. / 80000.);
	BENCH("P3D redistribute-m4p medium", redistribution_3D_m4, test_repeats(), 80000);
	BENCH("P3D redistribute-lambda0 medium", redistribution_3D_lambda0, test_repeats(), 80000);
	BENCH("P3D redistribute-lambda1 medium", redistribution_3D_lambda1, test_repeats(), 80000);
	BENCH("P3D redistribute-lambda2 medium", redistribution_3D_lambda2, test_repeats(), 80000);
	BENCH("P3D redistribute-lambda3 medium", redistribution_3D_lambda3, test_repeats(), 80000);
	meshsize = cbrt(2. / 250000.);
	BENCH("P3D redistribute-m4p large", redistribution_3D_m4, test_repeats(), 250000);
	BENCH("P3D redistribute-lambda1 large", redistribution_3D_lambda1, test_repeats(), 250000);
	BENCH("P3D redistribute-lambda0 large", redistribution_3D_lambda0, test_repeats(), 250000);
	BENCH("P3D redistribute-lambda2 large", redistribution_3D_lambda2, test_repeats(), 250000);
	BENCH("P3D redistribute-lambda3 large", redistribution_3D_lambda3, test_repeats(), 250000);
	meshsize = cbrt(2. / 500000.);
	BENCH("P3D redistribute-m4p vlarge", redistribution_3D_m4, test_repeats(), 500000);
	BENCH("P3D redistribute-lambda0 vlarge", redistribution_3D_lambda0, test_repeats(), 500000);
	BENCH("P3D redistribute-lambda1 vlarge", redistribution_3D_lambda1, test_repeats(), 500000);
	BENCH("P3D redistribute-lambda2 vlarge", redistribution_3D_lambda2, test_repeats(), 500000);
	BENCH("P3D redistribute-lambda3 vlarge", redistribution_3D_lambda3, test_repeats(), 500000);
	meshsize = cbrt(2. / 1e6);
	BENCH("P3D redistribute-m4p huge", redistribution_3D_m4, test_repeats(), 1000000);
	BENCH("P3D redistribute-lambda0 huge", redistribution_3D_lambda0, test_repeats(), 1000000);
	BENCH("P3D redistribute-lambda1 huge", redistribution_3D_lambda1, test_repeats(), 1000000);
	BENCH("P3D redistribute-lambda2 huge", redistribution_3D_lambda2, test_repeats(), 1000000);
	BENCH("P3D redistribute-lambda3 huge", redistribution_3D_lambda3, test_repeats(), 1000000);
	destroy_particles_3D();
	destroy_oparticles_3D();

	create_particles_2D(1000000, 1., 4e-6);
	create_particles_2D_outarr(8 * 1000000);
	meshsize = cbrt(2. / 1000.);
	BENCH("P2D redistribute-m4p vsmall", redistribution_2D_m4, test_repeats(), 1000);
	BENCH("P2D redistribute-lambda0 vsmall", redistribution_2D_lambda0, test_repeats(), 1000);
	BENCH("P2D redistribute-lambda1 vsmall", redistribution_2D_lambda1, test_repeats(), 1000);
	BENCH("P2D redistribute-lambda2 vsmall", redistribution_2D_lambda2, test_repeats(), 1000);
	BENCH("P2D redistribute-lambda3 vsmall", redistribution_2D_lambda3, test_repeats(), 1000);
	meshsize = cbrt(2. / 10000.);
	BENCH("P2D redistribute-m4p small", redistribution_2D_m4, test_repeats(), 10000);
	BENCH("P2D redistribute-lambda0 small", redistribution_2D_lambda0, test_repeats(), 10000);
	BENCH("P2D redistribute-lambda1 small", redistribution_2D_lambda1, test_repeats(), 10000);
	BENCH("P2D redistribute-lambda2 small", redistribution_2D_lambda2, test_repeats(), 10000);
	BENCH("P2D redistribute-lambda3 small", redistribution_2D_lambda3, test_repeats(), 10000);
	meshsize = cbrt(2. / 80000.);
	BENCH("P2D redistribute-m4p medium", redistribution_2D_m4, test_repeats(), 80000);
	BENCH("P2D redistribute-lambda0 medium", redistribution_2D_lambda0, test_repeats(), 80000);
	BENCH("P2D redistribute-lambda1 medium", redistribution_2D_lambda1, test_repeats(), 80000);
	BENCH("P2D redistribute-lambda2 medium", redistribution_2D_lambda2, test_repeats(), 80000);
	BENCH("P2D redistribute-lambda3 medium", redistribution_2D_lambda3, test_repeats(), 80000);
	meshsize = cbrt(2. / 250000.);
	BENCH("P2D redistribute-m4p large", redistribution_2D_m4, test_repeats(), 250000);
	BENCH("P2D redistribute-lambda1 large", redistribution_2D_lambda1, test_repeats(), 250000);
	BENCH("P2D redistribute-lambda0 large", redistribution_2D_lambda0, test_repeats(), 250000);
	BENCH("P2D redistribute-lambda2 large", redistribution_2D_lambda2, test_repeats(), 250000);
	BENCH("P2D redistribute-lambda3 large", redistribution_2D_lambda3, test_repeats(), 250000);
	meshsize = cbrt(2. / 500000.);
	BENCH("P2D redistribute-m4p vlarge", redistribution_2D_m4, test_repeats(), 500000);
	BENCH("P2D redistribute-lambda0 vlarge", redistribution_2D_lambda0, test_repeats(), 500000);
	BENCH("P2D redistribute-lambda1 vlarge", redistribution_2D_lambda1, test_repeats(), 500000);
	BENCH("P2D redistribute-lambda2 vlarge", redistribution_2D_lambda2, test_repeats(), 500000);
	BENCH("P2D redistribute-lambda3 vlarge", redistribution_2D_lambda3, test_repeats(), 500000);
	meshsize = cbrt(2. / 1e6);
	BENCH("P2D redistribute-m4p huge", redistribution_2D_m4, test_repeats(), 1000000);
	BENCH("P2D redistribute-lambda0 huge", redistribution_2D_lambda0, test_repeats(), 1000000);
	BENCH("P2D redistribute-lambda1 huge", redistribution_2D_lambda1, test_repeats(), 1000000);
	BENCH("P2D redistribute-lambda2 huge", redistribution_2D_lambda2, test_repeats(), 1000000);
	BENCH("P2D redistribute-lambda3 huge", redistribution_2D_lambda3, test_repeats(), 1000000);
	destroy_particles_2D();
	destroy_oparticles_2D();
	return;
}

/* 3D -----------------------------------------------------------------------*/
/* Benchmark the M4 distribution for randomally distributed particles in 3D. */
void redistribution_3D_m4(int probsz) {
	cvtx_RedistFunc rfunc = cvtx_RedistFunc_m4p();
	/* Call function. */
	cvtx_P3D_redistribute_on_grid(
		particle_3D_pptr(),
		probsz,
		oparticle_3D_ptr(),
		4 * probsz,		/* Set to resultant num particles.   */
		&rfunc,
		meshsize,
		1e-4);
}

/* Benchmark the lambda 0 distribution for randomally distributed particles in 3D. */
void redistribution_3D_lambda0(int probsz) {
	cvtx_RedistFunc rfunc = cvtx_RedistFunc_lambda0();
	/* Call function. */
	cvtx_P3D_redistribute_on_grid(
		particle_3D_pptr(),
		probsz,
		oparticle_3D_ptr(),
		4 * probsz,		/* Set to resultant num particles.   */
		&rfunc,
		meshsize,
		1e-4);
}

/* Benchmark the lambda1 distribution for randomally distributed particles in 3D. */
void redistribution_3D_lambda1(int probsz) {
	cvtx_RedistFunc rfunc = cvtx_RedistFunc_lambda1();
	/* Call function. */
	cvtx_P3D_redistribute_on_grid(
		particle_3D_pptr(),
		probsz,
		oparticle_3D_ptr(),
		4 * probsz,		/* Set to resultant num particles.   */
		&rfunc,
		meshsize,
		1e-4);
}

/* Benchmark the lambda2 distribution for randomally distributed particles in 3D. */
void redistribution_3D_lambda2(int probsz) {
	cvtx_RedistFunc rfunc = cvtx_RedistFunc_lambda2();
	/* Call function. */
	cvtx_P3D_redistribute_on_grid(
		particle_3D_pptr(),
		probsz,
		oparticle_3D_ptr(),
		4 * probsz,		/* Set to resultant num particles.   */
		&rfunc,
		meshsize,
		1e-4);
}

/* Benchmark the lambda3 distribution for randomally distributed particles in 3D. */
void redistribution_3D_lambda3(int probsz) {
	cvtx_RedistFunc rfunc = cvtx_RedistFunc_lambda3();
	/* Call function. */
	cvtx_P3D_redistribute_on_grid(
		particle_3D_pptr(),
		probsz,
		oparticle_3D_ptr(),
		4 * probsz,		/* Set to resultant num particles.   */
		&rfunc,
		meshsize,
		1e-4);
}

/* 2D redistribution --------------------------------------------------------*/

/* Benchmark the M4 distribution for randomally distributed particles in 3D. */
void redistribution_2D_m4(int probsz) {
	cvtx_RedistFunc rfunc = cvtx_RedistFunc_m4p();
	/* Call function. */
	cvtx_P2D_redistribute_on_grid(
		particle_2D_pptr(),
		probsz,
		oparticle_2D_ptr(),
		4 * probsz,		/* Set to resultant num particles.   */
		&rfunc,
		meshsize,
		1e-4);
}

/* Benchmark the lambda 0 distribution for randomally distributed particles in 2D. */
void redistribution_2D_lambda0(int probsz) {
	cvtx_RedistFunc rfunc = cvtx_RedistFunc_lambda0();
	/* Call function. */
	cvtx_P2D_redistribute_on_grid(
		particle_2D_pptr(),
		probsz,
		oparticle_2D_ptr(),
		4 * probsz,		/* Set to resultant num particles.   */
		&rfunc,
		meshsize,
		1e-4);
}

/* Benchmark the lambda1 distribution for randomally distributed particles in 2D. */
void redistribution_2D_lambda1(int probsz) {
	cvtx_RedistFunc rfunc = cvtx_RedistFunc_lambda1();
	/* Call function. */
	cvtx_P2D_redistribute_on_grid(
		particle_2D_pptr(),
		probsz,
		oparticle_2D_ptr(),
		4 * probsz,		/* Set to resultant num particles.   */
		&rfunc,
		meshsize,
		1e-4);
}

/* Benchmark the lambda2 distribution for randomally distributed particles in 2D. */
void redistribution_2D_lambda2(int probsz) {
	cvtx_RedistFunc rfunc = cvtx_RedistFunc_lambda2();
	/* Call function. */
	cvtx_P2D_redistribute_on_grid(
		particle_2D_pptr(),
		probsz,
		oparticle_2D_ptr(),
		4 * probsz,		/* Set to resultant num particles.   */
		&rfunc,
		meshsize,
		1e-4);
}

/* Benchmark the lambda3 distribution for randomally distributed particles in 2D. */
void redistribution_2D_lambda3(int probsz) {
	cvtx_RedistFunc rfunc = cvtx_RedistFunc_lambda3();
	/* Call function. */
	cvtx_P2D_redistribute_on_grid(
		particle_2D_pptr(),
		probsz,
		oparticle_2D_ptr(),
		4 * probsz,		/* Set to resultant num particles.   */
		&rfunc,
		meshsize,
		1e-4);
}
