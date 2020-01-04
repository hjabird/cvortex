#include "benchP2D.h"
/*============================================================================
benchP2D.h

Benchmark 2D particles for cvortex.

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
#include "libcvtx.h"
#include "benchtools.h"
#include "bencharraysetup.h"

void bench_P2D_vel_singular(int np);
void bench_P2D_vel_gaussian(int np);
void bench_P2D_vel_winckelmans(int np);
void bench_P2D_vel_planetary(int np);

void bench_P2D_viscdvort_gaussian(int np);
void bench_P2D_viscdvort_winckelmans(int np);

void run_P2D_bench(void) {
	create_particles_2D(1000000, 10.f, 0.01);
	create_V2f_arr(1000000, 10.f);
	create_V2f_arr2(1000000, 10.f);

	/* Run first with GPUs enabled. */
	int i, n;
	n = cvtx_num_accelerators();
	for (i = 0; i < n; ++i) {
		cvtx_accelerator_enable(i);
	}

	BENCH("P2D vel-singular-gpu vsmall", bench_P2D_vel_singular, test_repeats(), 1000);
	BENCH("P2D vel-gaussian-gpu vsmall", bench_P2D_vel_gaussian, test_repeats(), 1000);
	BENCH("P2D vel-winckelmans-gpu vsmall", bench_P2D_vel_winckelmans, test_repeats(), 1000);
	BENCH("P2D vel-planetary-gpu vsmall", bench_P2D_vel_planetary, test_repeats(), 1000);
	
	BENCH("P2D vel-singular-gpu small", bench_P2D_vel_singular, test_repeats(), 10000);
	BENCH("P2D vel-gaussian-gpu small", bench_P2D_vel_gaussian, test_repeats(), 10000);
	BENCH("P2D vel-winckelmans-gpu small", bench_P2D_vel_winckelmans, test_repeats(), 10000);
	BENCH("P2D vel-planetary-gpu small", bench_P2D_vel_planetary, test_repeats(), 10000);

	BENCH("P2D vel-singular-gpu medium", bench_P2D_vel_singular, test_repeats(), 80000);
	BENCH("P2D vel-gaussian-gpu medium", bench_P2D_vel_gaussian, test_repeats(), 80000);
	BENCH("P2D vel-winckelmans-gpu medium", bench_P2D_vel_winckelmans, test_repeats(), 80000);
	BENCH("P2D vel-planetary-gpu medium", bench_P2D_vel_planetary, test_repeats(), 80000);

	BENCH("P2D vel-singular-gpu large", bench_P2D_vel_singular, test_repeats(), 250000);
	BENCH("P2D vel-gaussian-gpu large", bench_P2D_vel_gaussian, test_repeats(), 250000);
	BENCH("P2D vel-winckelmans-gpu large", bench_P2D_vel_winckelmans, test_repeats(), 250000);
	BENCH("P2D vel-planetary-gpu large", bench_P2D_vel_planetary, test_repeats(), 250000);

	BENCH("P2D vel-singular-gpu vlarge", bench_P2D_vel_singular, test_repeats(), 500000);
	BENCH("P2D vel-gaussian-gpu vlarge", bench_P2D_vel_gaussian, test_repeats(), 500000);
	BENCH("P2D vel-winckelmans-gpu vlarge", bench_P2D_vel_winckelmans, test_repeats(), 500000);
	BENCH("P2D vel-planetary-gpu vlarge", bench_P2D_vel_planetary, test_repeats(), 500000);

	BENCH("P2D vel-singular-gpu huge", bench_P2D_vel_singular, test_repeats(), 1000000);
	BENCH("P2D vel-gaussian-gpu huge", bench_P2D_vel_gaussian, test_repeats(), 1000000);
	BENCH("P2D vel-winckelmans-gpu huge", bench_P2D_vel_winckelmans, test_repeats(), 1000000);
	BENCH("P2D vel-planetary-gpu huge", bench_P2D_vel_planetary, test_repeats(), 1000000);

	BENCH("P2D viscdvort-gaussian-gpu vsmall", bench_P2D_viscdvort_gaussian, test_repeats(), 1000);
	BENCH("P2D viscdvort-winckelmans-gpu vsmall", bench_P2D_viscdvort_winckelmans, test_repeats(), 1000);
	BENCH("P2D viscdvort-gaussian-gpu small", bench_P2D_viscdvort_gaussian, test_repeats(), 10000);
	BENCH("P2D viscdvort-winckelmans-gpu small", bench_P2D_viscdvort_winckelmans, test_repeats(), 10000);
	BENCH("P2D viscdvort-gaussian-gpu medium", bench_P2D_viscdvort_gaussian, test_repeats(), 80000);
	BENCH("P2D viscdvort-winckelmans-gpu medium", bench_P2D_viscdvort_winckelmans, test_repeats(), 80000);
	BENCH("P2D viscdvort-gaussian-gpu large", bench_P2D_viscdvort_gaussian, test_repeats(), 250000);
	BENCH("P2D viscdvort-winckelmans-gpu large", bench_P2D_viscdvort_winckelmans, test_repeats(), 250000);
	BENCH("P2D viscdvort-gaussian-gpu vlarge", bench_P2D_viscdvort_gaussian, test_repeats(), 500000);
	BENCH("P2D viscdvort-winckelmans-gpu vlarge", bench_P2D_viscdvort_winckelmans, test_repeats(), 500000);
	BENCH("P2D viscdvort-gaussian-gpu huge", bench_P2D_viscdvort_gaussian, test_repeats(), 1000000);
	BENCH("P2D viscdvort-winckelmans-gpu huge", bench_P2D_viscdvort_winckelmans, test_repeats(), 1000000);


	/* Now use the CPU, disabling accelerators. */
	for (i = 0; i < n; ++i) {
		cvtx_accelerator_disable(i);
	}

	BENCH("P2D vel-singular-cpu vsmall", bench_P2D_vel_singular, test_repeats(), 1000);
	BENCH("P2D vel-gaussian-cpu vsmall", bench_P2D_vel_gaussian, test_repeats(), 1000);
	BENCH("P2D vel-winckelmans-cpu vsmall", bench_P2D_vel_winckelmans, test_repeats(), 1000);
	BENCH("P2D vel-planetary-cpu vsmall", bench_P2D_vel_planetary, test_repeats(), 1000);

	BENCH("P2D vel-singular-cpu small", bench_P2D_vel_singular, test_repeats(), 10000);
	BENCH("P2D vel-gaussian-cpu small", bench_P2D_vel_gaussian, test_repeats(), 10000);
	BENCH("P2D vel-winckelmans-cpu small", bench_P2D_vel_winckelmans, test_repeats(), 10000);
	BENCH("P2D vel-planetary-cpu small", bench_P2D_vel_planetary, test_repeats(), 10000);

	BENCH("P2D vel-singular-cpu medium", bench_P2D_vel_singular, test_repeats(), 80000);
	BENCH("P2D vel-gaussian-cpu medium", bench_P2D_vel_gaussian, test_repeats(), 80000);
	BENCH("P2D vel-winckelmans-cpu medium", bench_P2D_vel_winckelmans, test_repeats(), 80000);
	BENCH("P2D vel-planetary-cpu medium", bench_P2D_vel_planetary, test_repeats(), 80000);

	BENCH("P2D vel-singular-cpu large", bench_P2D_vel_singular, test_repeats(), 250000);
	BENCH("P2D vel-gaussian-cpu large", bench_P2D_vel_gaussian, test_repeats(), 250000);
	BENCH("P2D vel-winckelmans-cpu large", bench_P2D_vel_winckelmans, test_repeats(), 250000);
	BENCH("P2D vel-planetary-cpu large", bench_P2D_vel_planetary, test_repeats(), 250000);

	BENCH("P2D vel-singular-cpu vlarge", bench_P2D_vel_singular, test_repeats(), 500000);
	BENCH("P2D vel-gaussian-cpu vlarge", bench_P2D_vel_gaussian, test_repeats(), 500000);
	BENCH("P2D vel-winckelmans-cpu vlarge", bench_P2D_vel_winckelmans, test_repeats(), 500000);
	BENCH("P2D vel-planetary-cpu vlarge", bench_P2D_vel_planetary, test_repeats(), 500000);

	BENCH("P2D vel-singular-cpu huge", bench_P2D_vel_singular, test_repeats(), 1000000);
	BENCH("P2D vel-gaussian-cpu huge", bench_P2D_vel_gaussian, test_repeats(), 1000000);
	BENCH("P2D vel-winckelmans-cpu huge", bench_P2D_vel_winckelmans, test_repeats(), 1000000);
	BENCH("P2D vel-planetary-cpu huge", bench_P2D_vel_planetary, test_repeats(), 1000000);

	BENCH("P2D viscdvort-gaussian-cpu vsmall", bench_P2D_viscdvort_gaussian, test_repeats(), 1000);
	BENCH("P2D viscdvort-winckelmans-cpu vsmall", bench_P2D_viscdvort_winckelmans, test_repeats(), 1000);
	BENCH("P2D viscdvort-gaussian-cpu small", bench_P2D_viscdvort_gaussian, test_repeats(), 10000);
	BENCH("P2D viscdvort-winckelmans-cpu small", bench_P2D_viscdvort_winckelmans, test_repeats(), 10000);
	BENCH("P2D viscdvort-gaussian-cpu medium", bench_P2D_viscdvort_gaussian, test_repeats(), 80000);
	BENCH("P2D viscdvort-winckelmans-cpu medium", bench_P2D_viscdvort_winckelmans, test_repeats(), 80000);
	BENCH("P2D viscdvort-gaussian-cpu large", bench_P2D_viscdvort_gaussian, test_repeats(), 250000);
	BENCH("P2D viscdvort-winckelmans-cpu large", bench_P2D_viscdvort_winckelmans, test_repeats(), 250000);
	BENCH("P2D viscdvort-gaussian-cpu vlarge", bench_P2D_viscdvort_gaussian, test_repeats(), 500000);
	BENCH("P2D viscdvort-winckelmans-cpu vlarge", bench_P2D_viscdvort_winckelmans, test_repeats(), 500000);
	BENCH("P2D viscdvort-gaussian-cpu huge", bench_P2D_viscdvort_gaussian, test_repeats(), 1000000);
	BENCH("P2D viscdvort-winckelmans-cpu huge", bench_P2D_viscdvort_winckelmans, test_repeats(), 1000000);
	/* Re-enable GPUs. */
	for (i = 0; i < n; ++i) {
		cvtx_accelerator_enable(i);
	}
	destroy_particles_2D();
	destroy_V2f_arr();
	destroy_V2f_arr2();
	return;
}

/* VEL ---------------------------------------------------------------------*/
void bench_P2D_vel_singular(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_singular();
	cvtx_P2D_M2M_vel(
		particle_2D_pptr(),
		np,
		v2f_arr(),
		np,
		v2f_arr2(),
		&vf,
		0.02
	);
}

void bench_P2D_vel_gaussian(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_gaussian();
	cvtx_P2D_M2M_vel(
		particle_2D_pptr(),
		np,
		v2f_arr(),
		np,
		v2f_arr2(),
		&vf,
		0.02
	);
}

void bench_P2D_vel_winckelmans(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_winckelmans();
	cvtx_P2D_M2M_vel(
		particle_2D_pptr(),
		np,
		v2f_arr(),
		np,
		v2f_arr2(),
		&vf,
		0.02
	);
}

void bench_P2D_vel_planetary(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_planetary();
	cvtx_P2D_M2M_vel(
		particle_2D_pptr(),
		np,
		v2f_arr(),
		np,
		v2f_arr2(),
		&vf,
		0.02
	);
}

/* VISC DVORT --------------------------------------------------------------*/
void bench_P2D_viscdvort_gaussian(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_gaussian();
	cvtx_P2D_M2M_visc_dvort(
		particle_2D_pptr(),
		np,
		particle_2D_pptr(),
		np,
		v2f_arr(),
		&vf,
		0.02,
		1.
	);
}

void bench_P2D_viscdvort_winckelmans(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_winckelmans();
	cvtx_P2D_M2M_visc_dvort(
		particle_2D_pptr(),
		np,
		particle_2D_pptr(),
		np,
		v2f_arr(),
		&vf,
		0.02,
		1.
	);
}
