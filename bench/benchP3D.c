#include "benchP3D.h"
/*============================================================================
benchP3D.h

Benchmark 3D particles for cvortex.

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

void bench_P3D_vel_singular(int np);
void bench_P3D_vel_gaussian(int np);
void bench_P3D_vel_winckelmans(int np);
void bench_P3D_vel_planetary(int np);

void bench_P3D_dvort_singular(int np);
void bench_P3D_dvort_gaussian(int np);
void bench_P3D_dvort_winckelmans(int np);
void bench_P3D_dvort_planetary(int np);

void bench_P3D_viscdvort_gaussian(int np);
void bench_P3D_viscdvort_winckelmans(int np);

void bench_P3D_vort_singular(int np);
void bench_P3D_vort_gaussian(int np);
void bench_P3D_vort_winckelmans(int np);
void bench_P3D_vort_planetary(int np);

void run_P3D_bench(void) {
	create_particles_3D(1000000, 10.f, 0.01);
	create_V3f_arr(1000000, 10.f);
	create_V3f_arr2(1000000, 10.f);

	/* Run first with GPUs enabled. */
	int i, n;
	n = cvtx_num_accelerators();
	for (i = 0; i < n; ++i) {
		cvtx_accelerator_enable(i);
	}

	BENCH("P3D vel-singular-gpu vsmall", bench_P3D_vel_singular, test_repeats(), 1000);
	BENCH("P3D vel-gaussian-gpu vsmall", bench_P3D_vel_gaussian, test_repeats(), 1000);
	BENCH("P3D vel-winckelmans-gpu vsmall", bench_P3D_vel_winckelmans, test_repeats(), 1000);
	BENCH("P3D vel-planetary-gpu vsmall", bench_P3D_vel_planetary, test_repeats(), 1000);
	
	BENCH("P3D vel-singular-gpu small", bench_P3D_vel_singular, test_repeats(), 10000);
	BENCH("P3D vel-gaussian-gpu small", bench_P3D_vel_gaussian, test_repeats(), 10000);
	BENCH("P3D vel-winckelmans-gpu small", bench_P3D_vel_winckelmans, test_repeats(), 10000);
	BENCH("P3D vel-planetary-gpu small", bench_P3D_vel_planetary, test_repeats(), 10000);

	BENCH("P3D vel-singular-gpu medium", bench_P3D_vel_singular, test_repeats(), 80000);
	BENCH("P3D vel-gaussian-gpu medium", bench_P3D_vel_gaussian, test_repeats(), 80000);
	BENCH("P3D vel-winckelmans-gpu medium", bench_P3D_vel_winckelmans, test_repeats(), 80000);
	BENCH("P3D vel-planetary-gpu medium", bench_P3D_vel_planetary, test_repeats(), 80000);

	BENCH("P3D vel-singular-gpu large", bench_P3D_vel_singular, test_repeats(), 250000);
	BENCH("P3D vel-gaussian-gpu large", bench_P3D_vel_gaussian, test_repeats(), 250000);
	BENCH("P3D vel-winckelmans-gpu large", bench_P3D_vel_winckelmans, test_repeats(), 250000);
	BENCH("P3D vel-planetary-gpu large", bench_P3D_vel_planetary, test_repeats(), 250000);

	BENCH("P3D vel-singular-gpu vlarge", bench_P3D_vel_singular, test_repeats(), 500000);
	BENCH("P3D vel-gaussian-gpu vlarge", bench_P3D_vel_gaussian, test_repeats(), 500000);
	BENCH("P3D vel-winckelmans-gpu vlarge", bench_P3D_vel_winckelmans, test_repeats(), 500000);
	BENCH("P3D vel-planetary-gpu vlarge", bench_P3D_vel_planetary, test_repeats(), 500000);

	BENCH("P3D vel-singular-gpu huge", bench_P3D_vel_singular, test_repeats(), 1000000);
	BENCH("P3D vel-gaussian-gpu huge", bench_P3D_vel_gaussian, test_repeats(), 1000000);
	BENCH("P3D vel-winckelmans-gpu huge", bench_P3D_vel_winckelmans, test_repeats(), 1000000);
	BENCH("P3D vel-planetary-gpu huge", bench_P3D_vel_planetary, test_repeats(), 1000000);

	BENCH("P3D dvort-singular-gpu vsmall", bench_P3D_dvort_singular, test_repeats(), 1000);
	BENCH("P3D dvort-gaussian-gpu vsmall", bench_P3D_dvort_gaussian, test_repeats(), 1000);
	BENCH("P3D dvort-winckelmans-gpu vsmall", bench_P3D_dvort_winckelmans, test_repeats(), 1000);
	BENCH("P3D dvort-planetary-gpu vsmall", bench_P3D_dvort_planetary, test_repeats(), 1000);

	BENCH("P3D dvort-singular-gpu small", bench_P3D_dvort_singular, test_repeats(), 10000);
	BENCH("P3D dvort-gaussian-gpu small", bench_P3D_dvort_gaussian, test_repeats(), 10000);
	BENCH("P3D dvort-winckelmans-gpu small", bench_P3D_dvort_winckelmans, test_repeats(), 10000);
	BENCH("P3D dvort-planetary-gpu small", bench_P3D_dvort_planetary, test_repeats(), 10000);

	BENCH("P3D dvort-singular-gpu medium", bench_P3D_dvort_singular, test_repeats(), 80000);
	BENCH("P3D dvort-gaussian-gpu medium", bench_P3D_dvort_gaussian, test_repeats(), 80000);
	BENCH("P3D dvort-winckelmans-gpu medium", bench_P3D_dvort_winckelmans, test_repeats(), 80000);
	BENCH("P3D dvort-planetary-gpu medium", bench_P3D_dvort_planetary, test_repeats(), 80000);

	BENCH("P3D dvort-singular-gpu large", bench_P3D_dvort_singular, test_repeats(), 250000);
	BENCH("P3D dvort-gaussian-gpu large", bench_P3D_dvort_gaussian, test_repeats(), 250000);
	BENCH("P3D dvort-winckelmans-gpu large", bench_P3D_dvort_winckelmans, test_repeats(), 250000);
	BENCH("P3D dvort-planetary-gpu large", bench_P3D_dvort_planetary, test_repeats(), 250000);

	BENCH("P3D dvort-singular-gpu vlarge", bench_P3D_dvort_singular, test_repeats(), 500000);
	BENCH("P3D dvort-gaussian-gpu vlarge", bench_P3D_dvort_gaussian, test_repeats(), 500000);
	BENCH("P3D dvort-winckelmans-gpu vlarge", bench_P3D_dvort_winckelmans, test_repeats(), 500000);
	BENCH("P3D dvort-planetary-gpu vlarge", bench_P3D_dvort_planetary, test_repeats(), 500000);

	BENCH("P3D dvort-singular-gpu huge", bench_P3D_dvort_singular, test_repeats(), 1000000);
	BENCH("P3D dvort-gaussian-gpu huge", bench_P3D_dvort_gaussian, test_repeats(), 1000000);
	BENCH("P3D dvort-winckelmans-gpu huge", bench_P3D_dvort_winckelmans, test_repeats(), 1000000);
	BENCH("P3D dvort-planetary-gpu huge", bench_P3D_dvort_planetary, test_repeats(), 1000000);

	BENCH("P3D viscdvort-gaussian-gpu vsmall", bench_P3D_viscdvort_gaussian, test_repeats(), 1000);
	BENCH("P3D viscdvort-winckelmans-gpu vsmall", bench_P3D_viscdvort_winckelmans, test_repeats(), 1000);
	BENCH("P3D viscdvort-gaussian-gpu small", bench_P3D_viscdvort_gaussian, test_repeats(), 10000);
	BENCH("P3D viscdvort-winckelmans-gpu small", bench_P3D_viscdvort_winckelmans, test_repeats(), 10000);
	BENCH("P3D viscdvort-gaussian-gpu medium", bench_P3D_viscdvort_gaussian, test_repeats(), 80000);
	BENCH("P3D viscdvort-winckelmans-gpu medium", bench_P3D_viscdvort_winckelmans, test_repeats(), 80000);
	BENCH("P3D viscdvort-gaussian-gpu large", bench_P3D_viscdvort_gaussian, test_repeats(), 250000);
	BENCH("P3D viscdvort-winckelmans-gpu large", bench_P3D_viscdvort_winckelmans, test_repeats(), 250000);
	BENCH("P3D viscdvort-gaussian-gpu vlarge", bench_P3D_viscdvort_gaussian, test_repeats(), 500000);
	BENCH("P3D viscdvort-winckelmans-gpu vlarge", bench_P3D_viscdvort_winckelmans, test_repeats(), 500000);
	BENCH("P3D viscdvort-gaussian-gpu huge", bench_P3D_viscdvort_gaussian, test_repeats(), 1000000);
	BENCH("P3D viscdvort-winckelmans-gpu huge", bench_P3D_viscdvort_winckelmans, test_repeats(), 1000000);

	BENCH("P3D vort-singular-gpu vsmall", bench_P3D_vort_singular, test_repeats(), 1000);
	BENCH("P3D vort-gaussian-gpu vsmall", bench_P3D_vort_gaussian, test_repeats(), 1000);
	BENCH("P3D vort-winckelmans-gpu vsmall", bench_P3D_vort_winckelmans, test_repeats(), 1000);
	BENCH("P3D vort-planetary-gpu vsmall", bench_P3D_vort_planetary, test_repeats(), 1000);

	BENCH("P3D vort-singular-gpu small", bench_P3D_vort_singular, test_repeats(), 10000);
	BENCH("P3D vort-gaussian-gpu small", bench_P3D_vort_gaussian, test_repeats(), 10000);
	BENCH("P3D vort-winckelmans-gpu small", bench_P3D_vort_winckelmans, test_repeats(), 10000);
	BENCH("P3D vort-planetary-gpu small", bench_P3D_vort_planetary, test_repeats(), 10000);

	BENCH("P3D vort-singular-gpu medium", bench_P3D_vort_singular, test_repeats(), 80000);
	BENCH("P3D vort-gaussian-gpu medium", bench_P3D_vort_gaussian, test_repeats(), 80000);
	BENCH("P3D vort-winckelmans-gpu medium", bench_P3D_vort_winckelmans, test_repeats(), 80000);
	BENCH("P3D vort-planetary-gpu medium", bench_P3D_vort_planetary, test_repeats(), 80000);

	BENCH("P3D vort-singular-gpu large", bench_P3D_vort_singular, test_repeats(), 250000);
	BENCH("P3D vort-gaussian-gpu large", bench_P3D_vort_gaussian, test_repeats(), 250000);
	BENCH("P3D vort-winckelmans-gpu large", bench_P3D_vort_winckelmans, test_repeats(), 250000);
	BENCH("P3D vort-planetary-gpu large", bench_P3D_vort_planetary, test_repeats(), 250000);

	BENCH("P3D vort-singular-gpu vlarge", bench_P3D_vort_singular, test_repeats(), 500000);
	BENCH("P3D vort-gaussian-gpu vlarge", bench_P3D_vort_gaussian, test_repeats(), 500000);
	BENCH("P3D vort-winckelmans-gpu vlarge", bench_P3D_vort_winckelmans, test_repeats(), 500000);
	BENCH("P3D vort-planetary-gpu vlarge", bench_P3D_vort_planetary, test_repeats(), 500000);

	BENCH("P3D vort-singular-gpu huge", bench_P3D_vort_singular, test_repeats(), 1000000);
	BENCH("P3D vort-gaussian-gpu huge", bench_P3D_vort_gaussian, test_repeats(), 1000000);
	BENCH("P3D vort-winckelmans-gpu huge", bench_P3D_vort_winckelmans, test_repeats(), 1000000);
	BENCH("P3D vort-planetary-gpu huge", bench_P3D_vort_planetary, test_repeats(), 1000000);

	/* Now use the CPU, disabling accelerators. */
	for (i = 0; i < n; ++i) {
		cvtx_accelerator_disable(i);
	}

	BENCH("P3D vel-singular-cpu vsmall", bench_P3D_vel_singular, test_repeats(), 1000);
	BENCH("P3D vel-gaussian-cpu vsmall", bench_P3D_vel_gaussian, test_repeats(), 1000);
	BENCH("P3D vel-winckelmans-cpu vsmall", bench_P3D_vel_winckelmans, test_repeats(), 1000);
	BENCH("P3D vel-planetary-cpu vsmall", bench_P3D_vel_planetary, test_repeats(), 1000);

	BENCH("P3D vel-singular-cpu small", bench_P3D_vel_singular, test_repeats(), 10000);
	BENCH("P3D vel-gaussian-cpu small", bench_P3D_vel_gaussian, test_repeats(), 10000);
	BENCH("P3D vel-winckelmans-cpu small", bench_P3D_vel_winckelmans, test_repeats(), 10000);
	BENCH("P3D vel-planetary-cpu small", bench_P3D_vel_planetary, test_repeats(), 10000);

	BENCH("P3D vel-singular-cpu medium", bench_P3D_vel_singular, test_repeats(), 80000);
	BENCH("P3D vel-gaussian-cpu medium", bench_P3D_vel_gaussian, test_repeats(), 80000);
	BENCH("P3D vel-winckelmans-cpu medium", bench_P3D_vel_winckelmans, test_repeats(), 80000);
	BENCH("P3D vel-planetary-cpu medium", bench_P3D_vel_planetary, test_repeats(), 80000);

	BENCH("P3D vel-singular-cpu large", bench_P3D_vel_singular, test_repeats(), 250000);
	BENCH("P3D vel-gaussian-cpu large", bench_P3D_vel_gaussian, test_repeats(), 250000);
	BENCH("P3D vel-winckelmans-cpu large", bench_P3D_vel_winckelmans, test_repeats(), 250000);
	BENCH("P3D vel-planetary-cpu large", bench_P3D_vel_planetary, test_repeats(), 250000);

	BENCH("P3D vel-singular-cpu vlarge", bench_P3D_vel_singular, test_repeats(), 500000);
	BENCH("P3D vel-gaussian-cpu vlarge", bench_P3D_vel_gaussian, test_repeats(), 500000);
	BENCH("P3D vel-winckelmans-cpu vlarge", bench_P3D_vel_winckelmans, test_repeats(), 500000);
	BENCH("P3D vel-planetary-cpu vlarge", bench_P3D_vel_planetary, test_repeats(), 500000);

	BENCH("P3D vel-singular-cpu huge", bench_P3D_vel_singular, test_repeats(), 1000000);
	BENCH("P3D vel-gaussian-cpu huge", bench_P3D_vel_gaussian, test_repeats(), 1000000);
	BENCH("P3D vel-winckelmans-cpu huge", bench_P3D_vel_winckelmans, test_repeats(), 1000000);
	BENCH("P3D vel-planetary-cpu huge", bench_P3D_vel_planetary, test_repeats(), 1000000);

	BENCH("P3D dvort-singular-cpu vsmall", bench_P3D_dvort_singular, test_repeats(), 1000);
	BENCH("P3D dvort-gaussian-cpu vsmall", bench_P3D_dvort_gaussian, test_repeats(), 1000);
	BENCH("P3D dvort-winckelmans-cpu vsmall", bench_P3D_dvort_winckelmans, test_repeats(), 1000);
	BENCH("P3D dvort-planetary-cpu vsmall", bench_P3D_dvort_planetary, test_repeats(), 1000);

	BENCH("P3D dvort-singular-cpu small", bench_P3D_dvort_singular, test_repeats(), 10000);
	BENCH("P3D dvort-gaussian-cpu small", bench_P3D_dvort_gaussian, test_repeats(), 10000);
	BENCH("P3D dvort-winckelmans-cpu small", bench_P3D_dvort_winckelmans, test_repeats(), 10000);
	BENCH("P3D dvort-planetary-cpu small", bench_P3D_dvort_planetary, test_repeats(), 10000);

	BENCH("P3D dvort-singular-cpu medium", bench_P3D_dvort_singular, test_repeats(), 80000);
	BENCH("P3D dvort-gaussian-cpu medium", bench_P3D_dvort_gaussian, test_repeats(), 80000);
	BENCH("P3D dvort-winckelmans-cpu medium", bench_P3D_dvort_winckelmans, test_repeats(), 80000);
	BENCH("P3D dvort-planetary-cpu medium", bench_P3D_dvort_planetary, test_repeats(), 80000);

	BENCH("P3D dvort-singular-cpu large", bench_P3D_dvort_singular, test_repeats(), 250000);
	BENCH("P3D dvort-gaussian-cpu large", bench_P3D_dvort_gaussian, test_repeats(), 250000);
	BENCH("P3D dvort-winckelmans-cpu large", bench_P3D_dvort_winckelmans, test_repeats(), 250000);
	BENCH("P3D dvort-planetary-cpu large", bench_P3D_dvort_planetary, test_repeats(), 250000);

	BENCH("P3D dvort-singular-cpu vlarge", bench_P3D_dvort_singular, test_repeats(), 500000);
	BENCH("P3D dvort-gaussian-cpu vlarge", bench_P3D_dvort_gaussian, test_repeats(), 500000);
	BENCH("P3D dvort-winckelmans-cpu vlarge", bench_P3D_dvort_winckelmans, test_repeats(), 500000);
	BENCH("P3D dvort-planetary-cpu vlarge", bench_P3D_dvort_planetary, test_repeats(), 500000);

	BENCH("P3D dvort-singular-cpu huge", bench_P3D_dvort_singular, test_repeats(), 1000000);
	BENCH("P3D dvort-gaussian-cpu huge", bench_P3D_dvort_gaussian, test_repeats(), 1000000);
	BENCH("P3D dvort-winckelmans-cpu huge", bench_P3D_dvort_winckelmans, test_repeats(), 1000000);
	BENCH("P3D dvort-planetary-cpu huge", bench_P3D_dvort_planetary, test_repeats(), 1000000);

	BENCH("P3D viscdvort-gaussian-cpu vsmall", bench_P3D_viscdvort_gaussian, test_repeats(), 1000);
	BENCH("P3D viscdvort-winckelmans-cpu vsmall", bench_P3D_viscdvort_winckelmans, test_repeats(), 1000);
	BENCH("P3D viscdvort-gaussian-cpu small", bench_P3D_viscdvort_gaussian, test_repeats(), 10000);
	BENCH("P3D viscdvort-winckelmans-cpu small", bench_P3D_viscdvort_winckelmans, test_repeats(), 10000);
	BENCH("P3D viscdvort-gaussian-cpu medium", bench_P3D_viscdvort_gaussian, test_repeats(), 80000);
	BENCH("P3D viscdvort-winckelmans-cpu medium", bench_P3D_viscdvort_winckelmans, test_repeats(), 80000);
	BENCH("P3D viscdvort-gaussian-cpu large", bench_P3D_viscdvort_gaussian, test_repeats(), 250000);
	BENCH("P3D viscdvort-winckelmans-cpu large", bench_P3D_viscdvort_winckelmans, test_repeats(), 250000);
	BENCH("P3D viscdvort-gaussian-cpu vlarge", bench_P3D_viscdvort_gaussian, test_repeats(), 500000);
	BENCH("P3D viscdvort-winckelmans-cpu vlarge", bench_P3D_viscdvort_winckelmans, test_repeats(), 500000);
	BENCH("P3D viscdvort-gaussian-cpu huge", bench_P3D_viscdvort_gaussian, test_repeats(), 1000000);
	BENCH("P3D viscdvort-winckelmans-cpu huge", bench_P3D_viscdvort_winckelmans, test_repeats(), 1000000);

	BENCH("P3D vort-singular-cpu vsmall", bench_P3D_vort_singular, test_repeats(), 1000);
	BENCH("P3D vort-gaussian-cpu vsmall", bench_P3D_vort_gaussian, test_repeats(), 1000);
	BENCH("P3D vort-winckelmans-cpu vsmall", bench_P3D_vort_winckelmans, test_repeats(), 1000);
	BENCH("P3D vort-planetary-cpu vsmall", bench_P3D_vort_planetary, test_repeats(), 1000);

	BENCH("P3D vort-singular-cpu small", bench_P3D_vort_singular, test_repeats(), 10000);
	BENCH("P3D vort-gaussian-cpu small", bench_P3D_vort_gaussian, test_repeats(), 10000);
	BENCH("P3D vort-winckelmans-cpu small", bench_P3D_vort_winckelmans, test_repeats(), 10000);
	BENCH("P3D vort-planetary-cpu small", bench_P3D_vort_planetary, test_repeats(), 10000);

	BENCH("P3D vort-singular-cpu medium", bench_P3D_vort_singular, test_repeats(), 80000);
	BENCH("P3D vort-gaussian-cpu medium", bench_P3D_vort_gaussian, test_repeats(), 80000);
	BENCH("P3D vort-winckelmans-cpu medium", bench_P3D_vort_winckelmans, test_repeats(), 80000);
	BENCH("P3D vort-planetary-cpu medium", bench_P3D_vort_planetary, test_repeats(), 80000);

	BENCH("P3D vort-singular-cpu large", bench_P3D_vort_singular, test_repeats(), 250000);
	BENCH("P3D vort-gaussian-cpu large", bench_P3D_vort_gaussian, test_repeats(), 250000);
	BENCH("P3D vort-winckelmans-cpu large", bench_P3D_vort_winckelmans, test_repeats(), 250000);
	BENCH("P3D vort-planetary-cpu large", bench_P3D_vort_planetary, test_repeats(), 250000);

	BENCH("P3D vort-singular-cpu vlarge", bench_P3D_vort_singular, test_repeats(), 500000);
	BENCH("P3D vort-gaussian-cpu vlarge", bench_P3D_vort_gaussian, test_repeats(), 500000);
	BENCH("P3D vort-winckelmans-cpu vlarge", bench_P3D_vort_winckelmans, test_repeats(), 500000);
	BENCH("P3D vort-planetary-cpu vlarge", bench_P3D_vort_planetary, test_repeats(), 500000);

	BENCH("P3D vort-singular-cpu huge", bench_P3D_vort_singular, test_repeats(), 1000000);
	BENCH("P3D vort-gaussian-cpu huge", bench_P3D_vort_gaussian, test_repeats(), 1000000);
	BENCH("P3D vort-winckelmans-cpu huge", bench_P3D_vort_winckelmans, test_repeats(), 1000000);
	BENCH("P3D vort-planetary-cpu huge", bench_P3D_vort_planetary, test_repeats(), 1000000);

	/* Re-enable GPUs. */
	for (i = 0; i < n; ++i) {
		cvtx_accelerator_enable(i);
	}
	destroy_particles_3D();
	destroy_V3f_arr();
	destroy_V3f_arr2();
	return;
}

/* VEL ---------------------------------------------------------------------*/
void bench_P3D_vel_singular(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_singular();
	cvtx_P3D_M2M_vel(
		particle_3D_pptr(),
		np,
		v3f_arr(),
		np,
		v3f_arr2(),
		&vf,
		0.02
	);
}

void bench_P3D_vel_gaussian(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_gaussian();
	cvtx_P3D_M2M_vel(
		particle_3D_pptr(),
		np,
		v3f_arr(),
		np,
		v3f_arr2(),
		&vf,
		0.02
	);
}

void bench_P3D_vel_winckelmans(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_winckelmans();
	cvtx_P3D_M2M_vel(
		particle_3D_pptr(),
		np,
		v3f_arr(),
		np,
		v3f_arr2(),
		&vf,
		0.02
	);
}

void bench_P3D_vel_planetary(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_planetary();
	cvtx_P3D_M2M_vel(
		particle_3D_pptr(),
		np,
		v3f_arr(),
		np,
		v3f_arr2(),
		&vf,
		0.02
	);
}

/* DVORT -------------------------------------------------------------------*/
void bench_P3D_dvort_singular(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_singular();
	cvtx_P3D_M2M_dvort(
		particle_3D_pptr(),
		np, 
		particle_3D_pptr(),
		np,
		v3f_arr(),
		&vf,
		0.02
	);
}

void bench_P3D_dvort_gaussian(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_gaussian();
	cvtx_P3D_M2M_dvort(
		particle_3D_pptr(),
		np,
		particle_3D_pptr(),
		np,
		v3f_arr(),
		&vf,
		0.02
	);
}

void bench_P3D_dvort_winckelmans(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_winckelmans();
	cvtx_P3D_M2M_dvort(
		particle_3D_pptr(),
		np,
		particle_3D_pptr(),
		np,
		v3f_arr(),
		&vf,
		0.02
	);
}

void bench_P3D_dvort_planetary(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_planetary();
	cvtx_P3D_M2M_dvort(
		particle_3D_pptr(),
		np,
		particle_3D_pptr(),
		np,
		v3f_arr(),
		&vf,
		0.02
	);
}

/* VISC DVORT --------------------------------------------------------------*/
void bench_P3D_viscdvort_gaussian(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_gaussian();
	cvtx_P3D_M2M_visc_dvort(
		particle_3D_pptr(),
		np,
		particle_3D_pptr(),
		np,
		v3f_arr(),
		&vf,
		0.02,
		1.
	);
}

void bench_P3D_viscdvort_winckelmans(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_winckelmans();
	cvtx_P3D_M2M_visc_dvort(
		particle_3D_pptr(),
		np,
		particle_3D_pptr(),
		np,
		v3f_arr(),
		&vf,
		0.02,
		1.
	);
}

/* VEL ---------------------------------------------------------------------*/
void bench_P3D_vort_singular(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_singular();
	cvtx_P3D_M2M_vort(
		particle_3D_pptr(),
		np,
		v3f_arr(),
		np,
		v3f_arr2(),
		&vf,
		0.02
	);
}

void bench_P3D_vort_gaussian(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_gaussian();
	cvtx_P3D_M2M_vort(
		particle_3D_pptr(),
		np,
		v3f_arr(),
		np,
		v3f_arr2(),
		&vf,
		0.02
	);
}

void bench_P3D_vort_winckelmans(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_winckelmans();
	cvtx_P3D_M2M_vort(
		particle_3D_pptr(),
		np,
		v3f_arr(),
		np,
		v3f_arr2(),
		&vf,
		0.02
	);
}

void bench_P3D_vort_planetary(int np) {
	cvtx_VortFunc vf = cvtx_VortFunc_planetary();
	cvtx_P3D_M2M_vort(
		particle_3D_pptr(),
		np,
		v3f_arr(),
		np,
		v3f_arr2(),
		&vf,
		0.02
	);
}
