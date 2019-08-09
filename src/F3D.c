#include "libcvtx.h"
/*============================================================================
StraightVortFil.c

Basic representation of a straight vortex filament.

Copyright(c) 2019 HJA Bird

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
#include <stdlib.h>
#include "ocl_F3D.h"

static const float pi_f = 3.14159265359f;

CVTX_EXPORT bsv_V3f cvtx_F3D_S2S_vel(
	const cvtx_F3D *self,
	const bsv_V3f mes_point) 
{

	assert(self != NULL);
	bsv_V3f r0, r1, r2, crosstmp;
	float t1, t2, t21, t22;
	r1 = bsv_V3f_minus(mes_point, self->start);
	r2 = bsv_V3f_minus(mes_point, self->end);
	r0 = bsv_V3f_minus(r1, r2);
	crosstmp = bsv_V3f_cross(r1, r2);
	t1 = self->strength / (4 * pi_f * powf(bsv_V3f_abs(crosstmp), 2));
	t21 = bsv_V3f_dot(r1, r0) / bsv_V3f_abs(r1);
	t22 = bsv_V3f_dot(r2, r0) / bsv_V3f_abs(r2);
	t2 = t21 - t22;
	/* (NaN != NaN) == TRUE*/
	return fabsf(t1 * t2) <= 3.40282346e38 ? bsv_V3f_mult(crosstmp, t1 * t2) : bsv_V3f_zero();
}

CVTX_EXPORT bsv_V3f cvtx_F3D_S2S_dvort(
	const cvtx_F3D *self,
	const cvtx_P3D *induced_particle) 
{
	assert(self != NULL);
	/* HJAB, Notes 4, pg.42 - pg. 43 for general theme. */
	bsv_V3f r0, r1, r2, t211, A, ret;
	float t1, t2121, t2122, t221, t212, t222, t2221, t2222, B;
	r1 = bsv_V3f_minus(induced_particle->coord, self->start);
	r2 = bsv_V3f_minus(induced_particle->coord, self->end);
	r0 = bsv_V3f_minus(r1, r2);
	t1 = self->strength / (4 * pi_f);
	t211 = bsv_V3f_div(r0, -powf(bsv_V3f_abs(bsv_V3f_cross(r1, r0)), 2));
	t2121 = bsv_V3f_dot(r0, r1) / bsv_V3f_abs(r1);
	t2122 = -bsv_V3f_dot(r0, r2) / bsv_V3f_abs(r2);
	t221 = (float)3.0 / bsv_V3f_abs(r0);
	t2221 = bsv_V3f_abs(bsv_V3f_cross(r0, r1)) / bsv_V3f_abs(r1);
	t2222 = -bsv_V3f_abs(bsv_V3f_cross(r0, r1)) / bsv_V3f_abs(r2);
	t222 = t2221 + t2222;
	t212 = t2121 + t2122;
	A = bsv_V3f_mult(t211, t1*t212);
	B = t221 * t1 * t222;
	ret = bsv_V3f_plus(
			bsv_V3f_mult(induced_particle->vorticity, B),
			bsv_V3f_cross(A, induced_particle->vorticity));
	if ((t222 != t222) || (t212 != t212)) { ret = bsv_V3f_zero(); }
	return ret;
};

CVTX_EXPORT bsv_V3f cvtx_F3D_M2S_vel(
	const cvtx_F3D **array_start,
	const int num_particles,
	const bsv_V3f mes_point) 
{
	assert(num_particles >= 0);
	assert(array_start != NULL);
	bsv_V3f vel;
	double rx = 0, ry = 0, rz = 0;
	long i;
	for (i = 0; i < num_particles; ++i) {
		vel = cvtx_F3D_S2S_vel(array_start[i],
			mes_point);
		rx += vel.x[0];
		ry += vel.x[1];
		rz += vel.x[2];
	}
	bsv_V3f ret = { (float)rx, (float)ry, (float)rz };
	return ret;
}

CVTX_EXPORT bsv_V3f cvtx_F3D_M2S_dvort(
	const cvtx_F3D **array_start,
	const int num_particles,
	const cvtx_P3D *induced_particle) 
{
	assert(num_particles >= 0);
	assert(array_start != NULL);
	bsv_V3f dvort;
	double rx = 0, ry = 0, rz = 0;
	long i;
	for (i = 0; i < num_particles; ++i) {
		dvort = cvtx_F3D_S2S_dvort(array_start[i],
			induced_particle);
		rx += dvort.x[0];
		ry += dvort.x[1];
		rz += dvort.x[2];
	}
	bsv_V3f ret = { (float)rx, (float)ry, (float)rz };
	return ret;
}

void cpu_brute_force_StraightVortFilArr_Arr_ind_vel(
	const cvtx_F3D **array_start,
	const int num_particles,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array) 
{
	long i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < num_mes; ++i) {
		result_array[i] = cvtx_F3D_M2S_vel(
			array_start, num_particles, mes_start[i]);
	}
	return;
}

void cpu_brute_force_StraightVortFilArr_Arr_ind_dvort(
	const cvtx_F3D **array_start,
	const int num_particles,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array) 
{
	long i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < num_induced; ++i) {
		result_array[i] = cvtx_F3D_M2S_dvort(
			array_start, num_particles, induced_start[i]);
	}
	return;
}

CVTX_EXPORT void cvtx_F3D_M2M_vel(
	const cvtx_F3D **array_start,
	const int num_filaments,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array)
{
#ifdef CVTX_USING_OPENCL
	if (opencl_brute_force_F3D_M2M_vel(
			array_start, num_filaments, mes_start,
			num_mes, result_array) != 0)
#endif
	{
		cpu_brute_force_StraightVortFilArr_Arr_ind_vel(
			array_start, num_filaments, mes_start,
			num_mes, result_array);
	}
	return;
}

CVTX_EXPORT void cvtx_F3D_M2M_dvort(
	const cvtx_F3D **array_start,
	const int num_fil,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array)
{
#ifdef CVTX_USING_OPENCL
	if (num_fil < 256
		|| num_induced < 256
		|| opencl_brute_force_F3D_M2M_dvort(
			array_start, num_fil, induced_start,
			num_induced, result_array) != 0)
#endif
	{
		cpu_brute_force_StraightVortFilArr_Arr_ind_dvort(
			array_start, num_fil, induced_start,
			num_induced, result_array);
	}
	return;
}

CVTX_EXPORT void cvtx_F3D_inf_mtrx(
	const cvtx_F3D **array_start,
	const int num_filaments,
	const bsv_V3f *mes_start,
	const bsv_V3f *dir_start,
	const int num_mes,
	float *result_array) {
	assert(array_start != NULL);
	assert(num_filaments >= 0);
	assert(mes_start != NULL);
	assert(dir_start != NULL);
	assert(num_mes >= 0);
	assert(result_array != NULL);
	int i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < num_mes; ++i) {
		int j;
		bsv_V3f vel;
		for (j = 0; j < num_filaments; ++j) {
			vel = cvtx_F3D_S2S_vel(array_start[j], mes_start[i]);
			result_array[i * num_filaments + j] = bsv_V3f_dot(vel, dir_start[i]);
		}
	}
}
