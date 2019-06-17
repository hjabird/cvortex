#include "libcvtx.h"
/*============================================================================
P2D.c

Vortex particle in 2D with CPU based code.

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
#include <stddef.h>

#ifdef CVTX_USING_OPENCL
#	include "ocl_P2D.h"
#endif

/* The induced velocity for a particle excluding the constant
coefficient 1 / 4pi */
inline bsv_V2f P2D_vel_inner(
	const cvtx_P2D * self,
	const bsv_V2f mes_point,
	const cvtx_VortFunc * kernel,
	float recip_reg_rad)
{
	bsv_V2f rad, ret;
	float radd, rho, g;
	if (bsv_V2f_isequal(self->coord, mes_point)) {
		ret = bsv_V2f_zero();
	}
	else {
		rad = bsv_V2f_minus(mes_point, self->coord);
		radd = bsv_V2f_abs(rad);
		rho = radd * recip_reg_rad;
		g = kernel->g_2D(rho);
		ret.x[0] = rad.x[1] * self->vorticity * g / (radd * radd);
		ret.x[1] = -rad.x[0] * self->vorticity * g / (radd * radd);
	}
	return ret;
}

CVTX_EXPORT bsv_V2f cvtx_P2D_S2S_vel(
	const cvtx_P2D * self,
	const bsv_V2f mes_point,
	const cvtx_VortFunc * kernel,
	float regularisation_radius)
{
	bsv_V2f ret;
	ret = P2D_vel_inner(self, mes_point, kernel,
		1.f / fabsf(regularisation_radius));
	return bsv_V2f_mult(ret, 1.f / (2.f * acosf(-1.f)));
}

CVTX_EXPORT bsv_V2f cvtx_P2D_M2S_vel(
	const cvtx_P2D **array_start,
	const int num_particles,
	const bsv_V2f mes_point,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
	double rx = 0, ry = 0;
	long i;
	float recip_reg_rad = 1.f / fabsf(regularisation_radius);
	assert(num_particles >= 0);
#pragma omp parallel for reduction(+:rx, ry)
	for (i = 0; i < num_particles; ++i) {
		bsv_V2f vel = P2D_vel_inner(array_start[i],
			mes_point, kernel, recip_reg_rad);
		rx += vel.x[0];
		ry += vel.x[1];
	}
	bsv_V2f ret = { (float)rx, (float)ry };
	return bsv_V2f_mult(ret, 1.f / (2.f * acosf(-1.f)));
}


static void cpu_brute_force_P2D_M2M_vel(
	const cvtx_P2D **array_start,
	const int num_particles,
	const bsv_V2f *mes_start,
	const int num_mes,
	bsv_V2f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
	long i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < num_mes; ++i) {
		result_array[i] = cvtx_P2D_M2S_vel(
			array_start, num_particles, mes_start[i],
			kernel, regularisation_radius);
	}
	return;
}

CVTX_EXPORT void cvtx_P2D_M2M_vel(
	const cvtx_P2D **array_start,
	const int num_particles,
	const bsv_V2f *mes_start,
	const int num_mes,
	bsv_V2f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
#ifdef CVTX_USING_OPENCL
	if (num_particles < 256
		|| num_mes < 256
		|| kernel->cl_kernel_name_ext == ""
		|| opencl_brute_force_P2D_M2M_vel(
			array_start, num_particles, mes_start,
			num_mes, result_array, kernel, regularisation_radius) != 0)
#endif
	{
		cpu_brute_force_P2D_M2M_vel(
			array_start, num_particles, mes_start,
			num_mes, result_array, kernel, regularisation_radius);
	}
	return;
}

