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
#include <stdlib.h>
#include <string.h>

#include "gridkey.h"

#ifdef CVTX_USING_OPENCL
#	include "ocl_P2D.h"
#endif

#ifdef _WIN32
#	define qsort_r qsort_s
#elif defined(linux)
#	define qsort_r qsort_r
#endif

#define NG_FOR_REDUCING_PARICLES 64

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


/* Visous vorticity exchange methods ----------------------------------------*/

CVTX_EXPORT float cvtx_P2D_S2S_visc_dvort(
	const cvtx_P2D * self,
	const cvtx_P2D * induced_particle,
	const cvtx_VortFunc * kernel,
	float regularisation_radius,
	float kinematic_visc)
{
	bsv_V2f rad;
	float radd, rho, ret, t1, t2, t22, t21, t211, t212;
	assert(kernel->eta_2D != NULL && "Used vortex regularisation"
		"that did have a defined eta function");
	if (bsv_V2f_isequal(self->coord, induced_particle->coord)) {
		ret = 0.f;
	}
	else {
		rad = bsv_V2f_minus(self->coord, induced_particle->coord);
		radd = bsv_V2f_abs(rad);
		rho = fabsf(radd / regularisation_radius);
		t1 = 2 * kinematic_visc / powf(regularisation_radius, 2);
		t211 = self->vorticity * induced_particle->area;
		t212 = -induced_particle->vorticity * self->area;
		t21 = t211 + t212;
		t22 = kernel->eta_3D(rho);
		t2 = t21* t22;
		ret = t2 * t1;
	}
	return ret;
}

CVTX_EXPORT float cvtx_P2D_M2S_visc_dvort(
	const cvtx_P2D **array_start,
	const int num_particles,
	const cvtx_P2D *induced_particle,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc)
{
	float dvort = 0.f;
	long i;
	assert(num_particles >= 0);
#pragma omp parallel for reduction(+:dvort)
	for (i = 0; i < num_particles; ++i) {
		dvort += cvtx_P2D_S2S_visc_dvort(array_start[i],
			induced_particle, kernel, regularisation_radius, kinematic_visc);
	}
	return dvort;
}

void cpu_brute_force_P2D_M2M_visc_dvort(
	const cvtx_P2D **array_start,
	const int num_particles,
	const cvtx_P2D **induced_start,
	const int num_induced,
	float *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc)
{
	long i;
	for (i = 0; i < num_induced; ++i) {
		result_array[i] = cvtx_P2D_M2S_visc_dvort(
			array_start, num_particles, induced_start[i],
			kernel, regularisation_radius, kinematic_visc);
	}
	return;
}

CVTX_EXPORT void cvtx_P2D_M2M_visc_dvort(
	const cvtx_P2D **array_start,
	const int num_particles,
	const cvtx_P2D **induced_start,
	const int num_induced,
	float *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc)
{
#ifdef CVTX_USING_OPENCL
	if (num_particles < 256
		|| num_induced < 256
		|| kernel->cl_kernel_name_ext == ""
		|| opencl_brute_force_P2D_M2M_visc_dvort(
			array_start, num_particles, induced_start,
			num_induced, result_array, kernel, regularisation_radius, kinematic_visc) != 0)
#endif
	{
		cpu_brute_force_P2D_M2M_visc_dvort(
			array_start, num_particles, induced_start,
			num_induced, result_array, kernel, regularisation_radius, kinematic_visc);
	}
	return;
}

/* Particle redistribution -------------------------------------------------*/

CVTX_EXPORT int cvtx_P2D_redistribute_on_grid(
	const cvtx_P2D **input_array_start,
	const int n_input_particles,
	cvtx_P2D *output_particles,		/* input is &(*cvtx_P2D) to write to */
	int max_output_particles,		/* Set to resultant num particles.   */
	const cvtx_RedistFunc *redistributor,
	float grid_density,
	float negligible_vort) {

	assert(n_input_particles >= 0);
	assert(max_output_particles >= 0);
	assert(grid_density > 0.f);
	assert(negligible_vort >= 0.f);
	assert(negligible_vort < 1.f);

	int i, j, k, n_created_particles;
	int grid_radius;				/* Value of U that returns zero.	*/
	int ppop;						/* Particles per input particle.	*/
	float minx, miny;				/* Bounds of the particle box.		*/
	/* Index array and grid location array of input particles.			*/
	unsigned int *oidx_array = NULL;
	struct Gridkey2D *okey_array = NULL;
	/* Index, grid location and vorticity arrays of new particles.		*/
	struct Gridkey2D *nkey_array = NULL, *nnkey_array = NULL;
	float *nvort_array = NULL, *nnvort_array = NULL;
	unsigned int *nidx_array = NULL;
	/* For particle removal: */
	float ave_particle_str, min_keepable_particle, vorticity_deficit, 
		maxv, minv;

	/* Generate grid keys for existing particles. */
	minmax_xy_posn(input_array_start, n_input_particles, &minx, NULL, &miny, NULL);
	grid_radius = (int)roundf(redistributor->radius);
	minx -= grid_radius * grid_density;
	miny -= grid_radius * grid_density;

	oidx_array = malloc(sizeof(unsigned int) * n_input_particles);
	okey_array = malloc(sizeof(struct Gridkey2D) * n_input_particles);
	for (i = 0; i < n_input_particles; ++i) {
		oidx_array[i] = i;
		okey_array[i] = g_P2D_gridkey2D(input_array_start[i],
			grid_density, minx, miny);
	}
	/* Sort our key indexs by gridkey (makes later array much faster to sort) */
	qsort_r(oidx_array, n_input_particles, sizeof(unsigned int), comp_Gridkey2D_by_idx, okey_array);
	
	/* Now we make new particles based on grid. */
	/* ppop = Particles per orginal particle. */
	ppop = (grid_radius * 2 + 1) * (grid_radius * 2 + 1);
	nkey_array = malloc(sizeof(struct Gridkey2D) * ppop * n_input_particles);
	nvort_array = malloc(sizeof(float) * ppop * n_input_particles); 
	nidx_array = malloc(sizeof(unsigned int) * ppop * n_input_particles);
	for (i = 0; i < n_input_particles; ++i) {
		int widx = oidx_array[i];
		unsigned int okx, oky;
		bsv_V2f p_coord;
		float p_vort;
		okx = okey_array[widx].xk;
		oky = okey_array[widx].yk;
		p_coord = input_array_start[widx]->coord;
		p_vort = input_array_start[widx]->vorticity;
		for (j = -grid_radius; j <= grid_radius; ++j) {
			for (k = -grid_radius; k <= grid_radius; ++k) {
				float U, W, vortfrac; 
				bsv_V2f n_coord, dx;
				int np_idx;
				n_coord.x[0] = okx * grid_density + j * grid_density + minx;
				n_coord.x[1] = oky * grid_density + k * grid_density + miny;
				dx = bsv_V2f_minus(p_coord, n_coord);
				U = fabsf(dx.x[0] / grid_density);
				W = fabsf(dx.x[1] / grid_density);
				vortfrac = redistributor->func(U) * redistributor->func(W);
				np_idx = k + grid_radius + 
					(j + grid_radius) * (2 * grid_radius + 1) + widx * ppop;
				nidx_array[np_idx] = np_idx;
				nkey_array[np_idx].xk = okx + j;
				nkey_array[np_idx].yk = oky + k;
				nvort_array[np_idx] = vortfrac * p_vort;
			}
		}
	}
	free(oidx_array);
	free(okey_array);

	/* Now merge our new particles */
	qsort_r(nidx_array, n_input_particles * ppop, sizeof(unsigned int), 
		comp_Gridkey2D_by_idx, nkey_array);
	nnkey_array = malloc(sizeof(struct Gridkey2D) * n_input_particles * ppop);
	nnvort_array = malloc(sizeof(float) * n_input_particles * ppop);
	for (i = 0; i < ppop * n_input_particles; ++i) {
		nnkey_array[i] = nkey_array[nidx_array[i]];
		nnvort_array[i] = nvort_array[nidx_array[i]];
	}
	j = 0;
	if (ppop * n_input_particles > 0) {
		nkey_array[0] = nnkey_array[0];
		nvort_array[0] = nvort_array[0];
	}
	for (i = 1; i < ppop * n_input_particles; ++i) {
		if (nnkey_array[i].xk == nkey_array[j].xk &&
			nnkey_array[i].yk == nkey_array[j].yk) {
			nvort_array[j] += nnvort_array[i];
		}
		else
		{
			++j;
			nkey_array[j] = nnkey_array[i];
			nvort_array[j] = nnvort_array[i];
		}
	}
	free(nnkey_array);
	free(nnvort_array);
	n_created_particles = (j < ppop * n_input_particles ? j : 0);

	/* Remove particles with neglidgible vorticity. */
	ave_particle_str = 0.f;
	for (i = 0; i < n_created_particles; ++i) { 
		ave_particle_str += fabsf(nvort_array[i]); 
	}
	ave_particle_str /= n_created_particles;
	min_keepable_particle = ave_particle_str * negligible_vort;
	j = 0; vorticity_deficit = 0;
	for (i = 0; i < n_created_particles; ++i) {
		if (fabsf(nvort_array[i]) >= min_keepable_particle) {
			nvort_array[j] = nvort_array[i];
			nkey_array[j] = nkey_array[i];
			++j;
		}
		else {
			vorticity_deficit += nvort_array[i];	/* For vorticity conservation. */
		}
	}
	n_created_particles = j;
	for (i = 0; i < n_created_particles; ++i) {
		nvort_array[i] += vorticity_deficit / n_created_particles;
	}

	/* Now to handle what we return to the caller */
	if (output_particles != NULL) {
		if (n_created_particles > max_output_particles) {
			/* We need to remove more particles... */
			minv = maxv = fabsf(nvort_array[0]);
			ave_particle_str = 0.f;
			for (i = 0; i < n_created_particles; ++i) {
				minv = minv < fabsf(nvort_array[i]) ? minv : fabsf(nvort_array[i]);
				maxv = maxv > fabsf(nvort_array[i]) ? maxv : fabsf(nvort_array[i]);
				ave_particle_str += fabsf(nvort_array[i]);
			}
			ave_particle_str /= n_created_particles;

			/* Guess vorticity thresholds and count the kept particles.*/
			float guesses[NG_FOR_REDUCING_PARICLES];
			int grms[NG_FOR_REDUCING_PARICLES];
			while (1) {
				for (i = 0; i < NG_FOR_REDUCING_PARICLES; ++i) {
					grms[i] = 0;
					guesses[i] = minv + (i + 1) * (maxv - minv) / (NG_FOR_REDUCING_PARICLES + 1);
				}
				for (i = 0; i < n_created_particles; ++i) {
					for (j = 0; j < 16; ++j) {
						grms[j] += fabsf(nvort_array[i]) < guesses[j] ? 0 : 1;
					}
				}
				for (i = NG_FOR_REDUCING_PARICLES-2; i >= 0; ++i) {
					maxv = guesses[i + 1];
					minv = guesses[i];
					if (grms[i] < max_output_particles) { 
						k = i;
						break;
					}
				}
				/* Termination condition. */
				if (minv == maxv || grms[k] == grms[k + 1] || 
					((float)(max_output_particles - grms[k]) 
						/ ((float)max_output_particles)) < 0.05) {
					break;
				}
			}
			min_keepable_particle = guesses[k];
			for (i = 0; i < n_created_particles; ++i) {
				if (fabsf(nvort_array[i]) >= min_keepable_particle) {
					nvort_array[j] = nvort_array[i];
					nkey_array[j] = nkey_array[i];
					++j;
				}
				else {
					vorticity_deficit += nvort_array[i];	/* For vorticity conservation. */
				}
			}
			n_created_particles = j;
			for (i = 0; i < n_created_particles; ++i) {
				nvort_array[i] += vorticity_deficit / n_created_particles;
			}
		}
		/* And now make an array to return to our caller. */
		for (i = 0; i < n_created_particles; ++i) {
			output_particles[i].area = grid_density * grid_density;
			output_particles[i].vorticity = nvort_array[i];
			output_particles[i].coord.x[0] = minx + nkey_array[i].xk * grid_density;
			output_particles[i].coord.x[1] = miny + nkey_array[i].yk * grid_density;
		}
	}

	/* Free remaining arrays. */
	free(nidx_array);
	free(nkey_array);
	free(nvort_array);
	return n_created_particles;
}
