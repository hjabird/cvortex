#include "libcvtx.h"
/*============================================================================
P3D.c

Vortex particle in 2D with CPU based code.

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

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "redistribution_helper_funcs.h"
#include "uintkey.h"

#ifdef CVTX_USING_OPENCL
#	include "ocl_P3D.h"
#endif

#define CVTX_PI_F 3.14159265359f

/* The induced velocity for a particle excluding the constant
coefficient 1 / 4pi */
static inline bsv_V3f P3D_vel_inner(
	const cvtx_P3D * self,
	const bsv_V3f mes_point,
	const cvtx_VortFunc * kernel,
	float recip_reg_rad)
{
	bsv_V3f rad, num, ret;
	if (bsv_V3f_isequal(self->coord, mes_point)) {
		ret = bsv_V3f_zero();
	}
	else {
		float cor, den, rho, radd;
		rad = bsv_V3f_minus(mes_point, self->coord);
		radd = bsv_V3f_abs(rad);
		rho = radd * recip_reg_rad; /* Assume positive. */
		cor = -kernel->g_3D(rho);
		den = powf(radd, -3);
		num = bsv_V3f_cross(rad, self->vorticity);
		ret = bsv_V3f_mult(num, cor * den);
	}
	return ret;
}

CVTX_EXPORT bsv_V3f cvtx_P3D_S2S_vel(
	const cvtx_P3D * self,
	const bsv_V3f mes_point,
	const cvtx_VortFunc * kernel,
	float regularisation_radius)
{
	bsv_V3f ret;
	ret = P3D_vel_inner(self, mes_point, kernel, 
		1.f/fabsf(regularisation_radius));
	return bsv_V3f_mult(ret, 1.f / (4.f * CVTX_PI_F));
}

CVTX_EXPORT bsv_V3f cvtx_P3D_S2S_dvort(
	const cvtx_P3D * self,
	const cvtx_P3D * induced_particle,
	const cvtx_VortFunc * kernel,
	float regularisation_radius)
{
	bsv_V3f ret, rad, cross_om, t2, t21, t21n, t22;
	float g, f, radd, rho, t1, t21d, t221, t222, t223;
	if(bsv_V3f_isequal(self->coord, induced_particle->coord)){
		ret = bsv_V3f_zero();
	} else {
		rad = bsv_V3f_minus(induced_particle->coord, self->coord);
		radd = bsv_V3f_abs(rad);
		rho = fabsf(radd / regularisation_radius);
		kernel->combined_3D(rho, &g, &f);
		cross_om = bsv_V3f_cross(induced_particle->vorticity, self->vorticity);
		t1 = 1.f / (4.f * CVTX_PI_F * powf(regularisation_radius, 3));
		t21n = bsv_V3f_mult(cross_om, g);
		t21d = rho * rho * rho;
		t21 = bsv_V3f_div(t21n, t21d);
		t221 = -1.f / (radd * radd);
		t222 = (3 * g) / t21d - f;
		t223 = bsv_V3f_dot(rad, cross_om);
		t22 = bsv_V3f_mult(rad, t221 * t222 * t223);
		t2 = bsv_V3f_plus(t21, t22);
		ret = bsv_V3f_mult(t2, t1);
	}
	return ret;
}

CVTX_EXPORT bsv_V3f cvtx_P3D_S2S_visc_dvort(
	const cvtx_P3D * self,
	const cvtx_P3D * induced_particle,
	const cvtx_VortFunc * kernel,
	float regularisation_radius,
	float kinematic_visc)
{	
	bsv_V3f ret, rad, t211, t212, t21, t2;
	float radd, rho, t1, t22;
	assert(kernel->eta_3D != NULL && "Used vortex regularisation"
		"that did have a defined eta function");
	if(bsv_V3f_isequal(self->coord, induced_particle->coord)){
		ret = bsv_V3f_zero();
	} else {
		rad = bsv_V3f_minus(self->coord, induced_particle->coord);
		radd = bsv_V3f_abs(rad);
		rho = fabsf(radd / regularisation_radius);
		t1 =  2 * kinematic_visc / powf(regularisation_radius, 2);
		t211 = bsv_V3f_mult(self->vorticity, 
			induced_particle->volume);
		t212 = bsv_V3f_mult(induced_particle->vorticity, 
			-1 * self->volume);
		t21 = bsv_V3f_plus(t211, t212);
		t22 = kernel->eta_3D(rho);
		t2 = bsv_V3f_mult(t21, t22);
		ret = bsv_V3f_mult(t2, t1);
	}
	return ret;
}

CVTX_EXPORT bsv_V3f cvtx_P3D_S2S_vort(
	const cvtx_P3D* self,
	const bsv_V3f mes_point,
	const cvtx_VortFunc* kernel,
	float regularisation_radius) {
	bsv_V3f rad, ret;
	float radd, coeff, divisor;
	rad = bsv_V3f_minus(self->coord, mes_point);
	radd = bsv_V3f_abs(rad);
	coeff = kernel->zeta_3D(radd / regularisation_radius);
	divisor = 4.f * CVTX_PI_F * 
		regularisation_radius * regularisation_radius * regularisation_radius;
	coeff = coeff / divisor;
	ret = bsv_V3f_mult(self->vorticity, coeff);
	return ret;
}

CVTX_EXPORT bsv_V3f cvtx_P3D_M2S_vel(
	const cvtx_P3D **array_start,
	const int num_particles,
	const bsv_V3f mes_point,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
	double rx = 0, ry = 0, rz = 0;
	long i;
	float recip_reg_rad = 1.f / fabsf(regularisation_radius);
	assert(num_particles >= 0);
#pragma omp parallel for reduction(+:rx, ry, rz)
	for (i = 0; i < num_particles; ++i) {
		bsv_V3f vel = P3D_vel_inner(array_start[i],
			mes_point, kernel, recip_reg_rad);
		rx += vel.x[0];
		ry += vel.x[1];
		rz += vel.x[2];
	}
	bsv_V3f ret = {(float)rx, (float)ry, (float)rz};
	return bsv_V3f_mult(ret, 1.f / (4.f * CVTX_PI_F));
}

CVTX_EXPORT bsv_V3f cvtx_P3D_M2S_dvort(
	const cvtx_P3D **array_start,
	const int num_particles,
	const cvtx_P3D *induced_particle,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
	bsv_V3f dvort;
	double rx = 0, ry = 0, rz = 0;
	long i;
	assert(num_particles >= 0);
	for (i = 0; i < num_particles; ++i) {
		dvort = cvtx_P3D_S2S_dvort(array_start[i],
			induced_particle, kernel, regularisation_radius);
		rx += dvort.x[0];
		ry += dvort.x[1];
		rz += dvort.x[2];
	}
	bsv_V3f ret = {(float)rx, (float)ry, (float)rz};
	return ret;
}

CVTX_EXPORT bsv_V3f cvtx_P3D_M2S_visc_dvort(
	const cvtx_P3D **array_start,
	const int num_particles,
	const cvtx_P3D *induced_particle,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc)
{
	bsv_V3f dvort;
	double rx = 0, ry = 0, rz = 0;
	long i;
	assert(num_particles >= 0);
	for (i = 0; i < num_particles; ++i) {
		dvort = cvtx_P3D_S2S_visc_dvort(array_start[i],
			induced_particle, kernel, regularisation_radius, kinematic_visc);
		rx += dvort.x[0];
		ry += dvort.x[1];
		rz += dvort.x[2];
	}
	bsv_V3f ret = {(float)rx, (float)ry, (float)rz};
	return ret;
}

CVTX_EXPORT bsv_V3f cvtx_P3D_M2S_vort(
	const cvtx_P3D** array_start,
	const int num_particles,
	const bsv_V3f mes_point,
	const cvtx_VortFunc* kernel,
	float regularisation_radius) {
	float cutoff, rsigma, radd, coeff;
	bsv_V3f rad, sum = bsv_V3f_zero();
	long i;
	cutoff = 5.f * regularisation_radius;
	rsigma = 1 / regularisation_radius;
	assert(num_particles > 0);
	for (i = 0; i < num_particles; ++i) {
		rad = bsv_V3f_minus(array_start[i]->coord, mes_point);
		if (fabsf(rad.x[0]) < cutoff && fabsf(rad.x[1]) < cutoff
			&& fabsf(rad.x[2]) < cutoff) {
			radd = bsv_V3f_abs(rad);
			coeff = kernel->zeta_3D(radd * rsigma);
			sum = bsv_V3f_plus(bsv_V3f_mult(array_start[i] ->vorticity, coeff), sum);
		}
	}
	sum = bsv_V3f_div(sum, 4.f * CVTX_PI_F 
		* regularisation_radius * regularisation_radius * regularisation_radius);
	return sum;
} 

static void cpu_brute_force_P3D_M2M_vel(
	const cvtx_P3D **array_start,
	const int num_particles,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
	long i;
#pragma omp parallel for schedule(static)
	for(i = 0; i < num_mes; ++i){
		result_array[i] = cvtx_P3D_M2S_vel(
			array_start, num_particles, mes_start[i], 
			kernel, regularisation_radius);
	}
	return;
}

CVTX_EXPORT void cvtx_P3D_M2M_vel(
	const cvtx_P3D **array_start,
	const int num_particles,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
#ifdef CVTX_USING_OPENCL
	if (num_particles < 256
		|| num_mes < 256
		|| !strcmp(kernel->cl_kernel_name_ext, "")
		|| opencl_brute_force_P3D_M2M_vel(
			array_start, num_particles, mes_start,
			num_mes, result_array, kernel, regularisation_radius) != 0)
#endif
	{
		cpu_brute_force_P3D_M2M_vel(
			array_start, num_particles, mes_start,
			num_mes, result_array, kernel, regularisation_radius);
	}
	return;
}

void cpu_brute_force_P3D_M2M_dvort(
	const cvtx_P3D **array_start,
	const int num_particles,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
	long i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < num_induced; ++i) {
		result_array[i] = cvtx_P3D_M2S_dvort(
			array_start, num_particles, induced_start[i], 
			kernel, regularisation_radius);
	}
	return;
}

CVTX_EXPORT void cvtx_P3D_M2M_dvort(
	const cvtx_P3D **array_start,
	const int num_particles,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
#ifdef CVTX_USING_OPENCL
	if (	num_particles < 256
		||	num_induced < 256
		||	!strcmp(kernel->cl_kernel_name_ext, "")
		||	opencl_brute_force_P3D_M2M_dvort(
				array_start, num_particles, induced_start,
				num_induced, result_array, kernel, regularisation_radius) != 0)
#endif
	{
		cpu_brute_force_P3D_M2M_dvort(
			array_start, num_particles, induced_start,
			num_induced, result_array, kernel, regularisation_radius);
	}
	return;
}

void cpu_brute_force_P3D_M2M_visc_dvort(
	const cvtx_P3D **array_start,
	const int num_particles,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc)
{
	long i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < num_induced; ++i) {
		result_array[i] = cvtx_P3D_M2S_visc_dvort(
			array_start, num_particles, induced_start[i],
			kernel, regularisation_radius, kinematic_visc);
	}
	return;
}

CVTX_EXPORT void cvtx_P3D_M2M_visc_dvort(
	const cvtx_P3D **array_start,
	const int num_particles,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc)
{
#ifdef CVTX_USING_OPENCL
	if (	num_particles < 256
		||	num_induced < 256
		||	!strcmp(kernel->cl_kernel_name_ext, "")
		||	opencl_brute_force_P3D_M2M_visc_dvort(
				array_start, num_particles, induced_start,
				num_induced, result_array, kernel, regularisation_radius, kinematic_visc) != 0)
#endif
	{
		cpu_brute_force_P3D_M2M_visc_dvort(
			array_start, num_particles, induced_start,
			num_induced, result_array, kernel, regularisation_radius, kinematic_visc);
	}
	return;
}

CVTX_EXPORT void cpu_brute_force_P3D_M2M_vort(
	const cvtx_P3D** array_start,
	const int num_particles,
	const bsv_V3f* mes_start,
	const int num_mes,
	bsv_V3f* result_array,
	const cvtx_VortFunc* kernel,
	float regularisation_radius) {
	long i;
#pragma omp parallel for schedule(guided)
	for (i = 0; i < num_mes; ++i) {
		result_array[i] = cvtx_P3D_M2S_vort(
			array_start, num_particles, mes_start[i],
			kernel, regularisation_radius);
	}
	return;
}

CVTX_EXPORT void cvtx_P3D_M2M_vort(
	const cvtx_P3D** array_start,
	const int num_particles,
	const bsv_V3f* mes_start,
	const int num_mes,
	bsv_V3f* result_array,
	const cvtx_VortFunc* kernel,
	float regularisation_radius) {
#ifdef CVTX_USING_OPENCL
	if (num_particles < 256
		|| num_mes < 256
		|| !strcmp(kernel->cl_kernel_name_ext, "")
		|| opencl_brute_force_P3D_M2M_vort(
			array_start, num_particles, mes_start,
			num_mes, result_array, kernel, regularisation_radius) != 0)
#endif
	{
		cpu_brute_force_P3D_M2M_vort(
			array_start, num_particles, mes_start,
			num_mes, result_array, kernel, regularisation_radius);
	}
	return;
}


/* Particle redistribution -------------------------------------------------*/

/* Modifies io_arr to remove particles under a strength threshold,
returning the number of particles. */
static int cvtx_remove_particles_under_str_threshold(
	cvtx_P3D *io_arr, float* strs, int n_inpt_partices, float threshold,
	int max_keepable);

CVTX_EXPORT int cvtx_P3D_redistribute_on_grid(
	const cvtx_P3D **input_array_start,
	const int n_input_particles,
	cvtx_P3D *output_particles,		/* input is &(*cvtx_P3D) to write to */
	int max_output_particles,		/* Set to resultant num particles.   */
	const cvtx_RedistFunc *redistributor,
	float grid_density,
	float negligible_vort) {

	assert(n_input_particles >= 0);
	assert(max_output_particles >= 0);
	assert(grid_density > 0.f);
	assert(negligible_vort >= 0.f);
	assert(negligible_vort < 1.f);

	int i, j, k, m, n_created_particles; (void)m; (void)k; /* Avoid msvc err*/
	int grid_radius;				/* Value of U that returns zero.	*/
	int ppop;						/* Particles per input particle.	*/
	float minx, miny, minz;			/* Bounds of the particle box.		*/
	/* Index array and grid location array of input particles.			*/
	unsigned int *oidx_array = NULL;
	UInt32Key3D *okey_array = NULL;
	/* Index, grid location and vorticity arrays of new particles.		*/
	UInt32Key3D *nkey_array = NULL, *nnkey_array = NULL;
	bsv_V3f *nvort_array = NULL, *nnvort_array = NULL;
	unsigned int *nidx_array = NULL;
	/* For particle removal: */
	float min_keepable_particle;

	/* Generate grid keys for existing particles. */
	minmax_xyz_posn(input_array_start, n_input_particles, 
		&minx, NULL, &miny, NULL, &minz, NULL);
	grid_radius = (int)roundf(redistributor->radius);
	
	minx -= (grid_radius + (float)rand() / (float)(RAND_MAX)) * grid_density;
	miny -= (grid_radius + (float)rand() / (float)(RAND_MAX)) * grid_density;
	minz -= (grid_radius + (float)rand() / (float)(RAND_MAX)) * grid_density;

	oidx_array = malloc(sizeof(unsigned int) * n_input_particles);
	okey_array = malloc(sizeof(UInt32Key3D) * n_input_particles);
#pragma omp parallel for schedule(static)
	for (i = 0; i < n_input_particles; ++i) {
		oidx_array[i] = i;
		okey_array[i] = g_P3D_gridkey3D(input_array_start[i],
			grid_density, minx, miny, minz);
	}

	/* Now we make new particles based on grid. */
	/* ppop = Particles per orginal particle. */
	ppop = (grid_radius * 2 + 1) * (grid_radius * 2 + 1) 
		* (grid_radius * 2 + 1);
	nkey_array = malloc(sizeof(UInt32Key3D) * ppop * n_input_particles);
	nvort_array = malloc(sizeof(bsv_V3f) * ppop * n_input_particles);
	nidx_array = malloc(sizeof(unsigned int) * ppop * n_input_particles);
#pragma omp parallel for schedule(static) private(j, k, m)
	for (i = 0; i < n_input_particles; ++i) {
		int widx = oidx_array[i];
		unsigned int okx, oky, okz;
		bsv_V3f p_coord, p_vort;
		okx = okey_array[widx].k.x;
		oky = okey_array[widx].k.y;
		okz = okey_array[widx].k.z;
		p_coord = input_array_start[widx]->coord;
		p_vort = input_array_start[widx]->vorticity;
		for (j = -grid_radius; j <= grid_radius; ++j) {
			for (k = -grid_radius; k <= grid_radius; ++k) {
				for (m = -grid_radius; m <= grid_radius; ++m) {
					float U, W, V, vortfrac;
					bsv_V3f n_coord, dx;
					int np_idx;
					n_coord.x[0] = okx * grid_density + j * grid_density + minx;
					n_coord.x[1] = oky * grid_density + k * grid_density + miny;
					n_coord.x[2] = okz * grid_density + m * grid_density + minz;
					dx = bsv_V3f_minus(p_coord, n_coord);
					U = fabsf(dx.x[0] / grid_density);
					W = fabsf(dx.x[1] / grid_density);
					V = fabsf(dx.x[2] / grid_density);
					vortfrac = redistributor->func(U) * redistributor->func(W)
						* redistributor->func(V);
					np_idx = m + grid_radius +
						(k + grid_radius) * (2 * grid_radius + 1) +
						(j + grid_radius) * (2 * grid_radius + 1) 
						* (2 * grid_radius + 1) + widx * ppop;
					nkey_array[np_idx].k.x = okx + j;
					nkey_array[np_idx].k.y = oky + k;
					nkey_array[np_idx].k.z = okz + m;
					nvort_array[np_idx] = bsv_V3f_mult(p_vort, vortfrac);
				}
			}
		}
	}
	free(oidx_array);
	free(okey_array);

	/* Now merge our new particles */
	sort_perm_UInt32Key3D(nkey_array, nidx_array, n_input_particles* ppop);

	nnkey_array = malloc(sizeof(UInt32Key3D) * n_input_particles * ppop);
	nnvort_array = malloc(sizeof(bsv_V3f) * n_input_particles * ppop);
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
		if (nnkey_array[i].k.x == nkey_array[j].k.x &&
			nnkey_array[i].k.y == nkey_array[j].k.y &&
			nnkey_array[i].k.z == nkey_array[j].k.z) {
			nvort_array[j] = bsv_V3f_plus(nnvort_array[i], nvort_array[j]);
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
	n_created_particles = (j < ppop * n_input_particles ? j + 1 : 0);

	/* Go back to array of particles. */
	cvtx_P3D *created_particles = NULL;
	created_particles = malloc(n_created_particles * sizeof(cvtx_P3D));
#pragma omp parallel for
	for (i = 0; i < n_created_particles; ++i) {
		created_particles[i].volume = grid_density * grid_density * grid_density;
		created_particles[i].vorticity = nvort_array[i];
		created_particles[i].coord.x[0] = minx + nkey_array[i].k.x * grid_density;
		created_particles[i].coord.x[1] = miny + nkey_array[i].k.y * grid_density;
		created_particles[i].coord.x[2] = minz + nkey_array[i].k.z * grid_density;
	}
	free(nkey_array);
	free(nvort_array);
	free(nidx_array);
	
	/* Remove particles with neglidgible vorticity. */
	float* strengths = malloc(sizeof(float) * n_created_particles);
#pragma omp parallel for
	for (i = 0; i < n_created_particles; ++i) {
		strengths[i] = bsv_V3f_abs(created_particles[i].vorticity);
	}
	farray_info(strengths, n_created_particles, &min_keepable_particle, NULL, NULL);
	min_keepable_particle = min_keepable_particle * negligible_vort;
	n_created_particles = cvtx_remove_particles_under_str_threshold(
		created_particles, strengths, n_created_particles, 
		min_keepable_particle, n_created_particles);
	/* The strengths are modified to keep total vorticity constant. */
#pragma omp parallel for
	for (i = 0; i < n_created_particles; ++i) {
		strengths[i] = bsv_V3f_abs(created_particles[i].vorticity);
	}

	/* Now to handle what we return to the caller */
	if (output_particles != NULL) {
		if (n_created_particles > max_output_particles) {
			min_keepable_particle = get_strength_threshold(
				strengths, n_created_particles, max_output_particles);
			n_created_particles = cvtx_remove_particles_under_str_threshold(created_particles,
				strengths, n_created_particles, min_keepable_particle, max_output_particles);
		}
		/* And now make an array to return to our caller. */
		memcpy(output_particles, created_particles, sizeof(cvtx_P3D) * n_created_particles);
	}
	/* Free remaining arrays. */
	free(created_particles);
	return n_created_particles;
}

int cvtx_remove_particles_under_str_threshold(
	cvtx_P3D* io_arr, float* strs, 
	int n_inpt_partices, float min_keepable_str,
	int max_keepable) {

	bsv_V3f vorticity_deficit;
	int n_output_particles;
	int i, j;

	j = 0; 
	vorticity_deficit = bsv_V3f_zero();

	for (i = 0; i < n_inpt_partices; ++i) {
		if (strs[i] > min_keepable_str && i < max_keepable) {
			io_arr[j] = io_arr[i];
			++j;
		}
		else {
			/* For vorticity conservation. */
			vorticity_deficit = bsv_V3f_plus(io_arr[i].vorticity, vorticity_deficit);
		}
	}

	n_output_particles = j;
	vorticity_deficit = bsv_V3f_div(vorticity_deficit, (float)n_output_particles);
	for (i = 0; i < n_output_particles; ++i) {
		io_arr[i].vorticity = bsv_V3f_plus(io_arr[i].vorticity, vorticity_deficit);
	}
	return n_output_particles;
}

/* Relaxation --------------------------------------------------------------*/

CVTX_EXPORT void cvtx_P3D_pedrizzetti_relaxation(
	cvtx_P3D** input_array_start,
	const int n_input_particles,
	float fdt,
	const cvtx_VortFunc* kernel,
	float regularisation_radius) {
	/* Pedrizzetti relaxation scheme: 
		alpha_new =	(1-fq * dt) * alpha_old
					+ fq * dt * omega(x) * abs(alpha_old) / abs(omega(x)) */
	bsv_V3f *mes_posns = NULL, *omegas = NULL;
	int i;
	float tmp;
	mes_posns = malloc(sizeof(bsv_V3f) * n_input_particles);
	omegas = malloc(sizeof(bsv_V3f) * n_input_particles);
	cvtx_P3D_M2M_vort(input_array_start, n_input_particles, 
		mes_posns, n_input_particles, omegas, kernel, regularisation_radius);

	tmp = 1.f - fdt;
#pragma omp parallel for
	for (i = 0; i < n_input_particles; ++i) {
		bsv_V3f ovort, nvort;
		float coeff;
		ovort = input_array_start[i]->vorticity;
		coeff = bsv_V3f_abs(ovort) / bsv_V3f_abs(omegas[i]);
		nvort = bsv_V3f_mult(ovort, tmp);
		nvort = bsv_V3f_plus(nvort, bsv_V3f_mult(omegas[i], coeff));
		input_array_start[i]->vorticity = nvort;
	}

	free(mes_posns);
	free(omegas);
	return;
}
