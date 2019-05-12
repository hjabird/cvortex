#include "libcvtx.h"
/*============================================================================
Particle.c

Basic representation of a vortex particle.

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
#include <stddef.h>

#ifdef CVTX_USING_OPENCL
#	include "ocl_particle.h"
#endif

CVTX_EXPORT bsv_V3f cvtx_Particle_ind_vel(
	const cvtx_Particle * self,
	const bsv_V3f mes_point,
	const cvtx_VortFunc * kernel,
	float regularisation_radius)
{
	bsv_V3f rad, num, ret;
	if(bsv_V3f_isequal(self->coord, mes_point)){
		ret = bsv_V3f_zero();
	} else {
		float cor, den, rho;
		rad = bsv_V3f_minus(mes_point, self->coord);
		rho = fabsf(bsv_V3f_abs(rad) / regularisation_radius);
		cor = - kernel->g_fn(rho) / ((float)4. * (float)acos(-1));
		den = powf(bsv_V3f_abs(rad), 3);
		num = bsv_V3f_cross(rad, self->vorticity);
		ret = bsv_V3f_mult(num, cor / den);
	}
	return ret;
}

CVTX_EXPORT bsv_V3f cvtx_Particle_ind_dvort(
	const cvtx_Particle * self,
	const cvtx_Particle * induced_particle,
	const cvtx_VortFunc * kernel,
	float regularisation_radius)
{
	bsv_V3f ret, rad, cross_om, t2, t21, t21n, t22, t224;
	float g, f, radd, rho, t1, t21d, t221, t222, t223;
	if(bsv_V3f_isequal(self->coord, induced_particle->coord)){
		ret = bsv_V3f_zero();
	} else {
		rad = bsv_V3f_minus(induced_particle->coord, self->coord);
		radd = bsv_V3f_abs(rad);
		rho = fabsf(radd / regularisation_radius);
		kernel->combined_fn(rho, &g, &f);
		cross_om = bsv_V3f_cross(induced_particle->vorticity, self->vorticity);
		t1 = (float)1. / ((float)4. * (float)acos(-1) * powf(regularisation_radius, 3));
		t21n = bsv_V3f_mult(cross_om, -g);
		t21d = rho * rho * rho;
		t21 = bsv_V3f_div(t21n, t21d);
		t221 = (float)1. / (radd * radd);
		t222 = (3 * g) / (rho * rho * rho) - f;
		t223 = bsv_V3f_dot(induced_particle->vorticity, rad);
		t224 = bsv_V3f_cross(rad, self->vorticity);
		t22 = bsv_V3f_mult(t224, t221 * t222 * t223);
		t2 = bsv_V3f_plus(t21, t22);
		ret = bsv_V3f_mult(t2, t1);
	}
	return ret;
}

float sphere_volume(float radius){
	return 4 * (float)acos(-1) * radius * radius * radius / (float) 3.;
}

CVTX_EXPORT bsv_V3f cvtx_Particle_visc_ind_dvort(
	const cvtx_Particle * self,
	const cvtx_Particle * induced_particle,
	const cvtx_VortFunc * kernel,
	float regularisation_radius,
	float kinematic_visc)
{	
	bsv_V3f ret, rad, t211, t212, t21, t2;
	float radd, rho, t1, t22;
	assert(kernel->eta_fn != NULL && "Used vortex regularisation"
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
		t22 = kernel->eta_fn(rho);
		t2 = bsv_V3f_mult(t21, t22);
		ret = bsv_V3f_mult(t2, t1);
	}
	return ret;
}

CVTX_EXPORT bsv_V3f cvtx_ParticleArr_ind_vel(
	const cvtx_Particle **array_start,
	const int num_particles,
	const bsv_V3f mes_point,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
	bsv_V3f vel;
	double rx = 0, ry = 0, rz = 0;
	long i;
	assert(num_particles >= 0);
	for (i = 0; i < num_particles; ++i) {
		vel = cvtx_Particle_ind_vel(array_start[i],
			mes_point, kernel, regularisation_radius);
		rx += vel.x[0];
		ry += vel.x[1];
		rz += vel.x[2];
	}
	bsv_V3f ret = {(float)rx, (float)ry, (float)rz};
	return ret;
}

CVTX_EXPORT bsv_V3f cvtx_ParticleArr_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
	bsv_V3f dvort;
	double rx = 0, ry = 0, rz = 0;
	long i;
	assert(num_particles >= 0);
	for (i = 0; i < num_particles; ++i) {
		dvort = cvtx_Particle_ind_dvort(array_start[i],
			induced_particle, kernel, regularisation_radius);
		rx += dvort.x[0];
		ry += dvort.x[1];
		rz += dvort.x[2];
	}
	bsv_V3f ret = {(float)rx, (float)ry, (float)rz};
	return ret;
}

CVTX_EXPORT bsv_V3f cvtx_ParticleArr_visc_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc)
{
	bsv_V3f dvort;
	double rx = 0, ry = 0, rz = 0;
	long i;
	assert(num_particles >= 0);
	for (i = 0; i < num_particles; ++i) {
		dvort = cvtx_Particle_visc_ind_dvort(array_start[i],
			induced_particle, kernel, regularisation_radius, kinematic_visc);
		rx += dvort.x[0];
		ry += dvort.x[1];
		rz += dvort.x[2];
	}
	bsv_V3f ret = {(float)rx, (float)ry, (float)rz};
	return ret;
}

static void cpu_brute_force_ParticleArr_Arr_ind_vel(
	const cvtx_Particle **array_start,
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
		result_array[i] = cvtx_ParticleArr_ind_vel(
			array_start, num_particles, mes_start[i], 
			kernel, regularisation_radius);
	}
	return;
}

CVTX_EXPORT void cvtx_ParticleArr_Arr_ind_vel(
	const cvtx_Particle **array_start,
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
		|| kernel->cl_kernel_name_ext == ""
		|| opencl_brute_force_ParticleArr_Arr_ind_vel(
			array_start, num_particles, mes_start,
			num_mes, result_array, kernel, regularisation_radius) != 0)
#endif
	{
		cpu_brute_force_ParticleArr_Arr_ind_vel(
			array_start, num_particles, mes_start,
			num_mes, result_array, kernel, regularisation_radius);
	}
	return;
}

void cpu_brute_force_ParticleArr_Arr_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
	long i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < num_induced; ++i) {
		result_array[i] = cvtx_ParticleArr_ind_dvort(
			array_start, num_particles, induced_start[i], 
			kernel, regularisation_radius);
	}
	return;
}

CVTX_EXPORT void cvtx_ParticleArr_Arr_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
#ifdef CVTX_USING_OPENCL
	if (	num_particles < 256
		||	num_induced < 256
		||	kernel->cl_kernel_name_ext == ""
		||	opencl_brute_force_ParticleArr_Arr_ind_dvort(
				array_start, num_particles, induced_start,
				num_induced, result_array, kernel, regularisation_radius) != 0)
#endif
	{
		cpu_brute_force_ParticleArr_Arr_ind_dvort(
			array_start, num_particles, induced_start,
			num_induced, result_array, kernel, regularisation_radius);
	}
	return;
}

void cpu_brute_force_ParticleArr_Arr_visc_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc)
{
	long i;
#pragma omp parallel for schedule(static)
	for (i = 0; i < num_induced; ++i) {
		result_array[i] = cvtx_ParticleArr_visc_ind_dvort(
			array_start, num_particles, induced_start[i],
			kernel, regularisation_radius, kinematic_visc);
	}
	return;
}

CVTX_EXPORT void cvtx_ParticleArr_Arr_visc_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc)
{
#ifdef CVTX_USING_OPENCL
	if (	num_particles < 256
		||	num_induced < 256
		||	kernel->cl_kernel_name_ext == ""
		||	opencl_brute_force_ParticleArr_Arr_visc_ind_dvort(
				array_start, num_particles, induced_start,
				num_induced, result_array, kernel, regularisation_radius, kinematic_visc) != 0)
#endif
	{
		cpu_brute_force_ParticleArr_Arr_visc_ind_dvort(
			array_start, num_particles, induced_start,
			num_induced, result_array, kernel, regularisation_radius, kinematic_visc);
	}
	return;

}
