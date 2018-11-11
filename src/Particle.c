#include "Particle.h"
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

inline float sphere_volume(float radius);

cvtx_Vec3f cvtx_Particle_ind_vel(
	const cvtx_Particle * self,
	const cvtx_Vec3f mes_point,
	const cvtx_VortFunc * kernel)
{
	cvtx_Vec3f rad, num, ret;
	if(cvtx_Vec3f_isequal(self->coord, mes_point)){
		ret = cvtx_Vec3f_zero();
	} else {
		float cor, den, rho;
		rad = cvtx_Vec3f_minus(mes_point, self->coord);
		rho = fabsf(cvtx_Vec3f_abs(rad) / self->radius);
		cor = - kernel->g_fn(rho) / ((float)4. * (float)acos(-1));
		den = powf(cvtx_Vec3f_abs(rad), 3);
		num = cvtx_Vec3f_cross(rad, self->vorticity);
		ret = cvtx_Vec3f_mult(num, cor / den);
	}
	return ret;
}

cvtx_Vec3f cvtx_Particle_ind_dvort(
	const cvtx_Particle * self,
	const cvtx_Particle * induced_particle,
	const cvtx_VortFunc * kernel)
{
	cvtx_Vec3f ret, rad, cross_om, t2, t21, t21n, t22, t224;
	float g, f, radd, rho, t1, t21d, t221, t222, t223;
	if(cvtx_Vec3f_isequal(self->coord, induced_particle->coord)){
		ret = cvtx_Vec3f_zero();
	} else {
		rad = cvtx_Vec3f_minus(induced_particle->coord, self->coord);
		radd = cvtx_Vec3f_abs(rad);
		rho = fabsf(radd / self->radius);
		kernel->combined_fn(rho, &g, &f);
		cross_om = cvtx_Vec3f_cross(induced_particle->vorticity, self->vorticity);
		t1 = (float)1. / ((float)4. * (float)acos(-1) * powf(self->radius, 3));
		t21n = cvtx_Vec3f_mult(cross_om, -g);
		t21d = rho * rho * rho;
		t21 = cvtx_Vec3f_div(t21n, t21d);
		t221 = (float)1. / (radd * radd);
		t222 = (3 * g) / (rho * rho * rho) - f;
		t223 = cvtx_Vec3f_dot(induced_particle->vorticity, rad);
		t224 = cvtx_Vec3f_cross(rad, self->vorticity);
		t22 = cvtx_Vec3f_mult(t224, t221 * t222 * t223);
		t2 = cvtx_Vec3f_plus(t21, t22);
		ret = cvtx_Vec3f_mult(t2, t1);
	}
	return ret;
}

float sphere_volume(float radius){
	return 4 * (float)acos(-1) * radius * radius * radius / (float) 3.;
}

cvtx_Vec3f cvtx_Particle_visc_ind_dvort(
	const cvtx_Particle * self,
	const cvtx_Particle * induced_particle,
	const cvtx_VortFunc * kernel,
	const float kinematic_visc)
{	
	cvtx_Vec3f ret, rad, t211, t212, t21, t2;
	float radd, rho, t1, t22;
	assert(kernel->eta_fn != NULL && "Used vortex regularisation"
		"that did have a defined eta function");
	if(cvtx_Vec3f_isequal(self->coord, induced_particle->coord)){
		ret = cvtx_Vec3f_zero();
		
	} else {
		rad = cvtx_Vec3f_minus(induced_particle->coord, self->coord);
		radd = cvtx_Vec3f_abs(rad);
		rho = fabsf(radd / self->radius);
		t1 =  2 * kinematic_visc / powf(induced_particle->radius, 2);
		t211 = cvtx_Vec3f_mult(self->vorticity, 
			sphere_volume(induced_particle->radius));
		t212 = cvtx_Vec3f_mult(induced_particle->vorticity, 
			-1 * sphere_volume(self->radius));
		t21 = cvtx_Vec3f_plus(t211, t212);
		t22 = kernel->eta_fn(rho);
		t2 = cvtx_Vec3f_mult(t21, t22);
		ret = cvtx_Vec3f_mult(t2, t1);
	}
	return ret;
}

cvtx_Vec3f cvtx_ParticleArr_ind_vel(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Vec3f mes_point,
	const cvtx_VortFunc *kernel)
{
	cvtx_Vec3f vel;
	double rx = 0, ry = 0, rz = 0;
	int i;
	assert(num_particles >= 0);
	for (i = 0; i < num_particles; ++i) {
		vel = cvtx_Particle_ind_vel(array_start[i],
			mes_point, kernel);
		rx += vel.x[0];
		ry += vel.x[1];
		rz += vel.x[2];
	}
	cvtx_Vec3f ret = {(float)rx, (float)ry, (float)rz};
	return ret;
}

cvtx_Vec3f cvtx_ParticleArr_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel)
{
	cvtx_Vec3f dvort;
	double rx = 0, ry = 0, rz = 0;
	int i;
	assert(num_particles >= 0);
	for (i = 0; i < num_particles; ++i) {
		dvort = cvtx_Particle_ind_dvort(array_start[i],
			induced_particle, kernel);
		rx += dvort.x[0];
		ry += dvort.x[1];
		rz += dvort.x[2];
	}
	cvtx_Vec3f ret = {(float)rx, (float)ry, (float)rz};
	return ret;
}

cvtx_Vec3f cvtx_ParticleArr_visc_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel,
	const float kinematic_visc)
{
	cvtx_Vec3f dvort;
	double rx = 0, ry = 0, rz = 0;
	int i;
	assert(num_particles >= 0);
	for (i = 0; i < num_particles; ++i) {
		dvort = cvtx_Particle_visc_ind_dvort(array_start[i],
			induced_particle, kernel, kinematic_visc);
		rx += dvort.x[0];
		ry += dvort.x[1];
		rz += dvort.x[2];
	}
	cvtx_Vec3f ret = {(float)rx, (float)ry, (float)rz};
	return ret;
}

void cvtx_ParticleArr_Arr_ind_vel(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Vec3f *mes_start,
	const int num_mes,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel)
{
	int i;
#pragma omp parallel for schedule(static)
	for(i = 0; i < num_mes; ++i){
		result_array[i] = cvtx_ParticleArr_ind_vel(
			array_start, num_particles, mes_start[i], kernel);
	}
	return;
}

void cvtx_ParticleArr_Arr_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle **induced_start,
	const int num_induced,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel)
{
	int i;
#pragma omp parallel for schedule(static)
	for(i = 0; i < num_induced; ++i){
		result_array[i] = cvtx_ParticleArr_ind_dvort(
			array_start, num_particles, induced_start[i], kernel);
	}
	return;
}

void cvtx_ParticleArr_Arr_visc_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle **induced_start,
	const int num_induced,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel,
	const float kinematic_visc)
{
	int i;
#pragma omp parallel for schedule(static)
	for(i = 0; i < num_induced; ++i){
		result_array[i] = cvtx_ParticleArr_visc_ind_dvort(
			array_start, num_particles, induced_start[i], 
			kernel, kinematic_visc);
	}
	return;
}
