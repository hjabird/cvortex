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

#include <math.h>
#include <assert.h>

cvtx_Vec3f cvtx_Particle_ind_vel(
	const cvtx_Particle * self,
	const cvtx_Vec3f mes_point,
	const cvtx_VortFunc * kernel)
{
	cvtx_Vec3f rad, num, ret;
	float cor, den, rho;
	rad = cvtx_Vec3f_minus(mes_point, self->coord);
	num = cvtx_Vec3f_cross(rad, self->vorticity);
	den = powf(cvtx_Vec3f_abs(rad), 3);
	rho = fabsf(cvtx_Vec3f_abs(rad) / self->radius);
	cor = kernel->reduction_factor_fn(rho) * -(float)1. / (4 * (float)acos(-1));
	ret = cvtx_Vec3f_mult(num, cor / den);
	return ret;
}

cvtx_Vec3f cvtx_Particle_ind_dvort(
	const cvtx_Particle * self,
	const cvtx_Particle * induced_particle,
	const cvtx_VortFunc * kernel)
{
	cvtx_Vec3f ret, rad, cross_om, t2, t21, t21n, t22, t224;
	float g, f, radd, rho, t1, t21d, t221, t222, t223;
	rad = cvtx_Vec3f_minus(induced_particle->coord, self->coord);
	radd = cvtx_Vec3f_abs(rad);
	rho = fabsf(radd / self->radius);
	kernel->combined_fn(rho, &g, &f);
	t1 = (float)1. / ((float)4. * (float)acos(-1) * powf(self->radius, 3));
	cross_om = cvtx_Vec3f_cross(induced_particle->vorticity, self->vorticity);
	t21n = cvtx_Vec3f_mult(cross_om, g);
	t21d = rho * rho * rho;
	t21 = cvtx_Vec3f_div(t21n, t21d);
	t221 = 1 / (radd * radd);
	t222 = (3 * g) / (rho * rho * rho) - f;
	t223 = cvtx_Vec3f_dot(induced_particle->vorticity, rad);
	t224 = cvtx_Vec3f_cross(rad, self->vorticity);
	t22 = cvtx_Vec3f_mult(t224, t221 * t222 * t223);
	t2 = cvtx_Vec3f_plus(t21, t22);
	ret = cvtx_Vec3f_mult(t2, t1);
	return ret;
}

cvtx_Vec3f cvtx_Particle_array_ind_vel(
	const cvtx_Particle* array_start,
	const int num_particles,
	const cvtx_Vec3f mes_point,
	const cvtx_VortFunc *kernel )
{
	cvtx_Vec3f vel, ret;
	int i;
	assert(num_particles >= 0);
	ret = cvtx_Vec3f_zero();
	for (i = 0; i < num_particles; ++i) {
		vel = cvtx_Particle_ind_vel(array_start + i,
			mes_point, kernel);
		ret = cvtx_Vec3f_plus(ret, vel);
	}
	return ret;
}

cvtx_Vec3f cvtx_Particle_array_ind_dvort(
	const cvtx_Particle * array_start, 
	const int num_particles, 
	const cvtx_Particle * induced_particle, 
	const cvtx_VortFunc * kernel)
{
	cvtx_Vec3f dvort, ret;
	int i;
	assert(num_particles >= 0);
	ret = cvtx_Vec3f_zero();
	for (i = 0; i < num_particles; ++i) {
		dvort = cvtx_Particle_ind_dvort(array_start + i,
			induced_particle, kernel);
		ret = cvtx_Vec3f_plus(ret, dvort);
	}
	return ret;
}
