#ifndef CVTX_PARTICLE_H
#define CVTX_PARTICLE_H
/*============================================================================
Particle.h

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

#include "Vec3f.h"
#include "VortFunc.h"

typedef struct {
	cvtx_Vec3f coord;
	cvtx_Vec3f vorticity;
	float radius;
} cvtx_Particle;

cvtx_Vec3f cvtx_Particle_ind_vel(
	const cvtx_Particle *self, 
	const cvtx_Vec3f mes_point, 
	const cvtx_VortFunc *kernel);

cvtx_Vec3f cvtx_Particle_ind_dvort(
	const cvtx_Particle *self, 
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel);

cvtx_Vec3f cvtx_Particle_visc_ind_dvort(
	const cvtx_Particle *self,
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel,
	const float kinematic_visc);

cvtx_Vec3f cvtx_ParticleArr_ind_vel(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Vec3f mes_point,
	const cvtx_VortFunc *kernel);

cvtx_Vec3f cvtx_ParticleArr_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel);

cvtx_Vec3f cvtx_ParticleArr_visc_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel,
	const float kinematic_visc);

void cvtx_ParticleArr_Arr_ind_vel(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Vec3f *mes_start,
	const int num_mes,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel);

void cvtx_ParticleArr_Arr_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle **induced_start,
	const int num_induced,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel);

void cvtx_ParticleArr_Arr_visc_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle **induced_start,
	const int num_induced,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel,
	const float kinematic_visc);

#endif /* CVTX_PARTICLE_H */
