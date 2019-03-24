#ifndef CVTX_LIBCVTX_H
#define CVTX_LIBCVTX_H
/*============================================================================
libcvtx.h

An all inclusive header file for the cvortex library.

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
#ifndef CVTX_EXPORT
# ifdef _WIN32
#  define CVTX_EXPORT __declspec(dllimport)
# else
#  define CVTX_EXPORT
# endif
#endif 

#include <bsv/bsv.h>

typedef struct {
	bsv_V3f coord;
	bsv_V3f vorticity;
	float volume;
} cvtx_Particle;

typedef struct {
	bsv_V3f start, end;		/* Beginning & end coordinates of line segment */
	float strength;			/* Vort per unit length */
} cvtx_StraightVortFil;

typedef struct {
	float(*g_fn)(float rho);
	float(*zeta_fn)(float rho);
	void(*combined_fn)(float rho, float* g, float* zeta);
	float(*eta_fn)(float rho);
	char cl_kernel_name_ext[32];
} cvtx_VortFunc;

/* cvtx_libary controls */
CVTX_EXPORT void cvtx_initialise();
CVTX_EXPORT void cvtx_finalise();
CVTX_EXPORT int cvtx_num_accelerators();
CVTX_EXPORT int cvtx_num_enabled_accelerators();
CVTX_EXPORT char* cvtx_accelerator_name(int accelerator_id);
CVTX_EXPORT int cvtx_accelerator_enabled(int accelerator_id);
CVTX_EXPORT void cvtx_accelerator_enable(int accelerator_id);
CVTX_EXPORT void cvtx_accelerator_disable(int accelerator_id);

/* cvtx_Particle functions */
CVTX_EXPORT bsv_V3f cvtx_Particle_ind_vel(
	const cvtx_Particle *self,
	const bsv_V3f mes_point,
	const cvtx_VortFunc *kernel,
	float regularisation_radius);

CVTX_EXPORT bsv_V3f cvtx_Particle_ind_dvort(
	const cvtx_Particle *self,
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel,
	float regularisation_radius);

CVTX_EXPORT bsv_V3f cvtx_Particle_visc_ind_dvort(
	const cvtx_Particle *self,
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc);

CVTX_EXPORT bsv_V3f cvtx_ParticleArr_ind_vel(
	const cvtx_Particle **array_start,
	const int num_particles,
	const bsv_V3f mes_point,
	const cvtx_VortFunc *kernel,
	float regularisation_radius);

CVTX_EXPORT bsv_V3f cvtx_ParticleArr_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel,
	float regularisation_radius);

CVTX_EXPORT bsv_V3f cvtx_ParticleArr_visc_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle *induced_particle,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc);

CVTX_EXPORT void cvtx_ParticleArr_Arr_ind_vel(
	const cvtx_Particle **array_start,
	const int num_particles,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius);

CVTX_EXPORT void cvtx_ParticleArr_Arr_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius);

CVTX_EXPORT void cvtx_ParticleArr_Arr_visc_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc);

/* cvtx_VortFunc functions */
CVTX_EXPORT const cvtx_VortFunc cvtx_VortFunc_singular(void);
CVTX_EXPORT const cvtx_VortFunc cvtx_VortFunc_winckelmans(void);
CVTX_EXPORT const cvtx_VortFunc cvtx_VortFunc_planetary(void);
CVTX_EXPORT const cvtx_VortFunc cvtx_VortFunc_gaussian(void);

/* cvtx_straight vortex filament functions */

CVTX_EXPORT bsv_V3f cvtx_StraightVortFil_ind_vel(
	const cvtx_StraightVortFil *self,
	const bsv_V3f mes_point);

CVTX_EXPORT bsv_V3f cvtx_StraightVortFil_ind_dvort(
	const cvtx_StraightVortFil *self,
	const cvtx_Particle *induced_particle);

CVTX_EXPORT bsv_V3f cvtx_StraightVortFilArr_ind_vel(
	const cvtx_StraightVortFil **array_start,
	const int num_particles,
	const bsv_V3f mes_point);

CVTX_EXPORT bsv_V3f cvtx_StraightVortFilArr_ind_dvort(
	const cvtx_StraightVortFil **array_start,
	const int num_particles,
	const cvtx_Particle *induced_particle);

CVTX_EXPORT void cvtx_StraightVortFilArr_Arr_ind_vel(
	const cvtx_StraightVortFil **array_start,
	const int num_filaments,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array);

CVTX_EXPORT void cvtx_StraightVortFilArr_Arr_ind_dvort(
	const cvtx_StraightVortFil **array_start,
	const int num_filaments,
	const cvtx_Particle **induced_start,
	const int num_induced,
	bsv_V3f *result_array);

#endif /* CVTX_LIBCVTX_H */
