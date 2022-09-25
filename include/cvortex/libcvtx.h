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
/*!	\file libcvtx.h
 *	\brief Public header for CVortex library.
 *	\author HJA Bird
 *	
 *	See https://github.com/hjabird/cvortex to find the source and precompiled
 *	binaries and to report bugs.
 *	Full documentation is in libcvtx.dox and can be generated with
 *	using Doxygen.
 */

#ifndef CVTX_EXPORT
# ifdef _WIN32
#  define CVTX_EXPORT __declspec(dllimport)
# else
#  define CVTX_EXPORT
# endif
#endif 
#ifdef __cplusplus
extern "C"
{
#endif

#include <bsv/bsv.h>

/* A Vortex particle in 3D */
typedef struct {
	bsv_V3f coord;
	bsv_V3f vorticity;
	float volume;
} cvtx_P3D;

/* A Vortex singular vortex filament in 3D */
typedef struct {
	bsv_V3f start, end;		/* Beginning & end coordinates of line segment */
	float strength;			/* Vort per unit length */
} cvtx_F3D;

/* A Vortex particle/filament in 2D */
typedef struct {
	bsv_V2f coord;
	float vorticity;
	float area;
} cvtx_P2D;

/* Vortex particle regularisation functions
*/
enum cvtx_VortFunc {
	cvtx_VortFunc_singular,
	cvtx_VortFunc_gaussian,
	cvtx_VortFunc_planetary,
	cvtx_VortFunc_winckelmans,
};


typedef struct {
	float(*func)(float U);
	float radius;
} cvtx_RedistFunc;

/* cvtx libary accelerator controls */
CVTX_EXPORT void cvtx_initialise();
CVTX_EXPORT void cvtx_finalise();
CVTX_EXPORT const char* cvtx_information();
CVTX_EXPORT int cvtx_num_accelerators();
CVTX_EXPORT int cvtx_num_enabled_accelerators();
CVTX_EXPORT const char* cvtx_accelerator_name(int accelerator_id);
CVTX_EXPORT int cvtx_accelerator_enabled(int accelerator_id);
CVTX_EXPORT void cvtx_accelerator_enable(int accelerator_id);
CVTX_EXPORT void cvtx_accelerator_disable(int accelerator_id);

/* cvtx_RedistFunc functions */
CVTX_EXPORT const cvtx_RedistFunc cvtx_RedistFunc_lambda0(void);
CVTX_EXPORT const cvtx_RedistFunc cvtx_RedistFunc_lambda1(void);
CVTX_EXPORT const cvtx_RedistFunc cvtx_RedistFunc_lambda2(void);
CVTX_EXPORT const cvtx_RedistFunc cvtx_RedistFunc_lambda3(void);
CVTX_EXPORT const cvtx_RedistFunc cvtx_RedistFunc_m4p(void);

/* cvtx_P3D 3D vortex particle functions */
CVTX_EXPORT bsv_V3f cvtx_P3D_S2S_vel(
	const cvtx_P3D *self,
	const bsv_V3f mes_point,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT bsv_V3f cvtx_P3D_S2S_dvort(
	const cvtx_P3D *self,
	const cvtx_P3D *induced_particle,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT bsv_V3f cvtx_P3D_S2S_visc_dvort(
	const cvtx_P3D *self,
	const cvtx_P3D *induced_particle,
	const cvtx_VortFunc kernel,
	float regularisation_radius,
	float kinematic_visc);

CVTX_EXPORT bsv_V3f cvtx_P3D_S2S_vort(
	const cvtx_P3D* self,
	const bsv_V3f mes_point,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT void cvtx_P3D_S2M_vel(
	const cvtx_P3D* self,
	const bsv_V3f* mes_start,
	const int num_mes,
	bsv_V3f* result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT void cvtx_P3D_S2M_dvort(
	const cvtx_P3D* self,
	const cvtx_P3D* induced_start,
	const int num_induced,
	bsv_V3f* result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT void cvtx_P3D_S2M_visc_dvort(
	const cvtx_P3D* self,
	const cvtx_P3D* induced_start,
	const int num_induced,
	bsv_V3f* result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius,
	float kinematic_visc);

CVTX_EXPORT void cvtx_P3D_S2M_vort(
	const cvtx_P3D* self,
	const bsv_V3f* mes_start,
	const int num_mes,
	bsv_V3f* result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT bsv_V3f cvtx_P3D_M2S_vel(
	const cvtx_P3D *array_start,
	const int num_particles,
	const bsv_V3f mes_point,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT bsv_V3f cvtx_P3D_M2S_dvort(
	const cvtx_P3D *array_start,
	const int num_particles,
	const cvtx_P3D *induced_particle,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT bsv_V3f cvtx_P3D_M2S_visc_dvort(
	const cvtx_P3D *array_start,
	const int num_particles,
	const cvtx_P3D *induced_particle,
	const cvtx_VortFunc kernel,
	float regularisation_radius,
	float kinematic_visc);

CVTX_EXPORT bsv_V3f cvtx_P3D_M2S_vort(
	const cvtx_P3D* array_start,
	const int num_particles,
	const bsv_V3f mes_point,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT void cvtx_P3D_M2M_vel(
	const cvtx_P3D *array_start,
	const int num_particles,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT void cvtx_P3D_M2M_dvort(
	const cvtx_P3D *array_start,
	const int num_particles,
	const cvtx_P3D *induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT void cvtx_P3D_M2M_visc_dvort(
	const cvtx_P3D *array_start,
	const int num_particles,
	const cvtx_P3D *induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius,
	float kinematic_visc);

CVTX_EXPORT void cvtx_P3D_M2M_vort(
	const cvtx_P3D* array_start,
	const int num_particles,
	const bsv_V3f* mes_start,
	const int num_mes,
	bsv_V3f* result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT int cvtx_P3D_redistribute_on_grid(
	const cvtx_P3D *input_array_start,
	const int n_input_particles,
	cvtx_P3D *output_particles,		/* input is &(*cvtx_P3D) to write to */
	int max_output_particles,		/* Set to resultant num particles.   */
	const cvtx_RedistFunc *redistributor,
	float grid_density,
	float negligible_vort);	/* 0 implies nothing is neglidgle, 1 everything*/

CVTX_EXPORT void cvtx_P3D_pedrizzetti_relaxation(
	cvtx_P3D* input_array_start,
	const int n_input_particles,
	float fdt,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

/* cvtx_F3D straight vortex filament functions */
CVTX_EXPORT bsv_V3f cvtx_F3D_S2S_vel(
	const cvtx_F3D *self,
	const bsv_V3f mes_point);

CVTX_EXPORT bsv_V3f cvtx_F3D_S2S_dvort(
	const cvtx_F3D *self,
	const cvtx_P3D *induced_particle);

CVTX_EXPORT bsv_V3f cvtx_F3D_M2S_vel(
	const cvtx_F3D *array_start,
	const int num_filaments,
	const bsv_V3f mes_point);

CVTX_EXPORT bsv_V3f cvtx_F3D_M2S_dvort(
	const cvtx_F3D *array_start,
	const int num_filaments,
	const cvtx_P3D *induced_particle);

CVTX_EXPORT void cvtx_F3D_M2M_vel(
	const cvtx_F3D *array_start,
	const int num_filaments,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array);

CVTX_EXPORT void cvtx_F3D_M2M_dvort(
	const cvtx_F3D *array_start,
	const int num_filaments,
	const cvtx_P3D *induced_start,
	const int num_induced,
	bsv_V3f *result_array);

CVTX_EXPORT void cvtx_F3D_inf_mtrx(
	const cvtx_F3D *array_start,
	const int num_filaments,
	const bsv_V3f *mes_start,
	const bsv_V3f *dir_start,
	const int num_mes,
	float *result_matrix);

/* cvtx_P2D vortex particle 2D functions */
CVTX_EXPORT bsv_V2f cvtx_P2D_S2S_vel(
	const cvtx_P2D *self,
	const bsv_V2f mes_point,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT void cvtx_P2D_S2M_vel(
	const cvtx_P2D* self,
	const bsv_V2f* mes_start,
	const int num_mes,
	bsv_V2f* result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT bsv_V2f cvtx_P2D_M2S_vel(
	const cvtx_P2D *array_start,
	const int num_particles,
	const bsv_V2f mes_point,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT void cvtx_P2D_M2M_vel(
	const cvtx_P2D *array_start,
	const int num_particles,
	const bsv_V2f *mes_start,
	const int num_mes,
	bsv_V2f *result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius);

CVTX_EXPORT float cvtx_P2D_S2S_visc_dvort(
	const cvtx_P2D * self,
	const cvtx_P2D * induced_particle,
	const cvtx_VortFunc  kernel,
	float regularisation_radius,
	float kinematic_visc);

CVTX_EXPORT void cvtx_P2D_S2M_visc_dvort(
	const cvtx_P2D* self,
	const cvtx_P2D* induced_start,
	const int num_induced,
	float* result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius,
	float kinematic_visc);

CVTX_EXPORT float cvtx_P2D_M2S_visc_dvort(
	const cvtx_P2D *array_start,
	const int num_particles,
	const cvtx_P2D *induced_particle,
	const cvtx_VortFunc kernel,
	float regularisation_radius,
	float kinematic_visc);

CVTX_EXPORT void cvtx_P2D_M2M_visc_dvort(
	const cvtx_P2D *array_start,
	const int num_particles,
	const cvtx_P2D *induced_start,
	const int num_induced,
	float *result_array,
	const cvtx_VortFunc kernel,
	float regularisation_radius,
	float kinematic_visc);

CVTX_EXPORT int cvtx_P2D_redistribute_on_grid( /* Returns number of created particles. */
	const cvtx_P2D *input_array_start,
	const int num_particles,
	cvtx_P2D *output_particles,	/* Is preallocated array. */
	int num_output_particles,		/* Size of preallocated array.   */
	const cvtx_RedistFunc *redistributor,
	float grid_density,
	float negligible_vort);

#ifdef __cplusplus
} // extern "C"
#endif
#endif /* CVTX_LIBCVTX_H */
