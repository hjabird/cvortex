#include "libcvtx.h"
/*============================================================================
ocl_particle.h

Handles the opencl accelerated vortex particle methods.

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
#ifdef CVTX_USING_OPENCL
#include <bsv/bsv.h>
#include "opencl_acc.h"

int opencl_brute_force_ParticleArr_Arr_ind_vel(
	const cvtx_Particle **array_start,
	const long num_particles,
	const bsv_V3f *mes_start,
	const long num_mes,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius);

int opencl_brute_force_ParticleArr_Arr_ind_dvort(
	const cvtx_Particle **array_start,
	const long num_particles,
	const cvtx_Particle **induced_start,
	const long num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius);

int opencl_brute_force_ParticleArr_Arr_visc_ind_dvort(
	const cvtx_Particle **array_start,
	const long num_particles,
	const cvtx_Particle **induced_start,
	const long num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc);

int opencl_brute_force_ParticleArr_Arr_ind_vel_impl(
	const cvtx_Particle **array_start,
	const long num_particles,
	const bsv_V3f *mes_start,
	const long num_mes,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	cl_program program,
	cl_command_queue queue,
	cl_context context);

int opencl_brute_force_ParticleArr_Arr_ind_dvort_impl(
	const cvtx_Particle **array_start,
	const long num_particles,
	const cvtx_Particle **induced_start,
	const long num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	cl_program program,
	cl_command_queue queue,
	cl_context context);

int opencl_brute_force_ParticleArr_Arr_visc_ind_dvort_impl(
	const cvtx_Particle **array_start,
	const long num_particles,
	const cvtx_Particle **induced_start,
	const long num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc,
	cl_program program,
	cl_command_queue queue,
	cl_context context);

#endif /* CVTX_USING_OPENCL */
