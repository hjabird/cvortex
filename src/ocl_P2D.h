#include "libcvtx.h"
/*============================================================================
ocl_P2D.h

Handles the opencl accelerated 2D vortex particle methods.

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
#ifdef CVTX_USING_OPENCL
#include <bsv/bsv.h>
#include "opencl_acc.h"

int opencl_brute_force_P2D_M2M_vel(
	const cvtx_P2D **array_start,
	const int num_particles,
	const bsv_V2f *mes_start,
	const int num_mes,
	bsv_V2f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius);

int opencl_brute_force_P2D_M2M_visc_dvort(
	const cvtx_P2D **array_start,
	const int num_particles,
	const cvtx_P2D **induced_start,
	const int num_induced,
	float *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc);

int opencl_brute_force_P2D_M2M_vel_impl(
	const cvtx_P2D **array_start,
	const int num_particles,
	const bsv_V2f *mes_start,
	const int num_mes,
	bsv_V2f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	cl_program program,
	cl_command_queue queue,
	cl_context context);

/* For small number of measurement points. */
int opencl_brute_force_P2D_M2sM_vel_impl(
	const cvtx_P2D **array_start,
	const int num_particles,
	const bsv_V2f *mes_start,
	const int num_mes,
	bsv_V2f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	cl_program program,
	cl_command_queue queue,
	cl_context context);

int opencl_brute_force_P2D_M2M_visc_dvort_impl(
	const cvtx_P2D **array_start,
	const int num_particles,
	const cvtx_P2D **induced_start,
	const int num_induced,
	float *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc,
	cl_program program,
	cl_command_queue queue,
	cl_context context);

#endif /* CVTX_USING_OPENCL */
