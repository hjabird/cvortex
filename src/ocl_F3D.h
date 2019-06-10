#include "libcvtx.h"
/*============================================================================
ocl_filament.h

Handles the opencl accelerated vortex filament methods

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
#include <CL/cl.h>

int opencl_brute_force_StraightVortFilArr_Arr_ind_vel(
	const cvtx_F3D **array_start,
	const int num_filaments,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array);

int opencl_brute_force_StraightVortFilArr_Arr_ind_vel_impl(
	const cvtx_F3D **array_start,
	const int num_filaments,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array,
	cl_program program,
	cl_command_queue queue,
	cl_context context);

int opencl_brute_force_StraightVortFilArr_Arr_ind_dvort(
	const cvtx_F3D **array_start,
	const int num_fil,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array);

int opencl_brute_force_StraightVortFilArr_Arr_ind_dvort_impl(
	const cvtx_F3D **array_start,
	const int num_fil,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	cl_program program,
	cl_command_queue queue,
	cl_context context);

#endif /* CVTX_USING_OPENCL */
