/*============================================================================
opencl_acc.h

Acceleration of the n-body problem using OpenCL. Conditional compilation.

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
#ifndef CVTX_OPENCL_ACC_H
#define CVTX_OPENCL_ACC_H
#include "Particle.h"
#include "Vec3f.h"
#include "VortFunc.h"

int opencl_brute_force_ParticleArr_Arr_ind_vel(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Vec3f *mes_start,
	const int num_mes,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel);

int opencl_brute_force_ParticleArr_Arr_ind_dvort(
	const cvtx_Particle **array_start,
	const int num_particles,
	const cvtx_Particle **induced_start,
	const int num_induced,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel);

#endif CVTX_OPENCL_ACC_H
#endif CVTX_USING_OPENCL