#include "libcvtx.h"
/*============================================================================
ocl_particle.c

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
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>
#include "opencl_acc.h"
#include "ocl_P2D.h"

int opencl_brute_force_P2D_M2M_vel(
	const cvtx_P2D **array_start,
	const int num_particles,
	const bsv_V2f *mes_start,
	const int num_mes,
	bsv_V2f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius)
{
	/* Right now we just use the first active device. */
	assert(opencl_is_init());
	cl_program prog;
	cl_context cont;
	cl_command_queue queue;

	if (opencl_num_active_devices() > 0 &&
		opencl_get_device_state(0, &prog, &cont, &queue) == 0) {
		return opencl_brute_force_P2D_M2M_vel_impl(
			array_start, num_particles, mes_start,
			num_mes, result_array, kernel, regularisation_radius,
			prog, queue, cont);
	}
	else
	{
		return -1;
	}
}

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
	cl_context context)
{
	char kernel_name[128] = "cvtx_nb_P2D_vel_";
	int i, n_particle_groups, n_zeroed_particles, n_modelled_particles;
	float constant_multiplyer = 1.f / (2.f * acosf(-1));
	size_t global_work_size[2], workgroup_size[2];
	cl_float2 *mes_pos_buff_data, *part_pos_buff_data, *res_buff_data;
	cl_float *part_vort_buff_data;
	cl_mem mes_pos_buff, res_buff, *part_pos_buff, *part_vort_buff;
	cl_int status;
	cl_kernel cl_kernel;
	cl_event *event_chain;

	if (opencl_init() == 1)
	{
		strncat(kernel_name, kernel->cl_kernel_name_ext, 32);
		cl_kernel = clCreateKernel(program, kernel_name, &status);
		if (status != CL_SUCCESS) {
			clReleaseKernel(cl_kernel);
			return -1;
		}
		/* This has to match the opencl kernels, so be careful with fiddling */
		workgroup_size[0] = CVTX_WORKGROUP_SIZE;	/* Particles per group */
		workgroup_size[1] = 1;	/* Only 1 measure pos per workgroup. */
		global_work_size[0] = CVTX_WORKGROUP_SIZE;	/* We use multiple particle buffers */
		global_work_size[1] = num_mes;

		/* Generate an buffer for the measurement position data  */
		mes_pos_buff_data = malloc(num_mes * sizeof(cl_float2));
		for (i = 0; i < num_mes; ++i) {
			mes_pos_buff_data[i].x = mes_start[i].x[0];
			mes_pos_buff_data[i].y = mes_start[i].x[1];
		}
		mes_pos_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, num_mes * sizeof(cl_float2), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, mes_pos_buff, CL_FALSE,
			0, num_mes * sizeof(cl_float2), mes_pos_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 3, sizeof(cl_mem), &mes_pos_buff);
		if (status != CL_SUCCESS) {
			free(mes_pos_buff_data);
			clReleaseMemObject(mes_pos_buff);
			clReleaseKernel(cl_kernel);
			return -1;
		}

		cl_float cl_recip_regularisation_radius = 1.f / regularisation_radius;
		status = clSetKernelArg(cl_kernel, 2, sizeof(cl_float), &cl_recip_regularisation_radius);
		assert(status == CL_SUCCESS);

		/* Generate a results buffer */
		res_buff_data = malloc(num_mes * sizeof(cl_float2));
		res_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float2) * num_mes, NULL, &status);
		for (i = 0; i < num_mes; ++i) {
			res_buff_data[i].x = 0.f;
			res_buff_data[i].y = 0.f;
		}
		status = clEnqueueWriteBuffer(
			queue, res_buff, CL_FALSE,
			0, num_mes * sizeof(cl_float2), res_buff_data, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			assert(false);
		}
		status = clSetKernelArg(cl_kernel, 4, sizeof(cl_mem), &res_buff);
		assert(status == CL_SUCCESS);

		/* Now create & dispatch particle buffers and kernel. */
		n_particle_groups = num_particles / CVTX_WORKGROUP_SIZE;
		if (num_particles % CVTX_WORKGROUP_SIZE) {
			n_zeroed_particles = CVTX_WORKGROUP_SIZE
				- num_particles % CVTX_WORKGROUP_SIZE;
			n_particle_groups += 1;
		}
		n_modelled_particles = CVTX_WORKGROUP_SIZE * n_particle_groups;
		part_pos_buff_data = malloc(n_modelled_particles * sizeof(cl_float2));
		part_vort_buff_data = malloc(n_modelled_particles * sizeof(cl_float));
		for (i = 0; i < num_particles; ++i) {
			part_pos_buff_data[i].x = array_start[i]->coord.x[0];
			part_pos_buff_data[i].y = array_start[i]->coord.x[1];
			part_vort_buff_data[i] = array_start[i]->vorticity;
		}
		/* We need this so that we always have the minimum workgroup size. */
		for (i = num_particles; i < n_modelled_particles; ++i) {
			part_pos_buff_data[i].x = 0.f;
			part_pos_buff_data[i].y = 0.f;
			part_vort_buff_data[i] = 0.f;
		}
		part_pos_buff = malloc(n_particle_groups * sizeof(cl_mem));
		part_vort_buff = malloc(n_particle_groups * sizeof(cl_mem));
		event_chain = malloc(sizeof(cl_event) * n_particle_groups * 3);
		for (i = 0; i < n_particle_groups; ++i) {
			part_pos_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float2), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, part_pos_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float2),
				part_pos_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 3 * i);
			assert(status == CL_SUCCESS);
			part_vort_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, part_vort_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float),
				part_vort_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 3 * i + 1);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 0, sizeof(cl_mem), part_pos_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 1, sizeof(cl_mem), part_vort_buff + i);
			assert(status == CL_SUCCESS);
			if (i == 0) {
				status = clEnqueueNDRangeKernel(queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 2, event_chain, event_chain + 3 * i + 2);
			}
			else {
				status = clEnqueueNDRangeKernel(queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 3, event_chain + 3 * i - 1, event_chain + 3 * i + 2);
			}
			assert(status == CL_SUCCESS);
			clReleaseMemObject(part_pos_buff[i]);
			clReleaseMemObject(part_vort_buff[i]);
		}

		/* Read back our results! */
		clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0,
			sizeof(cl_float2) * num_mes, res_buff_data, 1,
			event_chain + 3 * n_particle_groups - 1, NULL);
		for (i = 0; i < n_particle_groups * 3; ++i) { clReleaseEvent(event_chain[i]); }
		free(event_chain);	/* Its tempting to do this earlier, but remember, this is asynchonous! */
		for (i = 0; i < num_mes; ++i) {
			/* Constant multiplyer is constant the 1/2pi term. */
			result_array[i].x[0] = res_buff_data[i].x * constant_multiplyer;
			result_array[i].x[1] = res_buff_data[i].y * constant_multiplyer;
		}
		free(res_buff_data);

		free(part_pos_buff);
		free(part_vort_buff);
		free(part_pos_buff_data);
		free(part_vort_buff_data);
		free(mes_pos_buff_data);
		clReleaseMemObject(res_buff);
		clReleaseMemObject(mes_pos_buff);
		clReleaseKernel(cl_kernel);
		return 0;
	}
	else
	{
		return -1;
	}
}

#endif /* CVTX_USING_OPENCL */
