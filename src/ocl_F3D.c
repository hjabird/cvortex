#include "libcvtx.h"
/*============================================================================
ocl_filament.h

Handles the opencl accelerated vortex filament methods.

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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "opencl_acc.h"
#include "ocl_F3D.h"

int opencl_brute_force_F3D_M2M_vel(
	const cvtx_F3D **array_start,
	const int num_filaments,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array) {

	/* Right now we just use the first active device. */
	assert(opencl_is_init());
	cl_program prog;
	cl_context cont;
	cl_command_queue queue;

	if (opencl_num_active_devices() > 0 &&
		opencl_get_device_state(0, &prog, &cont, &queue) == 0) {
		if (num_mes < CVTX_WORKGROUP_SIZE) {
			return opencl_brute_force_F3D_M2sM_vel_impl(
				array_start, num_filaments, mes_start,
				num_mes, result_array, prog, queue, cont);
		}
		else {
			return opencl_brute_force_F3D_M2M_vel_impl(
				array_start, num_filaments, mes_start,
				num_mes, result_array, prog, queue, cont);
		}
	}
	else
	{
		return -1;
	}
}

int opencl_brute_force_F3D_M2M_vel_impl(
	const cvtx_F3D **array_start,
	const int num_filaments,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array,
	cl_program program,
	cl_command_queue queue,
	cl_context context)
{
	char kernel_name[128] = "cvtx_nb_Filament_ind_vel_singular";
	int i, num_filament_groups, n_zeroed_particles, n_modelled_filaments;
	size_t global_work_size[2], workgroup_size[2];
	cl_float3 *mes_pos_buff_data, *fil_start_buff_data, *fil_end_buff_data, *res_buff_data;
	cl_float *fil_strength_buff_data;
	cl_mem mes_pos_buff, res_buff, *fil_start_buff, *fil_end_buff, *fil_strength_buff;
	cl_int status;
	cl_kernel cl_kernel;
	cl_event *event_chain;

	if (opencl_init() == 1)
	{
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
		mes_pos_buff_data = malloc(num_mes * sizeof(cl_float3));
		for (i = 0; i < num_mes; ++i) {
			mes_pos_buff_data[i].x = mes_start[i].x[0];
			mes_pos_buff_data[i].y = mes_start[i].x[1];
			mes_pos_buff_data[i].z = mes_start[i].x[2];
		}
		mes_pos_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, num_mes * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, mes_pos_buff, CL_FALSE,
			0, num_mes * sizeof(cl_float3), mes_pos_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 3, sizeof(cl_mem), &mes_pos_buff);
		if (status != CL_SUCCESS) {
			free(mes_pos_buff_data);
			clReleaseMemObject(mes_pos_buff);
			clReleaseKernel(cl_kernel);
			return -1;
		}

		/* Generate a results buffer */
		res_buff_data = malloc(num_mes * sizeof(cl_float3));
		res_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float3) * num_mes, NULL, &status);
		for (i = 0; i < num_mes; ++i) {
			res_buff_data[i].x = 0;
			res_buff_data[i].y = 0;
			res_buff_data[i].z = 0;
		}
		status = clEnqueueWriteBuffer(
			queue, res_buff, CL_FALSE,
			0, num_mes * sizeof(cl_float3), res_buff_data, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			assert(false);
			printf("OPENCL:\tFailed to enqueue write buffer.");
		}
		status = clSetKernelArg(cl_kernel, 4, sizeof(cl_mem), &res_buff);
		assert(status == CL_SUCCESS);

		/* Now create & dispatch particle buffers and kernel. */
		num_filament_groups = num_filaments / CVTX_WORKGROUP_SIZE;
		if (num_filaments % CVTX_WORKGROUP_SIZE) {
			n_zeroed_particles = CVTX_WORKGROUP_SIZE
				- num_filaments % CVTX_WORKGROUP_SIZE;
			num_filament_groups += 1;
		}
		n_modelled_filaments = CVTX_WORKGROUP_SIZE * num_filament_groups;
		fil_start_buff_data = malloc(n_modelled_filaments * sizeof(cl_float3));
		fil_end_buff_data = malloc(n_modelled_filaments * sizeof(cl_float3));
		fil_strength_buff_data = malloc(n_modelled_filaments * sizeof(cl_float));
		for (i = 0; i < num_filaments; ++i) {
			fil_start_buff_data[i].x = array_start[i]->start.x[0];
			fil_start_buff_data[i].y = array_start[i]->start.x[1];
			fil_start_buff_data[i].z = array_start[i]->start.x[2];
			fil_end_buff_data[i].x = array_start[i]->end.x[0];
			fil_end_buff_data[i].y = array_start[i]->end.x[1];
			fil_end_buff_data[i].z = array_start[i]->end.x[2];
			fil_strength_buff_data[i] = array_start[i]->strength;
		}
		/* We need this so that we always have the minimum workgroup size. */
		for (i = num_filaments; i < n_modelled_filaments; ++i) {
			fil_start_buff_data[i].x = (float)0.0;
			fil_start_buff_data[i].y = (float)0.0;
			fil_start_buff_data[i].z = (float)0.0;
			fil_end_buff_data[i].x = (float)0.0;
			fil_end_buff_data[i].y = (float)0.0;
			fil_end_buff_data[i].z = (float)0.0;
			fil_strength_buff_data[i] = (float)0.0;
		}
		fil_start_buff = malloc(num_filament_groups * sizeof(cl_mem));
		fil_end_buff = malloc(num_filament_groups * sizeof(cl_mem));
		fil_strength_buff = malloc(num_filament_groups * sizeof(cl_mem));
		event_chain = malloc(sizeof(cl_event) * num_filament_groups * 4);
		for (i = 0; i < num_filament_groups; ++i) {
			fil_start_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, fil_start_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				fil_start_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i);
			assert(status == CL_SUCCESS);
			fil_end_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, fil_end_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				fil_end_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i + 1);
			assert(status == CL_SUCCESS);
			fil_strength_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, fil_strength_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float),
				fil_strength_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i + 2);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 0, sizeof(cl_mem), fil_start_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 1, sizeof(cl_mem), fil_end_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 2, sizeof(cl_mem), fil_strength_buff + i);
			assert(status == CL_SUCCESS);
			if (i == 0) {
				status = clEnqueueNDRangeKernel(queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 3, event_chain, event_chain + 4 * i + 3);
			}
			else {
				status = clEnqueueNDRangeKernel(queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 4, event_chain + 4 * i - 1, event_chain + 4 * i + 3);
			}
			assert(status == CL_SUCCESS);
			clReleaseMemObject(fil_start_buff[i]);
			clReleaseMemObject(fil_end_buff[i]);
			clReleaseMemObject(fil_strength_buff[i]);
		}

		/* Read back our results! */
		clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0,
			sizeof(cl_float3) * num_mes, res_buff_data, 1,
			event_chain + 4 * num_filament_groups - 1, NULL);
		for (i = 0; i < num_filament_groups * 4; ++i) { clReleaseEvent(event_chain[i]); }
		free(event_chain);	/* Its tempting to do this earlier, but remember, this is asynchonous! */
		for (i = 0; i < num_mes; ++i) {
			result_array[i].x[0] = res_buff_data[i].x;
			result_array[i].x[1] = res_buff_data[i].y;
			result_array[i].x[2] = res_buff_data[i].z;
		}
		free(res_buff_data);

		free(fil_start_buff);
		free(fil_end_buff);
		free(fil_strength_buff);
		free(fil_start_buff_data);
		free(fil_end_buff_data);
		free(fil_strength_buff_data);
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

int opencl_brute_force_F3D_M2sM_vel_impl(
	const cvtx_F3D** array_start,
	const int num_filaments,
	const bsv_V3f* mes_start,
	const int num_mes,
	bsv_V3f* result_array,
	cl_program program,
	cl_command_queue queue,
	cl_context context)
{

	char kernel_name[128] = "cvtx_nb_Filament_ind_vel_singular_smes";
	int i, num_filament_groups, n_zeroed_particles, n_modelled_filaments;
	size_t global_work_size[2], workgroup_size[2];
	cl_float3 *mes_pos_buff_data, * fil_start_buff_data, * fil_end_buff_data, * res_buff_data;
	cl_float *fil_strength_buff_data;
	cl_mem mes_pos_buff, res_buff, fil_start_buff, fil_end_buff, fil_strength_buff;
	cl_int status;
	cl_kernel cl_kernel;

	if (opencl_init() == 1)
	{
		cl_kernel = clCreateKernel(program, kernel_name, &status);
		if (status != CL_SUCCESS) {
			clReleaseKernel(cl_kernel);
			return -1;
		}

		num_filament_groups = num_filaments / CVTX_WORKGROUP_SIZE +
			(num_filaments % CVTX_WORKGROUP_SIZE == 0 ? 0 : 1);

		/* This has to match the opencl kernels, so be careful with fiddling */
		workgroup_size[0] = CVTX_WORKGROUP_SIZE;	/* Particles per group */
		workgroup_size[1] = 1;	/* Only 1 measure pos per workgroup. */
		global_work_size[0] = CVTX_WORKGROUP_SIZE;	/* We use multiple particle buffers */
		global_work_size[1] = num_mes * num_filament_groups;

		/* Generate an buffer for the measurement position data  */
		mes_pos_buff_data = malloc(num_mes * sizeof(cl_float3));
		for (i = 0; i < num_mes; ++i) {
			mes_pos_buff_data[i].x = mes_start[i].x[0];
			mes_pos_buff_data[i].y = mes_start[i].x[1];
			mes_pos_buff_data[i].z = mes_start[i].x[2];
		}
		mes_pos_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, num_mes * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, mes_pos_buff, CL_FALSE,
			0, num_mes * sizeof(cl_float3), mes_pos_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 3, sizeof(cl_mem), &mes_pos_buff);
		if (status != CL_SUCCESS) {
			free(mes_pos_buff_data);
			clReleaseMemObject(mes_pos_buff);
			clReleaseKernel(cl_kernel);
			return -1;
		}

		/* Generate a results buffer */
		res_buff_data = malloc(num_mes * num_filament_groups * sizeof(cl_float3));
		res_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float3) * num_mes * num_filament_groups, NULL, &status);
		for (i = 0; i < num_mes * num_filament_groups; ++i) {
			res_buff_data[i].x = 0;
			res_buff_data[i].y = 0;
			res_buff_data[i].z = 0;
		}
		status = clEnqueueWriteBuffer(
			queue, res_buff, CL_FALSE, 0, 
			num_mes * num_filament_groups * sizeof(cl_float3), res_buff_data, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			assert(false);
			printf("OPENCL:\tFailed to enqueue write buffer.");
		}
		status = clSetKernelArg(cl_kernel, 4, sizeof(cl_mem), &res_buff);
		assert(status == CL_SUCCESS);

		/* Now create & dispatch particle buffers and kernel. */
		if (num_filaments % CVTX_WORKGROUP_SIZE) {
			n_zeroed_particles = CVTX_WORKGROUP_SIZE
				- num_filaments % CVTX_WORKGROUP_SIZE;
		}
		n_modelled_filaments = CVTX_WORKGROUP_SIZE * num_filament_groups;
		fil_start_buff_data = malloc(n_modelled_filaments * sizeof(cl_float3));
		fil_end_buff_data = malloc(n_modelled_filaments * sizeof(cl_float3));
		fil_strength_buff_data = malloc(n_modelled_filaments * sizeof(cl_float));
		for (i = 0; i < num_filaments; ++i) {
			fil_start_buff_data[i].x = array_start[i]->start.x[0];
			fil_start_buff_data[i].y = array_start[i]->start.x[1];
			fil_start_buff_data[i].z = array_start[i]->start.x[2];
			fil_end_buff_data[i].x = array_start[i]->end.x[0];
			fil_end_buff_data[i].y = array_start[i]->end.x[1];
			fil_end_buff_data[i].z = array_start[i]->end.x[2];
			fil_strength_buff_data[i] = array_start[i]->strength;
		}
		/* We need this so that we always have the minimum workgroup size. */
		for (i = num_filaments; i < n_modelled_filaments; ++i) {
			fil_start_buff_data[i].x = (float)0.0;
			fil_start_buff_data[i].y = (float)0.0;
			fil_start_buff_data[i].z = (float)0.0;
			fil_end_buff_data[i].x = (float)0.0;
			fil_end_buff_data[i].y = (float)0.0;
			fil_end_buff_data[i].z = (float)0.0;
			fil_strength_buff_data[i] = (float)0.0;
		}
		fil_start_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, n_modelled_filaments * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, fil_start_buff, CL_FALSE,
			0, n_modelled_filaments * sizeof(cl_float3), fil_start_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 0, sizeof(cl_mem), &fil_start_buff);

		fil_end_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, n_modelled_filaments * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, fil_end_buff, CL_FALSE,
			0, n_modelled_filaments * sizeof(cl_float3), fil_end_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 1, sizeof(cl_mem), &fil_end_buff);

		fil_strength_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, n_modelled_filaments * sizeof(cl_float), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, fil_strength_buff, CL_FALSE,
			0, n_modelled_filaments * sizeof(cl_float), fil_strength_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 2, sizeof(cl_mem), &fil_strength_buff);

		clFinish(queue);
		status = clEnqueueNDRangeKernel(queue, cl_kernel, 2,
			NULL, global_work_size, workgroup_size, 0, NULL, NULL);
		assert(status == CL_SUCCESS);

		/* Read back our results! */
		clFinish(queue);
		clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0,
			sizeof(cl_float3) * num_mes * num_filament_groups, res_buff_data, 0,
			NULL, NULL);
		for (i = 0; i < num_mes; ++i) {
			result_array[i].x[0] = res_buff_data[i].x;
			result_array[i].x[1] = res_buff_data[i].y;
			result_array[i].x[2] = res_buff_data[i].z;
		}
		for (i = num_mes; i < num_mes * num_filament_groups; ++i) {
			result_array[i % num_mes].x[0] += res_buff_data[i].x;
			result_array[i % num_mes].x[1] += res_buff_data[i].y;
			result_array[i % num_mes].x[2] += res_buff_data[i].z;
		}
		free(res_buff_data);

		free(fil_start_buff_data);
		free(fil_end_buff_data);
		free(fil_strength_buff_data);
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

int opencl_brute_force_F3D_M2M_dvort(
	const cvtx_F3D **array_start,
	const int num_fil,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array) {

	/* Right now we just use the first active device. */
	assert(opencl_is_init());
	cl_program prog;
	cl_context cont;
	cl_command_queue queue;

	if (opencl_num_active_devices() > 0 &&
		opencl_get_device_state(0, &prog, &cont, &queue) == 0) {
		return opencl_brute_force_F3D_M2M_dvort_impl(
			array_start, num_fil, induced_start,
			num_induced, result_array, prog, queue, cont);
	}
	else
	{
		return -1;
	}
}

int opencl_brute_force_F3D_M2M_dvort_impl(
	const cvtx_F3D **array_start,
	const int num_fil,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	cl_program program,
	cl_command_queue queue,
	cl_context context)
{
	char kernel_name[128] = "cvtx_nb_Filament_ind_dvort_singular";
	int i, num_filament_groups, n_zeroed_particles, n_modelled_filaments;
	size_t global_work_size[2], workgroup_size[2];
	cl_float3 *part_pos_buff_data, *part_vort_buff_data,
		*fil_start_buff_data, *fil_end_buff_data, *res_buff_data;
	cl_float *fil_strength_buff_data;
	cl_mem part_pos_buff, part_vort_buff, res_buff,
		*fil_start_buff, *fil_end_buff, *fil_strength_buff;
	cl_int status;
	cl_kernel cl_kernel;
	cl_event *event_chain;

	if (opencl_init() == 1)
	{
		cl_kernel = clCreateKernel(program, kernel_name, &status);
		if (status != CL_SUCCESS) {
			clReleaseKernel(cl_kernel);
			return -1;
		}
		/* This has to match the opencl kernels, so be careful with fiddling */
		workgroup_size[0] = CVTX_WORKGROUP_SIZE;	/* Particles per group */
		workgroup_size[1] = 1;	/* Only 1 measure pos per workgroup. */
		global_work_size[0] = CVTX_WORKGROUP_SIZE;	/* We use multiple particle buffers */
		global_work_size[1] = num_induced;

		/* Generate an buffer for the measurement position data  */
		part_pos_buff_data = malloc(num_induced * sizeof(cl_float3));
		part_vort_buff_data = malloc(num_induced * sizeof(cl_float3));
		for (i = 0; i < num_induced; ++i) {
			part_pos_buff_data[i].x = induced_start[i]->coord.x[0];
			part_pos_buff_data[i].y = induced_start[i]->coord.x[1];
			part_pos_buff_data[i].z = induced_start[i]->coord.x[2];
			part_vort_buff_data[i].x = induced_start[i]->vorticity.x[0];
			part_vort_buff_data[i].y = induced_start[i]->vorticity.x[1];
			part_vort_buff_data[i].z = induced_start[i]->vorticity.x[2];
		}
		part_pos_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, part_pos_buff, CL_FALSE,
			0, num_induced * sizeof(cl_float3), part_pos_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		part_vort_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, part_vort_buff, CL_FALSE,
			0, num_induced * sizeof(cl_float3), part_vort_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 3, sizeof(cl_mem), &part_vort_buff);
		if (status != CL_SUCCESS) {
			free(part_pos_buff_data);
			free(part_vort_buff_data);
			clReleaseMemObject(part_pos_buff);
			clReleaseMemObject(part_vort_buff);
			clReleaseKernel(cl_kernel);
			return -1;
		}

		/* Generate a results buffer */
		res_buff_data = malloc(num_induced * sizeof(cl_float3));
		res_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float3) * num_induced, NULL, &status);
		for (i = 0; i < num_induced; ++i) {
			res_buff_data[i].x = 0;
			res_buff_data[i].y = 0;
			res_buff_data[i].z = 0;
		}
		status = clEnqueueWriteBuffer(
			queue, res_buff, CL_FALSE,
			0, num_induced * sizeof(cl_float3), res_buff_data, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			assert(false);
			printf("OPENCL:\tFailed to enqueue write buffer.");
		}
		status = clSetKernelArg(cl_kernel, 4, sizeof(cl_mem), &res_buff);
		assert(status == CL_SUCCESS);

		/* Now create & dispatch particle buffers and kernel. */
		num_filament_groups = num_fil / CVTX_WORKGROUP_SIZE;
		if (num_fil % CVTX_WORKGROUP_SIZE) {
			n_zeroed_particles = CVTX_WORKGROUP_SIZE
				- num_fil % CVTX_WORKGROUP_SIZE;
			num_filament_groups += 1;
		}
		n_modelled_filaments = CVTX_WORKGROUP_SIZE * num_filament_groups;
		fil_start_buff_data = malloc(n_modelled_filaments * sizeof(cl_float3));
		fil_end_buff_data = malloc(n_modelled_filaments * sizeof(cl_float3));
		fil_strength_buff_data = malloc(n_modelled_filaments * sizeof(cl_float));
		for (i = 0; i < num_fil; ++i) {
			fil_start_buff_data[i].x = array_start[i]->start.x[0];
			fil_start_buff_data[i].y = array_start[i]->start.x[1];
			fil_start_buff_data[i].z = array_start[i]->start.x[2];
			fil_end_buff_data[i].x = array_start[i]->end.x[0];
			fil_end_buff_data[i].y = array_start[i]->end.x[1];
			fil_end_buff_data[i].z = array_start[i]->end.x[2];
			fil_strength_buff_data[i] = array_start[i]->strength;
		}
		/* We need this so that we always have the minimum workgroup size. */
		for (i = num_fil; i < n_modelled_filaments; ++i) {
			fil_start_buff_data[i].x = (float)0.0;
			fil_start_buff_data[i].y = (float)0.0;
			fil_start_buff_data[i].z = (float)0.0;
			fil_end_buff_data[i].x = (float)0.0;
			fil_end_buff_data[i].y = (float)0.0;
			fil_end_buff_data[i].z = (float)0.0;
			fil_strength_buff_data[i] = (float)0.0;
		}
		fil_start_buff = malloc(num_filament_groups * sizeof(cl_mem));
		fil_end_buff = malloc(num_filament_groups * sizeof(cl_mem));
		fil_strength_buff = malloc(num_filament_groups * sizeof(cl_mem));
		event_chain = malloc(sizeof(cl_event) * num_filament_groups * 4);
		for (i = 0; i < num_filament_groups; ++i) {
			fil_start_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, fil_start_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				fil_start_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i);
			assert(status == CL_SUCCESS);
			fil_end_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, fil_end_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				fil_end_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i + 1);
			assert(status == CL_SUCCESS);
			fil_strength_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, fil_strength_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				fil_strength_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i + 2);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 0, sizeof(cl_mem), fil_start_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 1, sizeof(cl_mem), fil_end_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 1, sizeof(cl_mem), fil_strength_buff + i);
			assert(status == CL_SUCCESS);
			if (i == 0) {
				status = clEnqueueNDRangeKernel(queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 3, event_chain, event_chain + 4 * i + 3);
			}
			else {
				status = clEnqueueNDRangeKernel(queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 4, event_chain + 4 * i - 1, event_chain + 4 * i + 3);
			}
			assert(status == CL_SUCCESS);
			clReleaseMemObject(fil_start_buff[i]);
			clReleaseMemObject(fil_end_buff[i]);
			clReleaseMemObject(fil_strength_buff[i]);
		}

		/* Read back our results! */
		clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0,
			sizeof(cl_float3) * num_induced, res_buff_data, 1,
			event_chain + 4 * num_filament_groups - 1, NULL);
		for (i = 0; i < num_filament_groups * 4; ++i) { clReleaseEvent(event_chain[i]); }
		free(event_chain);	/* Its tempting to do this earlier, but remember, this is asynchonous! */
		for (i = 0; i < num_induced; ++i) {
			result_array[i].x[0] = res_buff_data[i].x;
			result_array[i].x[1] = res_buff_data[i].y;
			result_array[i].x[2] = res_buff_data[i].z;
		}
		free(res_buff_data);

		free(fil_start_buff);
		free(fil_end_buff);
		free(fil_strength_buff);
		free(fil_start_buff_data);
		free(fil_end_buff_data);
		free(fil_strength_buff_data);
		free(part_pos_buff_data);
		free(part_vort_buff_data);
		clReleaseMemObject(res_buff);
		clReleaseMemObject(part_pos_buff);
		clReleaseMemObject(part_vort_buff);
		clReleaseKernel(cl_kernel);
		return 0;
	}
	else
	{
		return -1;
	}
}

#endif /* CVTX_USING_OPENCL */
