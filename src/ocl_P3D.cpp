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

#include "opencl_acc.h"
#include "ocl_P3D.h"

int opencl_brute_force_P3D_M2M_vel(
	const cvtx_P3D **array_start,
	const int num_particles,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array,
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
		return opencl_brute_force_P3D_M2M_vel_impl(
			array_start, num_particles, mes_start,
			num_mes, result_array, kernel, regularisation_radius,
			prog, queue, cont);
	}
	else
	{
		return -1;
	}
}

int opencl_brute_force_P3D_M2M_dvort(
	const cvtx_P3D **array_start,
	const int num_particles,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
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
		return opencl_brute_force_P3D_M2M_dvort_impl(
			array_start, num_particles, induced_start,
			num_induced, result_array, kernel, regularisation_radius,
			prog, queue, cont);
	}
	else
	{
		return -1;
	}
}

int opencl_brute_force_P3D_M2M_visc_dvort(
	const cvtx_P3D **array_start,
	const int num_particles,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc)
{
	/* Right now we just use the first active device. */
	assert(opencl_is_init());
	cl_program prog;
	cl_context cont;
	cl_command_queue queue;

	if (opencl_num_active_devices() > 0 &&
		opencl_get_device_state(0, &prog, &cont, &queue) == 0) {
		return opencl_brute_force_P3D_M2M_visc_dvort_impl(
			array_start, num_particles, induced_start,
			num_induced, result_array, kernel, regularisation_radius,
			kinematic_visc, prog, queue, cont);
	}
	else
	{
		return -1;
	}
}

int opencl_brute_force_P3D_M2M_vort(
	const cvtx_P3D** array_start,
	const int num_particles,
	const bsv_V3f* mes_start,
	const int num_mes,
	bsv_V3f* result_array,
	const cvtx_VortFunc* kernel,
	float regularisation_radius)
{
	/* Right now we just use the first active device. */
	assert(opencl_is_init());
	cl_program prog;
	cl_context cont;
	cl_command_queue queue;

	if (opencl_num_active_devices() > 0 &&
		opencl_get_device_state(0, &prog, &cont, &queue) == 0) {
		return opencl_brute_force_P3D_M2M_vort_impl(
			array_start, num_particles, mes_start,
			num_mes, result_array, kernel, regularisation_radius,
			prog, queue, cont);
	}
	else
	{
		return -1;
	}
}

/* This is *almost* identical to the vort impl so any bugs likely occur in both. */
int opencl_brute_force_P3D_M2M_vel_impl(
	const cvtx_P3D **array_start,
	const int num_particles,
	const bsv_V3f *mes_start,
	const int num_mes,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	cl_program program,
	cl_command_queue queue,
	cl_context context)
{
	char kernel_name[128] = "cvtx_nb_P3D_vel_";
	int i, n_particle_groups, n_zeroed_particles, n_modelled_particles;
	float constant_multiplyer = 1.f / (4.f * acosf(-1));
	size_t global_work_size[2], workgroup_size[2];
	cl_float3 *mes_pos_buff_data, *part_pos_buff_data, *part_vort_buff_data, *res_buff_data;
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
		mes_pos_buff_data = (cl_float3*) malloc(num_mes * sizeof(cl_float3));
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

		cl_float cl_recip_regularisation_radius = 1.f/regularisation_radius;
		status = clSetKernelArg(cl_kernel, 2, sizeof(cl_float), &cl_recip_regularisation_radius);
		assert(status == CL_SUCCESS);

		/* Generate a results buffer */
		res_buff_data = (cl_float3*) malloc(num_mes * sizeof(cl_float3));
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
			assert(0);
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
		part_pos_buff_data = (cl_float3*) malloc(n_modelled_particles * sizeof(cl_float3));
		part_vort_buff_data = (cl_float3*) malloc(n_modelled_particles * sizeof(cl_float3));
		for (i = 0; i < num_particles; ++i) {
			part_pos_buff_data[i].x = array_start[i]->coord.x[0];
			part_pos_buff_data[i].y = array_start[i]->coord.x[1];
			part_pos_buff_data[i].z = array_start[i]->coord.x[2];
			part_vort_buff_data[i].x = array_start[i]->vorticity.x[0];
			part_vort_buff_data[i].y = array_start[i]->vorticity.x[1];
			part_vort_buff_data[i].z = array_start[i]->vorticity.x[2];
		}
		/* We need this so that we always have the minimum workgroup size. */
		for (i = num_particles; i < n_modelled_particles; ++i) {
			part_pos_buff_data[i].x = 0;
			part_pos_buff_data[i].y = 0;
			part_pos_buff_data[i].z = 0;
			part_vort_buff_data[i].x = 0;
			part_vort_buff_data[i].y = 0;
			part_vort_buff_data[i].z = 0;
		}
		part_pos_buff = (cl_mem*) malloc(n_particle_groups * sizeof(cl_mem));
		part_vort_buff = (cl_mem*) malloc(n_particle_groups * sizeof(cl_mem));
		event_chain = (cl_event*) malloc(sizeof(cl_event) * n_particle_groups * 3);
		for (i = 0; i < n_particle_groups; ++i) {
			part_pos_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, part_pos_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				part_pos_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 3 * i);
			assert(status == CL_SUCCESS);
			part_vort_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, part_vort_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
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
			sizeof(cl_float3) * num_mes, res_buff_data, 1,
			event_chain + 3 * n_particle_groups - 1, NULL);
		for (i = 0; i < n_particle_groups * 3; ++i) { clReleaseEvent(event_chain[i]); }
		free(event_chain);	/* Its tempting to do this earlier, but remember, this is asynchonous! */
		for (i = 0; i < num_mes; ++i) {
			/* Constant multiplyer is constant the 1/4pi term. */
			result_array[i].x[0] = res_buff_data[i].x * constant_multiplyer;
			result_array[i].x[1] = res_buff_data[i].y * constant_multiplyer;
			result_array[i].x[2] = res_buff_data[i].z * constant_multiplyer;
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

int opencl_brute_force_P3D_M2M_dvort_impl(
	const cvtx_P3D **array_start,
	const int num_particles,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	cl_program program,
	cl_command_queue queue,
	cl_context context)
{
	char kernel_name[128] = "cvtx_nb_P3D_dvort_";
	int i, n_particle_groups, n_zeroed_particles, n_modelled_particles;
	float constant_multiplyer = 1.f / (4.f * acosf(-1) * powf(regularisation_radius, 3));
	size_t global_work_size[2], workgroup_size[2];
	cl_float3 *part1_pos_buff_data, *part1_vort_buff_data, *part2_pos_buff_data, *part2_vort_buff_data, *res_buff_data;
	cl_mem res_buff, *part1_pos_buff, *part1_vort_buff, part2_pos_buff, part2_vort_buff;
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
		workgroup_size[1] = 1;	/* Only 1 induced particle pos per workgroup. */
		global_work_size[0] = CVTX_WORKGROUP_SIZE;	/* We use multiple inducing particle buffers */
		global_work_size[1] = num_induced;

		/* Generate buffers for induced particle data  */
		part2_pos_buff_data = (cl_float3*) malloc(num_induced * sizeof(cl_float3));
		part2_vort_buff_data = (cl_float3*) malloc(num_induced * sizeof(cl_float3));
		for (i = 0; i < num_induced; ++i) {
			part2_pos_buff_data[i].x = induced_start[i]->coord.x[0];
			part2_pos_buff_data[i].y = induced_start[i]->coord.x[1];
			part2_pos_buff_data[i].z = induced_start[i]->coord.x[2];
			part2_vort_buff_data[i].x = induced_start[i]->vorticity.x[0];
			part2_vort_buff_data[i].y = induced_start[i]->vorticity.x[1];
			part2_vort_buff_data[i].z = induced_start[i]->vorticity.x[2];
		}
		/* Induced particle Create buffer, enqueue write and set kernel arg. */
		part2_pos_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, part2_pos_buff, CL_FALSE,
			0, num_induced * sizeof(cl_float3), part2_pos_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 3, sizeof(cl_mem), &part2_pos_buff);
		assert(status == CL_SUCCESS);
		part2_vort_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, part2_vort_buff, CL_FALSE,
			0, num_induced * sizeof(cl_float3), part2_vort_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 4, sizeof(cl_mem), &part2_vort_buff);
		assert(status == CL_SUCCESS);

		/* Generate a results buffer										*/
		res_buff_data = (cl_float3*) malloc(num_induced * sizeof(cl_float3));
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
			assert(0);
		}
		status = clSetKernelArg(cl_kernel, 5, sizeof(cl_mem), &res_buff);
		assert(status == CL_SUCCESS);

		cl_float cl_recip_regularisation_rad = 1.f/regularisation_radius;
		status = clSetKernelArg(cl_kernel, 2, sizeof(cl_float), &cl_recip_regularisation_rad);
		assert(status == CL_SUCCESS);

		/* Now create & dispatch particle buffers and kernel.
		Inducing particle count needs to be a multiple of the CVTX_WORKGROUP_SIZE,
		so we add some zerod particles onto the end of the array. */
		n_particle_groups = num_particles / CVTX_WORKGROUP_SIZE;
		if (num_particles % CVTX_WORKGROUP_SIZE) {
			n_zeroed_particles = CVTX_WORKGROUP_SIZE
				- num_particles % CVTX_WORKGROUP_SIZE;
			n_particle_groups += 1;
		}
		n_modelled_particles = CVTX_WORKGROUP_SIZE * n_particle_groups;
		part1_pos_buff_data = (cl_float3*) malloc(n_modelled_particles * sizeof(cl_float3));
		part1_vort_buff_data = (cl_float3*) malloc(n_modelled_particles * sizeof(cl_float3));
		for (i = 0; i < num_particles; ++i) {
			part1_pos_buff_data[i].x = array_start[i]->coord.x[0];
			part1_pos_buff_data[i].y = array_start[i]->coord.x[1];
			part1_pos_buff_data[i].z = array_start[i]->coord.x[2];
			part1_vort_buff_data[i].x = array_start[i]->vorticity.x[0];
			part1_vort_buff_data[i].y = array_start[i]->vorticity.x[1];
			part1_vort_buff_data[i].z = array_start[i]->vorticity.x[2];
		}
		/* We need this so that we always have the minimum workgroup size. */
		for (i = num_particles; i < n_modelled_particles; ++i) {
			part1_pos_buff_data[i].x = 0;
			part1_pos_buff_data[i].y = 0;
			part1_pos_buff_data[i].z = 0;
			part1_vort_buff_data[i].x = 0;
			part1_vort_buff_data[i].y = 0;
			part1_vort_buff_data[i].z = 0;
		}
		part1_pos_buff = (cl_mem*) malloc(n_particle_groups * sizeof(cl_mem));
		part1_vort_buff = (cl_mem*) malloc(n_particle_groups * sizeof(cl_mem));
		event_chain = (cl_event*) malloc(sizeof(cl_event) * n_particle_groups * 3);
		for (i = 0; i < n_particle_groups; ++i) {
			part1_pos_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, part1_pos_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				part1_pos_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 3 * i);
			assert(status == CL_SUCCESS);
			part1_vort_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, part1_vort_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				part1_vort_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 3 * i + 1);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 0, sizeof(cl_mem), part1_pos_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 1, sizeof(cl_mem), part1_vort_buff + i);
			assert(status == CL_SUCCESS);
			if (i == 0) {
				status = clEnqueueNDRangeKernel(queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 2, event_chain + 3 * i, event_chain + 3 * i + 2);
			}
			else {
				status = clEnqueueNDRangeKernel(queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 3, event_chain + 3 * i - 1, event_chain + 3 * i + 2);
			}
			assert(status == CL_SUCCESS);
			clReleaseMemObject(part1_pos_buff[i]);
			clReleaseMemObject(part1_vort_buff[i]);
		}

		/* Read back our results! */
		clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0,
			sizeof(cl_float3) * num_induced, res_buff_data, 1,
			event_chain + 3 * n_particle_groups - 1, NULL);
		for (i = 0; i < n_particle_groups * 3; ++i) { clReleaseEvent(event_chain[i]); }
		free(event_chain);	/* Its tempting to do this earlier, but remember, this is asynchonous! */
		for (i = 0; i < num_induced; ++i) {
			/* We take the 1 / (4 pi * reg_dist^3) into account here as const mult. */
			result_array[i].x[0] = res_buff_data[i].x * constant_multiplyer;
			result_array[i].x[1] = res_buff_data[i].y * constant_multiplyer;
			result_array[i].x[2] = res_buff_data[i].z * constant_multiplyer;
		}
		free(res_buff_data);

		free(part1_pos_buff);
		free(part1_vort_buff);
		free(part2_pos_buff_data);
		free(part2_vort_buff_data);
		free(part1_pos_buff_data);
		free(part1_vort_buff_data);
		clReleaseMemObject(res_buff);
		clReleaseMemObject(part2_pos_buff);
		clReleaseMemObject(part2_vort_buff);
		clReleaseKernel(cl_kernel);
		return 0;
	}
	else
	{
		return -1;
	}
}

int opencl_brute_force_P3D_M2M_visc_dvort_impl(
	const cvtx_P3D **array_start,
	const int num_particles,
	const cvtx_P3D **induced_start,
	const int num_induced,
	bsv_V3f *result_array,
	const cvtx_VortFunc *kernel,
	float regularisation_radius,
	float kinematic_visc,
	cl_program program,
	cl_command_queue queue,
	cl_context context)
{
	char kernel_name[128] = "cvtx_nb_P3D_visc_dvort_";
	int i, n_particle_groups, n_zeroed_particles, n_modelled_particles;
	size_t global_work_size[2], workgroup_size[2];
	cl_float3 *part1_pos_buff_data, *part1_vort_buff_data, *part2_pos_buff_data, *part2_vort_buff_data, *res_buff_data;
	cl_float *part1_vol_buff_data, *part2_vol_buff_data;
	cl_mem res_buff, *part1_pos_buff, *part1_vort_buff, *part1_vol_buff,
		part2_pos_buff, part2_vort_buff, part2_vol_buff;
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
		workgroup_size[1] = 1;	/* Only 1 induced particle pos per workgroup. */
		global_work_size[0] = CVTX_WORKGROUP_SIZE;	/* We use multiple inducing particle buffers */
		global_work_size[1] = num_induced;

		/* Generate buffers for induced particle data  */
		part2_pos_buff_data = (cl_float3*) malloc(num_induced * sizeof(cl_float3));
		part2_vort_buff_data = (cl_float3*) malloc(num_induced * sizeof(cl_float3));
		part2_vol_buff_data = (cl_float*) malloc(num_induced * sizeof(cl_float));
		for (i = 0; i < num_induced; ++i) {
			part2_pos_buff_data[i].x = induced_start[i]->coord.x[0];
			part2_pos_buff_data[i].y = induced_start[i]->coord.x[1];
			part2_pos_buff_data[i].z = induced_start[i]->coord.x[2];
			part2_vort_buff_data[i].x = induced_start[i]->vorticity.x[0];
			part2_vort_buff_data[i].y = induced_start[i]->vorticity.x[1];
			part2_vort_buff_data[i].z = induced_start[i]->vorticity.x[2];
			part2_vol_buff_data[i] = induced_start[i]->volume;
		}
		/* Induced particle Create buffer, enqueue write and set kernel arg. */
		part2_pos_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, part2_pos_buff, CL_TRUE,
			0, num_induced * sizeof(cl_float3), part2_pos_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 3, sizeof(cl_mem), &part2_pos_buff);
		assert(status == CL_SUCCESS);
		part2_vort_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, part2_vort_buff, CL_TRUE,
			0, num_induced * sizeof(cl_float3), part2_vort_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 4, sizeof(cl_mem), &part2_vort_buff);
		assert(status == CL_SUCCESS);
		part2_vol_buff = clCreateBuffer(context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float), NULL, &status);
		status = clEnqueueWriteBuffer(
			queue, part2_vol_buff, CL_TRUE,
			0, num_induced * sizeof(cl_float), part2_vol_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 5, sizeof(cl_mem), &part2_vol_buff);
		assert(status == CL_SUCCESS);

		/* Generate a results buffer										*/
		res_buff_data = (cl_float3*)malloc(num_induced * sizeof(cl_float3));
		res_buff = clCreateBuffer(context, CL_MEM_READ_WRITE,
			sizeof(cl_float3) * num_induced, NULL, &status);
		for (i = 0; i < num_induced; ++i) {
			res_buff_data[i].x = 0;
			res_buff_data[i].y = 0;
			res_buff_data[i].z = 0;
		}
		status = clEnqueueWriteBuffer(
			queue, res_buff, CL_TRUE,
			0, num_induced * sizeof(cl_float3), res_buff_data, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			assert(0);
		}
		status = clSetKernelArg(cl_kernel, 6, sizeof(cl_mem), &res_buff);
		assert(status == CL_SUCCESS);

		/* Setup the kinematic viscocity argument */
		cl_float cl_regularisation_rad = regularisation_radius;
		status = clSetKernelArg(cl_kernel, 7, sizeof(cl_float), &cl_regularisation_rad);
		assert(status == CL_SUCCESS);
		cl_float cl_kinem_visc = kinematic_visc;
		status = clSetKernelArg(cl_kernel, 8, sizeof(cl_float), &cl_kinem_visc);
		assert(status == CL_SUCCESS);

		/* Now create & dispatch particle buffers and kernel.
		Inducing particle count needs to be a multiple of the CVTX_WORKGROUP_SIZE,
		so we add some zerod particles onto the end of the array. */
		n_particle_groups = num_particles / CVTX_WORKGROUP_SIZE;
		if (num_particles % CVTX_WORKGROUP_SIZE) {
			n_zeroed_particles = CVTX_WORKGROUP_SIZE
				- num_particles % CVTX_WORKGROUP_SIZE;
			n_particle_groups += 1;
		}
		n_modelled_particles = CVTX_WORKGROUP_SIZE * n_particle_groups;
		part1_pos_buff_data = (cl_float3*) malloc(n_modelled_particles * sizeof(cl_float3));
		part1_vort_buff_data = (cl_float3*) malloc(n_modelled_particles * sizeof(cl_float3));
		part1_vol_buff_data = (cl_float*) malloc(n_modelled_particles * sizeof(cl_float));
		for (i = 0; i < num_particles; ++i) {
			part1_pos_buff_data[i].x = array_start[i]->coord.x[0];
			part1_pos_buff_data[i].y = array_start[i]->coord.x[1];
			part1_pos_buff_data[i].z = array_start[i]->coord.x[2];
			part1_vort_buff_data[i].x = array_start[i]->vorticity.x[0];
			part1_vort_buff_data[i].y = array_start[i]->vorticity.x[1];
			part1_vort_buff_data[i].z = array_start[i]->vorticity.x[2];
			part1_vol_buff_data[i] = array_start[i]->volume;
		}
		/* We need this so that we always have the minimum workgroup size. */
		for (i = num_particles; i < n_modelled_particles; ++i) {
			part1_pos_buff_data[i].x = 0;
			part1_pos_buff_data[i].y = 0;
			part1_pos_buff_data[i].z = 0;
			part1_vort_buff_data[i].x = 0;
			part1_vort_buff_data[i].y = 0;
			part1_vort_buff_data[i].z = 0;
			part1_vol_buff_data[i] = 0;
		}
		part1_pos_buff = (cl_mem*) malloc(n_particle_groups * sizeof(cl_mem));
		part1_vort_buff = (cl_mem*) malloc(n_particle_groups * sizeof(cl_mem));
		part1_vol_buff = (cl_mem*) malloc(n_particle_groups * sizeof(cl_mem));
		event_chain = (cl_event*) malloc(sizeof(cl_event) * n_particle_groups * 4);
		for (i = 0; i < n_particle_groups; ++i) {
			part1_pos_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, part1_pos_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				part1_pos_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i);
			assert(status == CL_SUCCESS);
			part1_vort_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, part1_vort_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				part1_vort_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i + 1);
			assert(status == CL_SUCCESS);
			part1_vol_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, part1_vol_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float),
				part1_vol_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i + 2);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 0, sizeof(cl_mem), part1_pos_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 1, sizeof(cl_mem), part1_vort_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 2, sizeof(cl_mem), part1_vol_buff + i);
			assert(status == CL_SUCCESS);
			if (i == 0) {
				status = clEnqueueNDRangeKernel(queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 3, event_chain + 4 * i, event_chain + 4 * i + 3);
			}
			else {
				status = clEnqueueNDRangeKernel(queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 4, event_chain + 4 * i - 1, event_chain + 4 * i + 3);
			}
			assert(status == CL_SUCCESS);
			clReleaseMemObject(part1_pos_buff[i]);
			clReleaseMemObject(part1_vort_buff[i]);
			clReleaseMemObject(part1_vol_buff[i]);
		}

		/* Read back our results! */
		clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0,
			sizeof(cl_float3) * num_induced, res_buff_data, 1,
			event_chain + 4 * n_particle_groups - 1, NULL);
		for (i = 0; i < n_particle_groups * 4; ++i) { clReleaseEvent(event_chain[i]); }
		free(event_chain);	/* Its tempting to do this earlier, but remember, this is asynchonous! */
		for (i = 0; i < num_induced; ++i) {
			result_array[i].x[0] = res_buff_data[i].x;
			result_array[i].x[1] = res_buff_data[i].y;
			result_array[i].x[2] = res_buff_data[i].z;
		}
		free(res_buff_data);

		free(part1_pos_buff);
		free(part1_vort_buff);
		free(part1_vol_buff);
		free(part2_pos_buff_data);
		free(part2_vort_buff_data);
		free(part2_vol_buff_data);
		free(part1_pos_buff_data);
		free(part1_vort_buff_data);
		free(part1_vol_buff_data);
		clReleaseMemObject(res_buff);
		clReleaseMemObject(part2_pos_buff);
		clReleaseMemObject(part2_vort_buff);
		clReleaseMemObject(part2_vol_buff);
		clReleaseKernel(cl_kernel);
		return 0;
	}
	else
	{
		return -1;
	}
}

/* This is *almost* identical to the vel impl so any bugs likely occur in both. */
int opencl_brute_force_P3D_M2M_vort_impl(
	const cvtx_P3D** array_start,
	const int num_particles,
	const bsv_V3f* mes_start,
	const int num_mes,
	bsv_V3f* result_array,
	const cvtx_VortFunc* kernel,
	float regularisation_radius,
	cl_program program,
	cl_command_queue queue,
	cl_context context)
{
	char kernel_name[128] = "cvtx_nb_P3D_vort_";
	int i, n_particle_groups, n_zeroed_particles, n_modelled_particles;
	float constant_multiplyer = 1.f / (4.f * acosf(-1) * powf(regularisation_radius, 3));
	size_t global_work_size[2], workgroup_size[2];
	cl_float3* mes_pos_buff_data, * part_pos_buff_data, * part_vort_buff_data, * res_buff_data;
	cl_mem mes_pos_buff, res_buff, * part_pos_buff, * part_vort_buff;
	cl_int status;
	cl_kernel cl_kernel;
	cl_event* event_chain;

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
		mes_pos_buff_data = (cl_float3*) malloc(num_mes * sizeof(cl_float3));
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

		cl_float cl_recip_regularisation_radius = 1.f / regularisation_radius;
		status = clSetKernelArg(cl_kernel, 2, sizeof(cl_float), &cl_recip_regularisation_radius);
		assert(status == CL_SUCCESS);

		/* Generate a results buffer */
		res_buff_data = (cl_float3*) malloc(num_mes * sizeof(cl_float3));
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
			assert(0);
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
		part_pos_buff_data = (cl_float3*) malloc(n_modelled_particles * sizeof(cl_float3));
		part_vort_buff_data = (cl_float3*) malloc(n_modelled_particles * sizeof(cl_float3));
		for (i = 0; i < num_particles; ++i) {
			part_pos_buff_data[i].x = array_start[i]->coord.x[0];
			part_pos_buff_data[i].y = array_start[i]->coord.x[1];
			part_pos_buff_data[i].z = array_start[i]->coord.x[2];
			part_vort_buff_data[i].x = array_start[i]->vorticity.x[0];
			part_vort_buff_data[i].y = array_start[i]->vorticity.x[1];
			part_vort_buff_data[i].z = array_start[i]->vorticity.x[2];
		}
		/* We need this so that we always have the minimum workgroup size. */
		for (i = num_particles; i < n_modelled_particles; ++i) {
			part_pos_buff_data[i].x = 0;
			part_pos_buff_data[i].y = 0;
			part_pos_buff_data[i].z = 0;
			part_vort_buff_data[i].x = 0;
			part_vort_buff_data[i].y = 0;
			part_vort_buff_data[i].z = 0;
		}
		part_pos_buff = (cl_mem*) malloc(n_particle_groups * sizeof(cl_mem));
		part_vort_buff = (cl_mem*) malloc(n_particle_groups * sizeof(cl_mem));
		event_chain = (cl_event*) malloc(sizeof(cl_event) * n_particle_groups * 3);
		for (i = 0; i < n_particle_groups; ++i) {
			part_pos_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, part_pos_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				part_pos_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 3 * i);
			assert(status == CL_SUCCESS);
			part_vort_buff[i] = clCreateBuffer(context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				queue, part_vort_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
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
			sizeof(cl_float3) * num_mes, res_buff_data, 1,
			event_chain + 3 * n_particle_groups - 1, NULL);
		for (i = 0; i < n_particle_groups * 3; ++i) { clReleaseEvent(event_chain[i]); }
		free(event_chain);	/* Its tempting to do this earlier, but remember, this is asynchonous! */
		for (i = 0; i < num_mes; ++i) {
			/* Constant multiplyer is constant the 1/4pi term. */
			result_array[i].x[0] = res_buff_data[i].x * constant_multiplyer;
			result_array[i].x[1] = res_buff_data[i].y * constant_multiplyer;
			result_array[i].x[2] = res_buff_data[i].z * constant_multiplyer;
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
