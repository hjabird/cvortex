#include "../include/cvortex/opencl_acc.h"
/*============================================================================
opencl_acc.c

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
#define CVTX_WORKGROUP_SIZE 256

#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct {
	cl_platform_id *platforms;
	cl_device_id *devices;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
} nbody_ocl_state;

/* Initialisation function - only called once. */
static int opencl_print_platform_info() {
	int i, j;
	char* value;
	size_t valueSize;
	cl_uint platformCount;
	cl_platform_id* platforms;
	cl_uint deviceCount;
	cl_device_id* devices;
	cl_uint maxComputeUnits;

	// get all platforms
	clGetPlatformIDs(0, NULL, &platformCount);
	platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);

	for (i = 0; (unsigned int) i < platformCount; i++) {
		// print platform name
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &valueSize);
		value = (char*)malloc(valueSize);
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, valueSize, value, NULL);
		printf("%d. Platform: %s\n", i + 1, value);
		free(value);
		// get all devices
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
		devices = (cl_device_id*)malloc(sizeof(cl_device_id) * deviceCount);
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

		// for each device print critical attributes
		for (j = 0; (unsigned int) j < deviceCount; j++) {
			// print device name
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
			printf(" %d.%d. Device: %s\n", i + 1, j + 1, value);
			free(value);
			// print hardware device version
			clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value, NULL);
			printf("  %d.%d.%d Hardware version: %s\n", i + 1, j + 1, 1, value);
			free(value);
			// print software driver version
			clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value, NULL);
			printf("  %d.%d.%d Software version: %s\n", i + 1, j + 1, 2, value);
			free(value);
			// print c version supported by compiler for device
			clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &valueSize);
			value = (char*)malloc(valueSize);
			clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize, value, NULL);
			printf("  %d.%d.%d OpenCL C version: %s\n", i + 1, j + 1, 3, value);
			free(value);
			// print parallel compute units
			clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
				sizeof(maxComputeUnits), &maxComputeUnits, NULL);
			printf("  %d.%d.%d Parallel compute units: %d\n", i + 1, j + 1, 4, maxComputeUnits);
		}
		free(devices);
	}
	free(platforms);
	return 0;
}

int opencl_initialise() {
	static int tried_initialised = 0;
	static int opencl_working = -1;	/* meaning no, 0 meaning yes */
	int good = 1;
	cl_int status;
	cl_uint num_platforms, num_devices;
	char compile_options[1024] = "";
	char tmp[128];
	sprintf(tmp, "%i", CVTX_WORKGROUP_SIZE);
	strcat(compile_options, " -cl-fast-relaxed-math -D CVTX_CL_WORKGROUP_SIZE=");
	strcat(compile_options, tmp);
	if (!tried_initialised) {
		nbody_ocl_state.devices = NULL;
		nbody_ocl_state.platforms = NULL;

		status = clGetPlatformIDs(0, NULL, &num_platforms);
		nbody_ocl_state.platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
		status = clGetPlatformIDs(num_platforms, nbody_ocl_state.platforms, NULL);
		if (status != CL_SUCCESS) {
			printf("OPENCL:\tFailed to find opencl platforms.\n");
			good = 0;
			tried_initialised = 1;
			free(nbody_ocl_state.platforms);
			nbody_ocl_state.platforms = NULL;
		}
	}
	if (!tried_initialised && good) {
		status = clGetDeviceIDs(nbody_ocl_state.platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
		nbody_ocl_state.devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
		status = clGetDeviceIDs(nbody_ocl_state.platforms[0], CL_DEVICE_TYPE_GPU, num_devices, nbody_ocl_state.devices, NULL);
		if (status != CL_SUCCESS) {
			printf("OPENCL:\tFailed to find OpenCl GPU device.\n");
			good = 0;
			tried_initialised = 1;
			free(nbody_ocl_state.platforms);
			nbody_ocl_state.platforms = NULL;
			free(nbody_ocl_state.devices);
			nbody_ocl_state.devices = NULL;
		}
	}
	if (!tried_initialised && good) {
		nbody_ocl_state.context = clCreateContext(NULL, num_devices, nbody_ocl_state.devices, NULL, NULL, &status);
		assert(status == CL_SUCCESS);
		nbody_ocl_state.queue = clCreateCommandQueue(nbody_ocl_state.context, nbody_ocl_state.devices[0], 
			CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &status);
		if (status != CL_SUCCESS) {
			printf("OPENCL:\tCould not create context or command queue.\n");
		}
		const char* program_source =
#		include "nbody.cl"
			;	/* Including in source makes it easier to distribute a shared lib. */
		nbody_ocl_state.program = clCreateProgramWithSource(nbody_ocl_state.context, 1, (const char**)&program_source, NULL, &status);
		assert(status == CL_SUCCESS);
		status = clBuildProgram(nbody_ocl_state.program, num_devices, nbody_ocl_state.devices, compile_options, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf("OPENCL:\tFailed to build opencl program!\n");
			printf("OPENCL:\tBuild log:");
			char *buffer;
			size_t length;
			status = clGetProgramBuildInfo(
				nbody_ocl_state.program, nbody_ocl_state.devices[0], CL_PROGRAM_BUILD_LOG, NULL, NULL, &length);
			buffer = malloc(length);
			status = clGetProgramBuildInfo(
				nbody_ocl_state.program, nbody_ocl_state.devices[0], CL_PROGRAM_BUILD_LOG, length, buffer, &length);
			printf(buffer);
		}
		opencl_working = 0;
		tried_initialised = 1;
	}
	return opencl_working;
}

void opencl_shutdown() {
	if (nbody_ocl_state.platforms != NULL) {
		clReleaseProgram(nbody_ocl_state.program);
		clReleaseCommandQueue(nbody_ocl_state.queue);
		clReleaseContext(nbody_ocl_state.context);
		free(nbody_ocl_state.devices);
		free(nbody_ocl_state.platforms);
	}
	else {
		assert(nbody_ocl_state.devices == NULL);
	}
	return;
}

int opencl_brute_force_ParticleArr_Arr_ind_vel(
	const cvtx_Particle **array_start,
	const long num_particles,
	const cvtx_Vec3f *mes_start,
	const long num_mes,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel)
{
	char kernel_name[128] = "cvtx_nb_Particle_ind_vel_";
	int i, n_particle_groups, n_zeroed_particles, n_modelled_particles;
	size_t global_work_size[2], workgroup_size[2];
	cl_float3 *mes_pos_buff_data, *part_pos_buff_data, *part_vort_buff_data;
	cl_double3 *res_buff_data;
	cl_float *part_rad_buff_data;
	cl_mem mes_pos_buff, res_buff, *part_pos_buff, *part_vort_buff, *part_rad_buff;
	cl_int status;
	cl_kernel cl_kernel;
	cl_event *event_chain;

	if(opencl_initialise() == 0)
	{
		strncat(kernel_name, kernel->cl_kernel_name_ext, 32);
		cl_kernel = clCreateKernel(nbody_ocl_state.program, kernel_name, &status);
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
		mes_pos_buff = clCreateBuffer(nbody_ocl_state.context, 
			CL_MEM_READ_ONLY, num_mes * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			nbody_ocl_state.queue, mes_pos_buff, CL_FALSE,
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
		res_buff_data = malloc(num_mes * sizeof(cl_double3));
		res_buff = clCreateBuffer(nbody_ocl_state.context, CL_MEM_READ_WRITE,
			sizeof(cl_double3) * num_mes, NULL, &status);
		for (i = 0; i < num_mes; ++i) {
			res_buff_data[i].x = 0;
			res_buff_data[i].y = 0;
			res_buff_data[i].z = 0;
		}
		status = clEnqueueWriteBuffer(
			nbody_ocl_state.queue, res_buff, CL_FALSE,
			0, num_mes * sizeof(cl_double3), res_buff_data, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			assert(false);
			printf("OPENCL:\tFailed to enqueue write buffer.");
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
		part_pos_buff_data = malloc(n_modelled_particles * sizeof(cl_float3));
		part_vort_buff_data = malloc(n_modelled_particles * sizeof(cl_float3));
		part_rad_buff_data = malloc(n_modelled_particles * sizeof(cl_float));
		for (i = 0; i < num_particles; ++i) {
			part_pos_buff_data[i].x = array_start[i]->coord.x[0];
			part_pos_buff_data[i].y = array_start[i]->coord.x[1];
			part_pos_buff_data[i].z = array_start[i]->coord.x[2];
			part_vort_buff_data[i].x = array_start[i]->vorticity.x[0];
			part_vort_buff_data[i].y = array_start[i]->vorticity.x[1];
			part_vort_buff_data[i].z = array_start[i]->vorticity.x[2];
			part_rad_buff_data[i] = array_start[i]->radius;
		}
		/* We need this so that we always have the minimum workgroup size. */
		for (i = num_particles; i < n_modelled_particles; ++i) {
			part_pos_buff_data[i].x = 0;
			part_pos_buff_data[i].y = 0;
			part_pos_buff_data[i].z = 0;
			part_vort_buff_data[i].x = 0;
			part_vort_buff_data[i].y = 0;
			part_vort_buff_data[i].z = 0;
			part_rad_buff_data[i] = 1;
		}
		part_pos_buff  = malloc(n_particle_groups * sizeof(cl_mem));
		part_vort_buff = malloc(n_particle_groups * sizeof(cl_mem));
		part_rad_buff  = malloc(n_particle_groups * sizeof(cl_mem));
		clFinish(nbody_ocl_state.queue);
		event_chain = malloc(sizeof(cl_event) * n_particle_groups * 4);
		for (i = 0; i < n_particle_groups; ++i) {
			part_pos_buff[i] = clCreateBuffer(nbody_ocl_state.context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				nbody_ocl_state.queue, part_pos_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), 
				part_pos_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i);
			assert(status == CL_SUCCESS);
			part_vort_buff[i] = clCreateBuffer(nbody_ocl_state.context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				nbody_ocl_state.queue, part_vort_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				part_vort_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4*i + 1);
			assert(status == CL_SUCCESS);
			part_rad_buff[i] = clCreateBuffer(nbody_ocl_state.context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				nbody_ocl_state.queue, part_rad_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float),
				part_rad_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4*i + 2);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 0, sizeof(cl_mem), part_pos_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 1, sizeof(cl_mem), part_vort_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 2, sizeof(cl_mem), part_rad_buff + i);
			assert(status == CL_SUCCESS);
			if (i == 0) {
				status = clEnqueueNDRangeKernel(nbody_ocl_state.queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 3, event_chain, event_chain + 4 * i + 3);
			} else {
				status = clEnqueueNDRangeKernel(nbody_ocl_state.queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 4, event_chain + 4*i - 1, event_chain + 4 * i + 3);
			}
			if (status != CL_SUCCESS) {
				assert(false);
			}
			clReleaseMemObject(part_pos_buff[i]);
			clReleaseMemObject(part_vort_buff[i]);
			clReleaseMemObject(part_rad_buff[i]);
		}

		/* Read back our results! */
		clEnqueueReadBuffer(nbody_ocl_state.queue, res_buff, CL_TRUE, 0,
			sizeof(cl_double3) * num_mes, res_buff_data, 1,
			event_chain + 4 * n_particle_groups - 1, NULL);
		free(event_chain);	/* Its tempting to do this earlier, but remember, this is asynchonous! */
		for (i = 0; i < num_mes; ++i) {
			result_array[i].x[0] = res_buff_data[i].x;
			result_array[i].x[1] = res_buff_data[i].y;
			result_array[i].x[2] = res_buff_data[i].z;
		}
		free(res_buff_data);

		free(part_pos_buff);
		free(part_vort_buff);
		free(part_rad_buff);
		free(part_pos_buff_data);
		free(part_vort_buff_data);
		free(part_rad_buff_data);
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


int opencl_brute_force_ParticleArr_Arr_ind_dvort(
	const cvtx_Particle **array_start,
	const long num_particles,
	const cvtx_Particle **induced_start,
	const long num_induced,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel) 
{
	char kernel_name[128] = "cvtx_nb_Particle_ind_dvort_";
	int i, n_particle_groups, n_zeroed_particles, n_modelled_particles;
	size_t global_work_size[2], workgroup_size[2];
	cl_float3 *part1_pos_buff_data, *part1_vort_buff_data, *part2_pos_buff_data, *part2_vort_buff_data;
	cl_double3 *res_buff_data;
	cl_float *part1_rad_buff_data;
	cl_mem res_buff, *part1_pos_buff, *part1_vort_buff, *part1_rad_buff, 
		part2_pos_buff, part2_vort_buff;
	cl_int status;
	cl_kernel cl_kernel;
	cl_event *event_chain;

	if (opencl_initialise() == 0)
	{
		strncat(kernel_name, kernel->cl_kernel_name_ext, 32);
		cl_kernel = clCreateKernel(nbody_ocl_state.program, kernel_name, &status);
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
		part2_pos_buff_data = malloc(num_induced * sizeof(cl_float3));
		part2_vort_buff_data = malloc(num_induced * sizeof(cl_float3));
		for (i = 0; i < num_induced; ++i) {
			part2_pos_buff_data[i].x = induced_start[i]->coord.x[0];
			part2_pos_buff_data[i].y = induced_start[i]->coord.x[1];
			part2_pos_buff_data[i].z = induced_start[i]->coord.x[2];
			part2_vort_buff_data[i].x = induced_start[i]->vorticity.x[0];
			part2_vort_buff_data[i].y = induced_start[i]->vorticity.x[1];
			part2_vort_buff_data[i].z = induced_start[i]->vorticity.x[2];
		}
		/* Induced particle Create buffer, enqueue write and set kernel arg. */
		part2_pos_buff = clCreateBuffer(nbody_ocl_state.context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			nbody_ocl_state.queue, part2_pos_buff, CL_TRUE,
			0, num_induced * sizeof(cl_float3), part2_pos_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 3, sizeof(cl_mem), &part2_pos_buff);
		assert(status == CL_SUCCESS);
		part2_vort_buff = clCreateBuffer(nbody_ocl_state.context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			nbody_ocl_state.queue, part2_vort_buff, CL_TRUE,
			0, num_induced * sizeof(cl_float3), part2_vort_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 4, sizeof(cl_mem), &part2_vort_buff);
		assert(status == CL_SUCCESS);
			   		
		/* Generate a results buffer										*/
		res_buff_data = malloc(num_induced * sizeof(cl_double3));
		res_buff = clCreateBuffer(nbody_ocl_state.context, CL_MEM_READ_WRITE,
			sizeof(cl_double3) * num_induced, NULL, &status);
		for (i = 0; i < num_induced; ++i) {
			res_buff_data[i].x = 0;
			res_buff_data[i].y = 0;
			res_buff_data[i].z = 0;
		}
		status = clEnqueueWriteBuffer(
			nbody_ocl_state.queue, res_buff, CL_TRUE,
			0, num_induced * sizeof(cl_double3), res_buff_data, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			assert(false);
			printf("OPENCL:\tFailed to enqueue write buffer.");
		}
		status = clSetKernelArg(cl_kernel, 5, sizeof(cl_mem), &res_buff);
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
		part1_pos_buff_data = malloc(n_modelled_particles * sizeof(cl_float3));
		part1_vort_buff_data = malloc(n_modelled_particles * sizeof(cl_float3));
		part1_rad_buff_data = malloc(n_modelled_particles * sizeof(cl_float));
		for (i = 0; i < num_particles; ++i) {
			part1_pos_buff_data[i].x = array_start[i]->coord.x[0];
			part1_pos_buff_data[i].y = array_start[i]->coord.x[1];
			part1_pos_buff_data[i].z = array_start[i]->coord.x[2];
			part1_vort_buff_data[i].x = array_start[i]->vorticity.x[0];
			part1_vort_buff_data[i].y = array_start[i]->vorticity.x[1];
			part1_vort_buff_data[i].z = array_start[i]->vorticity.x[2];
			part1_rad_buff_data[i] = array_start[i]->radius;
		}
		/* We need this so that we always have the minimum workgroup size. */
		for (i = num_particles; i < n_modelled_particles; ++i) {
			part1_pos_buff_data[i].x = 0;
			part1_pos_buff_data[i].y = 0;
			part1_pos_buff_data[i].z = 0;
			part1_vort_buff_data[i].x = 0;
			part1_vort_buff_data[i].y = 0;
			part1_vort_buff_data[i].z = 0;
			part1_rad_buff_data[i] = 1;
		}
		part1_pos_buff = malloc(n_particle_groups * sizeof(cl_mem));
		part1_vort_buff = malloc(n_particle_groups * sizeof(cl_mem));
		part1_rad_buff = malloc(n_particle_groups * sizeof(cl_mem));
		clFinish(nbody_ocl_state.queue);
		event_chain = malloc(sizeof(cl_event) * n_particle_groups * 4);
		for (i = 0; i < n_particle_groups; ++i) {
			part1_pos_buff[i] = clCreateBuffer(nbody_ocl_state.context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				nbody_ocl_state.queue, part1_pos_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				part1_pos_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i);
			assert(status == CL_SUCCESS);
			part1_vort_buff[i] = clCreateBuffer(nbody_ocl_state.context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				nbody_ocl_state.queue, part1_vort_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				part1_vort_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i + 1);
			assert(status == CL_SUCCESS);
			part1_rad_buff[i] = clCreateBuffer(nbody_ocl_state.context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				nbody_ocl_state.queue, part1_rad_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float),
				part1_rad_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i + 2);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 0, sizeof(cl_mem), part1_pos_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 1, sizeof(cl_mem), part1_vort_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 2, sizeof(cl_mem), part1_rad_buff + i);
			assert(status == CL_SUCCESS);
			if (i == 0) {
				status = clEnqueueNDRangeKernel(nbody_ocl_state.queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 3, event_chain + 4 * i, event_chain + 4 * i + 3);
			}
			else {
				status = clEnqueueNDRangeKernel(nbody_ocl_state.queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 4, event_chain + 4 * i - 1, event_chain + 4 * i + 3);
			}
			assert(status == CL_SUCCESS);
			clReleaseMemObject(part1_pos_buff[i]);
			clReleaseMemObject(part1_vort_buff[i]);
			clReleaseMemObject(part1_rad_buff[i]);
		}

		/* Read back our results! */
		clEnqueueReadBuffer(nbody_ocl_state.queue, res_buff, CL_TRUE, 0,
			sizeof(cl_double3) * num_induced, res_buff_data, 1,
			event_chain + 4 * n_particle_groups - 1, NULL);
		free(event_chain);	/* Its tempting to do this earlier, but remember, this is asynchonous! */
		for (i = 0; i < num_induced; ++i) {
			result_array[i].x[0] = res_buff_data[i].x;
			result_array[i].x[1] = res_buff_data[i].y;
			result_array[i].x[2] = res_buff_data[i].z;
		}
		free(res_buff_data);

		free(part1_pos_buff);
		free(part1_vort_buff);
		free(part1_rad_buff);
		free(part2_pos_buff_data);
		free(part2_vort_buff_data);
		free(part1_pos_buff_data);
		free(part1_vort_buff_data);
		free(part1_rad_buff_data);
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

int opencl_brute_force_ParticleArr_Arr_visc_ind_dvort(
	const cvtx_Particle **array_start,
	const long num_particles,
	const cvtx_Particle **induced_start,
	const long num_induced,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel,
	const float kinematic_visc)
{
	char kernel_name[128] = "cvtx_nb_Particle_visc_ind_dvort_";
	int i, n_particle_groups, n_zeroed_particles, n_modelled_particles;
	size_t global_work_size[2], workgroup_size[2];
	cl_float3 *part1_pos_buff_data, *part1_vort_buff_data, *part2_pos_buff_data, *part2_vort_buff_data;
	cl_double3 *res_buff_data;
	cl_float *part1_rad_buff_data, *part2_rad_buff_data;
	cl_mem res_buff, *part1_pos_buff, *part1_vort_buff, *part1_rad_buff,
		part2_pos_buff, part2_vort_buff, part2_rad_buff;
	cl_int status;
	cl_kernel cl_kernel;
	cl_event *event_chain;

	if (opencl_initialise() == 0)
	{
		strncat(kernel_name, kernel->cl_kernel_name_ext, 32);
		cl_kernel = clCreateKernel(nbody_ocl_state.program, kernel_name, &status);
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
		part2_pos_buff_data = malloc(num_induced * sizeof(cl_float3));
		part2_vort_buff_data = malloc(num_induced * sizeof(cl_float3));
		part2_rad_buff_data = malloc(num_induced * sizeof(cl_float));
		for (i = 0; i < num_induced; ++i) {
			part2_pos_buff_data[i].x = induced_start[i]->coord.x[0];
			part2_pos_buff_data[i].y = induced_start[i]->coord.x[1];
			part2_pos_buff_data[i].z = induced_start[i]->coord.x[2];
			part2_vort_buff_data[i].x = induced_start[i]->vorticity.x[0];
			part2_vort_buff_data[i].y = induced_start[i]->vorticity.x[1];
			part2_vort_buff_data[i].z = induced_start[i]->vorticity.x[2];
			part2_rad_buff_data[i] = induced_start[i]->radius;
		}
		/* Induced particle Create buffer, enqueue write and set kernel arg. */
		part2_pos_buff = clCreateBuffer(nbody_ocl_state.context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			nbody_ocl_state.queue, part2_pos_buff, CL_TRUE,
			0, num_induced * sizeof(cl_float3), part2_pos_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 3, sizeof(cl_mem), &part2_pos_buff);
		assert(status == CL_SUCCESS);
		part2_vort_buff = clCreateBuffer(nbody_ocl_state.context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float3), NULL, &status);
		status = clEnqueueWriteBuffer(
			nbody_ocl_state.queue, part2_vort_buff, CL_TRUE,
			0, num_induced * sizeof(cl_float3), part2_vort_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 4, sizeof(cl_mem), &part2_vort_buff);
		assert(status == CL_SUCCESS);
		part2_rad_buff = clCreateBuffer(nbody_ocl_state.context,
			CL_MEM_READ_ONLY, num_induced * sizeof(cl_float), NULL, &status);
		status = clEnqueueWriteBuffer(
			nbody_ocl_state.queue, part2_rad_buff, CL_TRUE,
			0, num_induced * sizeof(cl_float), part2_rad_buff_data, 0, NULL, NULL);
		assert(status == CL_SUCCESS);
		status = clSetKernelArg(cl_kernel, 5, sizeof(cl_mem), &part2_rad_buff);
		assert(status == CL_SUCCESS);

		/* Generate a results buffer										*/
		res_buff_data = malloc(num_induced * sizeof(cl_double3));
		res_buff = clCreateBuffer(nbody_ocl_state.context, CL_MEM_READ_WRITE,
			sizeof(cl_double3) * num_induced, NULL, &status);
		for (i = 0; i < num_induced; ++i) {
			res_buff_data[i].x = 0;
			res_buff_data[i].y = 0;
			res_buff_data[i].z = 0;
		}
		status = clEnqueueWriteBuffer(
			nbody_ocl_state.queue, res_buff, CL_TRUE,
			0, num_induced * sizeof(cl_double3), res_buff_data, 0, NULL, NULL);
		if (status != CL_SUCCESS) {
			assert(false);
			printf("OPENCL:\tFailed to enqueue write buffer.");
		}
		status = clSetKernelArg(cl_kernel, 6, sizeof(cl_mem), &res_buff);
		assert(status == CL_SUCCESS);

		/* Setup the kinematic viscocity argument */
		cl_float cl_kinem_visc = kinematic_visc;
		status = clSetKernelArg(cl_kernel, 7, sizeof(cl_float), &cl_kinem_visc);
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
		part1_pos_buff_data = malloc(n_modelled_particles * sizeof(cl_float3));
		part1_vort_buff_data = malloc(n_modelled_particles * sizeof(cl_float3));
		part1_rad_buff_data = malloc(n_modelled_particles * sizeof(cl_float));
		for (i = 0; i < num_particles; ++i) {
			part1_pos_buff_data[i].x = array_start[i]->coord.x[0];
			part1_pos_buff_data[i].y = array_start[i]->coord.x[1];
			part1_pos_buff_data[i].z = array_start[i]->coord.x[2];
			part1_vort_buff_data[i].x = array_start[i]->vorticity.x[0];
			part1_vort_buff_data[i].y = array_start[i]->vorticity.x[1];
			part1_vort_buff_data[i].z = array_start[i]->vorticity.x[2];
			part1_rad_buff_data[i] = array_start[i]->radius;
		}
		/* We need this so that we always have the minimum workgroup size. */
		for (i = num_particles; i < n_modelled_particles; ++i) {
			part1_pos_buff_data[i].x = 0;
			part1_pos_buff_data[i].y = 0;
			part1_pos_buff_data[i].z = 0;
			part1_vort_buff_data[i].x = 0;
			part1_vort_buff_data[i].y = 0;
			part1_vort_buff_data[i].z = 0;
			part1_rad_buff_data[i] = 1;
		}
		part1_pos_buff = malloc(n_particle_groups * sizeof(cl_mem));
		part1_vort_buff = malloc(n_particle_groups * sizeof(cl_mem));
		part1_rad_buff = malloc(n_particle_groups * sizeof(cl_mem));
		clFlush(nbody_ocl_state.queue);
		event_chain = malloc(sizeof(cl_event) * n_particle_groups * 4);
		for (i = 0; i < n_particle_groups; ++i) {
			part1_pos_buff[i] = clCreateBuffer(nbody_ocl_state.context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				nbody_ocl_state.queue, part1_pos_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				part1_pos_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i);
			assert(status == CL_SUCCESS);
			part1_vort_buff[i] = clCreateBuffer(nbody_ocl_state.context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float3), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				nbody_ocl_state.queue, part1_vort_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float3),
				part1_vort_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i + 1);
			assert(status == CL_SUCCESS);
			part1_rad_buff[i] = clCreateBuffer(nbody_ocl_state.context,
				CL_MEM_READ_ONLY, CVTX_WORKGROUP_SIZE * sizeof(cl_float), NULL, &status);
			assert(status == CL_SUCCESS);
			status = clEnqueueWriteBuffer(
				nbody_ocl_state.queue, part1_rad_buff[i], CL_FALSE,
				0, CVTX_WORKGROUP_SIZE * sizeof(cl_float),
				part1_rad_buff_data + i * CVTX_WORKGROUP_SIZE, 0, NULL, event_chain + 4 * i + 2);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 0, sizeof(cl_mem), part1_pos_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 1, sizeof(cl_mem), part1_vort_buff + i);
			assert(status == CL_SUCCESS);
			status = clSetKernelArg(cl_kernel, 2, sizeof(cl_mem), part1_rad_buff + i);
			assert(status == CL_SUCCESS);
			if (i == 0) {
				status = clEnqueueNDRangeKernel(nbody_ocl_state.queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 3, event_chain + 4 * i, event_chain + 4 * i + 3);
			}
			else {
				status = clEnqueueNDRangeKernel(nbody_ocl_state.queue, cl_kernel, 2,
					NULL, global_work_size, workgroup_size, 4, event_chain + 4 * i - 1, event_chain + 4 * i + 3);
			}
			assert(status == CL_SUCCESS);
			clReleaseMemObject(part1_pos_buff[i]);
			clReleaseMemObject(part1_vort_buff[i]);
			clReleaseMemObject(part1_rad_buff[i]);
		}

		/* Read back our results! */
		clEnqueueReadBuffer(nbody_ocl_state.queue, res_buff, CL_TRUE, 0,
			sizeof(cl_double3) * num_induced, res_buff_data, 1, 
			event_chain + 4 * n_particle_groups - 1, NULL);
		free(event_chain);	/* Its tempting to do this earlier, but remember, this is asynchonous! */
		for (i = 0; i < num_induced; ++i) {
			result_array[i].x[0] = res_buff_data[i].x;
			result_array[i].x[1] = res_buff_data[i].y;
			result_array[i].x[2] = res_buff_data[i].z;
		}
		free(res_buff_data);

		free(part1_pos_buff);
		free(part1_vort_buff);
		free(part1_rad_buff);
		free(part2_pos_buff_data);
		free(part2_vort_buff_data);
		free(part2_rad_buff_data);
		free(part1_pos_buff_data);
		free(part1_vort_buff_data);
		free(part1_rad_buff_data);
		clReleaseMemObject(res_buff);
		clReleaseMemObject(part2_pos_buff);
		clReleaseMemObject(part2_vort_buff);
		clReleaseMemObject(part2_rad_buff);
		clReleaseKernel(cl_kernel);
		return 0;
	}
	else
	{
		return -1;
	}
}

#endif

