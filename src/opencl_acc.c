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

#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

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
	static int initialised = 0;
	static int opencl_working = -1;	/* meaning no */
	int good = 1;
	cl_int status;
	cl_uint num_platforms, num_devices;
	if (!initialised) {
		nbody_ocl_state.devices = NULL;
		nbody_ocl_state.platforms = NULL;

		status = clGetPlatformIDs(0, NULL, &num_platforms);
		nbody_ocl_state.platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
		status = clGetPlatformIDs(num_platforms, nbody_ocl_state.platforms, NULL);
		if (status != CL_SUCCESS) {
			printf("OPENCL:\tFailed to find opencl platforms.\n");
			good = 0;
			free(nbody_ocl_state.platforms);
			nbody_ocl_state.platforms = NULL;
		}
	}
	if (!initialised && good) {
		status = clGetDeviceIDs(nbody_ocl_state.platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
		nbody_ocl_state.devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
		status = clGetDeviceIDs(nbody_ocl_state.platforms[0], CL_DEVICE_TYPE_GPU, num_devices, nbody_ocl_state.devices, NULL);
		if (status != CL_SUCCESS) {
			printf("OPENCL:\tFailed to find OpenCl GPU device.\n");
			good = 0;
			free(nbody_ocl_state.platforms);
			nbody_ocl_state.platforms = NULL;
			free(nbody_ocl_state.devices);
			nbody_ocl_state.devices = NULL;
		}
	}
	if (!initialised && good) {
		nbody_ocl_state.context = clCreateContext(NULL, num_devices, nbody_ocl_state.devices, NULL, NULL, &status);
		nbody_ocl_state.queue = clCreateCommandQueue(nbody_ocl_state.context, nbody_ocl_state.devices[0], 0, &status);
		if (status != CL_SUCCESS) {
			printf("OPENCL:\tCould not create context or command queue.\n");
		}
		const char* program_source =
#		include "nbody.cl"
			;
		nbody_ocl_state.program = clCreateProgramWithSource(nbody_ocl_state.context, 1, (const char**)&program_source, NULL, &status);
		status = clBuildProgram(nbody_ocl_state.program, num_devices, nbody_ocl_state.devices, NULL, NULL, NULL);
		if (status != CL_SUCCESS) {
			printf("OPENCL:\tFailed to build opencl program!\n");
			printf("OPENCL:\tBuild log:");
			char buffer[1048 * 16];
			size_t length;
			status = clGetProgramBuildInfo(
				nbody_ocl_state.program, nbody_ocl_state.devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
			printf(buffer);
		}
		opencl_working = 0;
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
	const int num_particles,
	const cvtx_Vec3f *mes_start,
	const int num_mes,
	cvtx_Vec3f *result_array,
	const cvtx_VortFunc *kernel)
{
	opencl_initialise();
	return 0;
}

#endif

