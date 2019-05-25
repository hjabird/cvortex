#include "opencl_acc.h"
/*============================================================================
opencl_acc.c

Handles the opencl context(s).

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
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static struct {
	int initialised;						/* Indicates initialise run */
	int num_platforms;						/* Number of OCL platforms*/
	struct ocl_platform_state *platforms;	/* Owner of all OCL state */
	int num_active_devices;
	struct ocl_active_device *active_devices;	/* Devices in use. */
} ocl_state = { 0, 0, NULL, 0, NULL };

/* Returns number of platforms and loads them into the ocl_state. 
-1 for error.*/
static int load_platforms();

/* Initialises an individual platform into the ocl_state.
Returns 1 if successful, 0 otherwise. */
static int initialise_platform(struct ocl_platform_state *plat);

/* Initialises an ocl_platform_state struct with values 
for an unitialised state (ie. NULL pointers, etc). Returns 0. */
static int zero_new_platform(struct ocl_platform_state *plat);

/* Loads the information about a platform (ie. name, num devices)
and information about the devices (ie. names).
Returns the number of devices if successful, -1 otherwise.*/
static int load_platform_devices(struct ocl_platform_state *plat);

/* Generates an OpenCL context on the platform and builds OCL code.
Platfrom->good is set to 0 zero on any failure. If code fails to build,
platfrom->program_build_log will be set to point to a string describing
the build failure.
Returns 1 if successful, 0 otherwise. */
static int create_platform_context_and_program(struct ocl_platform_state *plat);

/* Generates OCL command queues. 
On failure, platform->good is set to bad (0)
Returns 1 if successful, 0 otherwise. */
static int load_platform_device_queues(struct ocl_platform_state *plat);

/* Deallocates / tears down a platform.
Returns 1. */
static int finalise_platform(struct ocl_platform_state *plat);

int opencl_init() {
	static int tried_init = 0;
	static int good = 0;
	if (tried_init == 0) {
		tried_init = 1;
		ocl_state.initialised = 1;
		ocl_state.num_platforms = 0;
		ocl_state.platforms = NULL;
		ocl_state.num_active_devices = 0;
		ocl_state.active_devices = NULL;
		good = load_platforms();
	}
	opencl_enable_default_accelerator();
	return good;
}

int opencl_is_init() {
	return ocl_state.initialised;
}

void opencl_finalise() {
	int i;
	if (ocl_state.initialised == 1) {
		for (i = 0; i < ocl_state.num_platforms; ++i) {
			finalise_platform(&ocl_state.platforms[i]);
		}
		ocl_state.num_platforms = 0;
		ocl_state.num_active_devices = 0;
		free(ocl_state.platforms);
		ocl_state.platforms = NULL;
		free(ocl_state.active_devices);
		ocl_state.initialised = 0;
	}
	assert(ocl_state.platforms == NULL);
}

int opencl_num_devices() {
	int count = -1;
	int n_plat, i;
	if (ocl_state.initialised == 1) {
		count = 0;
		n_plat = ocl_state.num_platforms;
		if (ocl_state.platforms != NULL) {
			for (i = 0; i < n_plat; ++i)
			{
				count += ocl_state.platforms[i].num_devices;
			}
		}
	}
	return count;
}

void opencl_deindex_device(int index, int *plat_idx, int *dev_idx) {
	*plat_idx = -1;
	*dev_idx = -1;
	int np, i, nd, acc;
	if (ocl_state.initialised == 1 || index < 0) {
		np = ocl_state.num_platforms;
		acc = 0;
		for (i = 0; i < np; ++i) {
			nd = ocl_state.platforms[i].num_devices;
			if (acc + nd > index) {
				*plat_idx = i;
				*dev_idx = index - acc;
				break;
			}
			acc += nd;
		}
		/* If index > acc then index is invalid. */
	}
	return;
}

void opencl_index_device(int *index, int plat_idx, int dev_idx) {
	*index = -1;
	int i, acc = 0;
	if (ocl_state.initialised == 1 && plat_idx < ocl_state.num_platforms
		&& ocl_state.platforms[plat_idx].num_devices < dev_idx) {
		for (i = 0; i < plat_idx - 1; ++i) {
			acc += ocl_state.platforms[i].num_devices;
		}
		acc += dev_idx;
		*index = acc;
	}
}

int opencl_num_active_devices() {
	int num;
	assert(opencl_is_init() == 1);
	num = ocl_state.num_active_devices;
	return num;
}

int opencl_add_active_device(int plat_idx, int dev_idx){
	int already_added = 0, retv;
	struct ocl_active_device *td;
	if (ocl_state.initialised == 1) {
		/* Check we haven't already added this device. */
		already_added = opencl_device_in_active_list(plat_idx, dev_idx) >= 0 ? 1 : 0;
		if (!already_added) {
			ocl_state.active_devices = realloc(
				ocl_state.active_devices,
				sizeof(struct ocl_active_device) * (ocl_state.num_active_devices + 1));
			ocl_state.num_active_devices += 1;
			td = &ocl_state.active_devices[ocl_state.num_active_devices - 1];
			td->device_idx = dev_idx;
			td->platform_idx = plat_idx;
			retv = ocl_state.num_active_devices;
		}
		else
		{
			retv = -1;
		}
	}
	return retv;
}

int opencl_remove_active_device(int plat_idx, int dev_idx) {
	int lindx = 0, retv = -1;	/* linear index */
	struct ocl_active_device *tmp_arr;
	if (ocl_state.initialised == 1) {
		lindx = opencl_device_in_active_list(plat_idx, dev_idx);
		if (lindx >= 0) {
			tmp_arr = malloc(sizeof(struct ocl_active_device) *
				(ocl_state.num_active_devices - 1));
			memcpy(tmp_arr, ocl_state.active_devices,
				sizeof(struct ocl_active_device) * lindx);
			memcpy(tmp_arr, ocl_state.active_devices + (lindx + 1),
				sizeof(struct ocl_active_device) *
				(ocl_state.num_active_devices - lindx - 1));
			free(ocl_state.active_devices);
			ocl_state.active_devices = tmp_arr;
			ocl_state.num_active_devices -= 1;
			retv = ocl_state.num_active_devices;
		}
		else
		{
			retv = -1;
		}
	}
	return retv;
}

int opencl_device_in_active_list(int plat_idx, int dev_idx) {
	int pos = -1, i;
	if (ocl_state.initialised == 1) {
		for (i = 0; i < ocl_state.num_active_devices; ++i) {
			if (ocl_state.active_devices[i].platform_idx == plat_idx
				&& ocl_state.active_devices[i].device_idx == dev_idx) {
				break;
			}
		}
		if (i < ocl_state.num_active_devices) {
			pos = i;
		}
	}
	return pos;
}

int opencl_enable_default_accelerator() {
	int nd, np;
	assert(ocl_state.initialised);
	/* For now we just select the first working device we find. */
	opencl_deindex_device(0, &np, &nd);
	if (np >= 0) {
		opencl_add_active_device(np, nd);
	}
	return np >= 0 ? 1 : 0;
}

int opencl_get_device_state(
	int ad_idx,
	cl_program *program,
	cl_context *context,
	cl_command_queue *queue) 
{
	assert(program != NULL);
	assert(context != NULL);
	assert(queue != NULL);
	assert(ocl_state.initialised);

	int nd = ocl_state.num_active_devices;
	int retv, didx, pidx;
	if (ad_idx < nd && ad_idx >= 0) {
		didx = ocl_state.active_devices[ad_idx].device_idx;
		pidx = ocl_state.active_devices[ad_idx].platform_idx;
		assert(pidx < ocl_state.num_platforms);
		assert(pidx >= 0);
		assert(didx < ocl_state.platforms[pidx].num_devices);
		assert(didx >= 0);
		if (!ocl_state.platforms[pidx].good) {
			retv = -1;
		}
		else
		{
			retv = 0;
			*program = ocl_state.platforms[pidx].program;
			*context = ocl_state.platforms[pidx].context;
			*queue = ocl_state.platforms[pidx].queues[didx];
		}
	}
	else
	{
		retv = -1;
	}
	return retv;
}

char* opencl_accelerator_name(int lindex) {
	char *res = NULL;
	int pidx, didx;
	assert(opencl_is_init() == 1);
	if (lindex >= 0 && lindex < opencl_num_devices()) {
		opencl_deindex_device(lindex, &pidx, &didx);
		assert(pidx >= 0);
		assert(pidx < ocl_state.num_platforms);
		assert(didx >= 0);
		assert(didx < ocl_state.platforms[pidx].num_devices);
		if (pidx >= 0 && pidx < ocl_state.num_platforms &&
				didx >= 0 && didx < ocl_state.platforms[pidx].num_devices &&
				ocl_state.platforms[pidx].device_names != NULL) {
			res = ocl_state.platforms[pidx].device_names[didx];
		}
	}
	return res;
}

/* STATIC FUNCTIONS ---------------------------------------------------------*/
static int load_platforms() {
	assert(ocl_state.platforms == NULL);

	int retv, i;
	cl_int status;
	cl_uint num_platforms;
	cl_platform_id *plats;	/* We can't directly load our structs */
	status = clGetPlatformIDs(0, NULL, &num_platforms);
	plats = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
	status = clGetPlatformIDs(num_platforms, plats, NULL);
	if (status != CL_SUCCESS) {
		ocl_state.platforms = NULL;
		retv = -1;
	}
	else
	{
		retv = num_platforms;
		ocl_state.platforms = malloc(sizeof(struct ocl_platform_state) * num_platforms);
		for (i = 0; i < (int)num_platforms; ++i) {
			zero_new_platform(&ocl_state.platforms[i]);
			ocl_state.platforms[i].platform = plats[i];
			initialise_platform(&ocl_state.platforms[i]);
		}
	}
	free(plats);
	ocl_state.num_platforms = num_platforms;
	return retv;
}

static int initialise_platform(struct ocl_platform_state *plat) {
	int good = 1;
	good = load_platform_devices(plat);
	if (good == 1) {
		good = create_platform_context_and_program(plat);
	}
	if (good == 1) {
		good = load_platform_device_queues(plat);
	}
	return good;
}

static int load_platform_devices(struct ocl_platform_state *plat){
	assert(plat != NULL);			/* Its expected that the platform isn't */
	assert(plat->platform != NULL);
	assert(plat->num_devices == 0);	/* initialised right now. */
	assert(plat->devices == NULL);
	assert(plat->device_names == NULL);
	assert(plat->platform_name == NULL);
	assert(plat->queues == NULL);

	int retv, i;
	cl_int status;
	size_t str_len;
	/* We're only interested in GPUs - cpus are slow in comparison and work better with OpenMP */
	status = clGetDeviceIDs(plat->platform, CL_DEVICE_TYPE_GPU, 0, NULL, &(plat->num_devices));
	plat->devices = malloc(sizeof(cl_device_id) * plat->num_devices);
	status = clGetDeviceIDs(plat->platform, CL_DEVICE_TYPE_GPU, plat->num_devices, plat->devices, NULL);
	if (status != CL_SUCCESS || plat->devices == NULL) {
		free(plat->devices); plat->devices = NULL;
		plat->num_devices = 0;
		retv = -1;
	}
	else
	{
		retv = plat->num_devices;
		/* Platform name */
		clGetPlatformInfo(plat->platform, CL_PLATFORM_NAME, 0, NULL, &str_len);
		plat->platform_name = (char*) malloc(str_len);
		clGetPlatformInfo(plat->platform, CL_PLATFORM_NAME,
			str_len, plat->platform_name, NULL);
		/*Loop over devices*/
		plat->device_names = malloc( sizeof(char*) * plat->num_devices );
		for (i = 0; i < plat->num_devices; ++i) {
			clGetDeviceInfo(plat->devices[i], CL_DEVICE_NAME, 0, NULL, &str_len);
			plat->device_names[i] = (char*)malloc(sizeof(char*) * str_len);
			clGetDeviceInfo(plat->devices[i], CL_DEVICE_NAME, str_len,
				plat->device_names[i], NULL);
		}
	}
	return retv;
}

static int create_platform_context_and_program(struct ocl_platform_state *plat) {
	assert(plat != NULL);
	assert(plat->platform != NULL);
	assert(plat->good = 1);
	assert(plat->program_build_log == NULL);
	assert(plat->context == NULL);
	assert(plat->program == NULL);
	if (plat->num_devices <= 0) { return 0; }

	cl_int status;
	char compile_options[1024] = "";
	char tmp[128];
	const char *program_source =
#		include "nbody.cl"
		;	/* Including in source makes it easier to distribute a shared lib. */
	sprintf(tmp, "%i", CVTX_WORKGROUP_SIZE);
	strcat(compile_options, " -cl-fast-relaxed-math -D CVTX_CL_WORKGROUP_SIZE=");
	strcat(compile_options, tmp);
	sprintf(tmp, "%i", (int)log2(CVTX_WORKGROUP_SIZE));
	strcat(compile_options, " -D CVTX_CL_LOG2_WORKGROUP_SIZE=");
	strcat(compile_options, tmp);

	plat->context = clCreateContext(
		NULL, plat->num_devices, plat->devices, NULL, NULL, &status);
	if (status != CL_SUCCESS) {
		plat->good = 0;
		return plat->good;
	}
	plat->program = clCreateProgramWithSource(
		plat->context, 1, (const char**)&program_source, NULL, &status);
	status = clBuildProgram(plat->program, plat->num_devices, 
		plat->devices, compile_options, NULL, NULL);
	if (status != CL_SUCCESS) {
		plat->good = 0;
		size_t length;
		status = clGetProgramBuildInfo(
			plat->program, plat->devices[0], CL_PROGRAM_BUILD_LOG, 0, 
			NULL, &length);
		plat->program_build_log = malloc(length);
		status = clGetProgramBuildInfo(
			plat->program, plat->devices[0], CL_PROGRAM_BUILD_LOG, length, 
			plat->program_build_log, &length);
	}
	return plat->good;
}

static int load_platform_device_queues(struct ocl_platform_state *plat) {
	assert(plat != NULL);
	assert(plat->platform != NULL);
	assert(plat->good = 1);
	assert(plat->program != NULL);
	assert(plat->context != NULL);
	assert(plat->devices != NULL);
	assert(plat->num_devices >= 0);
	assert(plat->queues == NULL);

	int i;
	cl_uint status;

	plat->queues = malloc(sizeof(cl_command_queue) * plat->num_devices);
	for (i = 0; i < plat->num_devices; ++i) {
		plat->queues[i] = clCreateCommandQueue(
			plat->context, plat->devices[i],
			(cl_command_queue_properties)NULL, &status);
		if (status != CL_SUCCESS) {
			plat->good = 0;
		}
	}
	return plat->good;
}

static int finalise_platform(struct ocl_platform_state *plat) {
	assert(plat != NULL);
	assert(plat->num_devices >= 0);

	int i;
	if (plat->program != NULL) {
		clReleaseProgram(plat->program);
	}
	if (plat->queues != NULL) {
		for (i = 0; i < plat->num_devices; ++i) {
			clReleaseCommandQueue(plat->queues[i]);
		}
		free(plat->queues);
	}
	if (plat->devices != NULL){
		for (i = 0; i < plat->num_devices; ++i) {
			clReleaseDevice(plat->devices[i]);
		}
		free(plat->devices);
	}
	if (plat->device_names != NULL) {
		for (i = 0; i < plat->num_devices; ++i) {
			free(plat->device_names[i]);
		}
		free(plat->device_names);
	}
	plat->num_devices = 0;
	if (plat->context != NULL) { clReleaseContext(plat->context); }
	/* if (plat->platform != NULL) { clReleasePlatform(plat->platform); } */
	free(plat->platform_name);
	free(plat->program_build_log);
	return 1;
}

static int zero_new_platform(struct ocl_platform_state *plat) {
	assert(plat != NULL);
	if (plat != NULL) {
		plat->good = 1;
		plat->platform = NULL;
		plat->num_devices = 0;
		plat->devices = NULL;
		plat->queues = NULL;
		plat->program = NULL;
		plat->context = NULL;
		plat->platform_name = NULL;
		plat->device_names = NULL;
		plat->program_build_log = NULL;
	}
	return 0;
}
#endif
