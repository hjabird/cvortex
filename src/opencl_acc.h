#include "libcvtx.h"
/*============================================================================
opencl_acc.h

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
#ifndef CVTX_OPENCL_ACC_H
#define CVTX_OPENCL_ACC_H

#include <CL/cl.h>
#include <bsv/bsv.h>

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CVTX_WORKGROUP_SIZE 256

struct ocl_platform_state{
	int good;
	cl_platform_id platform;
	int num_devices;
	cl_device_id *devices;
	cl_command_queue *queues;
	cl_program program;
	cl_context context;
	char *platform_name;
	char **device_names;
	char *program_build_log;
};

struct ocl_active_device {
	int platform_idx;
	int device_idx;
};

/* Make OpenCL code ready to use. */
int opencl_init();
int opencl_is_init();

/* Release all OpenCL resources - they can no longer be used. */
void opencl_finalise();

/* The total number of devices that are registered. */
int opencl_num_devices();

/* From a linear index get the index of the platform and that devices
on the platform. Returns -1 for both fields if invalid. */
void opencl_deindex_device(int index, int *plat_idx, int *dev_idx);

/* From and platform and device index, get a linear index. -1 for 
invalid input. */
void opencl_index_device(int *index, int plat_idx, int dev_idx);

/* The number of active devices. -1 for bad. */
int opencl_num_active_devices();

/* Add a device to the list of devices to use by linear index.
returns -1 for invalid index or uninitialised */
int opencl_add_active_device(int plat_idx, int dev_idx);

/* Remove a device from the list of devices to use by linear index.
returns -1 for invalid index or device not used or uninitialised  */
int opencl_remove_active_device(int plat_idx, int dev_idx);

/* Returns the index of a device in the active devices list,
or -1 if it isn't in the list. */
int opencl_device_in_active_list(int plat_idx, int dev_idx);

/* Enable a `default' accelerator on the user's behalf. */
int opencl_enable_default_accelerator(); 

/* Get the program, context and queue for a device
by its active device index. */
int opencl_get_device_state(
	int ad_idx,
	cl_program *program,
	cl_context *context,
	cl_command_queue *queue);

/* Get the name of an accelerator by linear index. */
char* opencl_accelerator_name(int lindex);

#endif CVTX_OPENCL_ACC_H
#endif CVTX_USING_OPENCL
