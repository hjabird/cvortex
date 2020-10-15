#include "libcvtx.h"
/*============================================================================
accelerators.c

Library initialisation and accelerator control.

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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>	/* Required for not CVTX_USING_OPENCL */
#include "opencl_acc.h"

static void cvtx_info_init(void);
static void cvtx_info_finalise(void);
static void compiler_name_string(char*);

static char* cvtx_info_string = NULL;

CVTX_EXPORT void cvtx_initialise() {
#ifdef CVTX_USING_OPENCL
	opencl_init();
#endif
	cvtx_info_init();
}

CVTX_EXPORT void cvtx_finalise() {
#ifdef CVTX_USING_OPENCL
	opencl_finalise();
#endif
	cvtx_info_finalise();
}

CVTX_EXPORT char* cvtx_information() {
	return cvtx_info_string;
}

CVTX_EXPORT int cvtx_num_accelerators() {
	int num_accelerators = 0;
#ifdef CVTX_USING_OPENCL
	assert(opencl_is_init() == 1);
	num_accelerators = opencl_num_devices();
#endif
	return num_accelerators;
}

CVTX_EXPORT int cvtx_num_enabled_accelerators() {
	int num = 0;
#ifdef CVTX_USING_OPENCL
	num = opencl_num_active_devices();
#endif
	return num;
}

CVTX_EXPORT char* cvtx_accelerator_name(int accelerator_id) {
	char *res = NULL;
#ifdef CVTX_USING_OPENCL
	res = opencl_accelerator_name(accelerator_id);
#endif
	return res;
}

CVTX_EXPORT int cvtx_accelerator_enabled(int accelerator_id) {
	int res = -1;
#ifdef CVTX_USING_OPENCL
	assert(opencl_is_init() == 1);
	int pidx, didx;
	opencl_deindex_device(accelerator_id, &pidx, &didx);
	res = opencl_device_in_active_list(pidx, didx);
#endif
	return res >= 0 ? 1 : 0;
}

CVTX_EXPORT void cvtx_accelerator_enable(int accelerator_id) {
#ifdef CVTX_USING_OPENCL
	assert(opencl_is_init() == 1);
	assert(accelerator_id < cvtx_num_accelerators());
	int pidx, didx;
	opencl_deindex_device(accelerator_id, &pidx, &didx);
	assert(pidx >= 0);
	assert(didx >= 0);
	opencl_add_active_device(pidx, didx);
#endif
	return;
}

CVTX_EXPORT void cvtx_accelerator_disable(int accelerator_id) {
#ifdef CVTX_USING_OPENCL
	assert(opencl_is_init() == 1);
	assert(accelerator_id < cvtx_num_accelerators());
	int pidx, didx;
	opencl_deindex_device(accelerator_id, &pidx, &didx);
	assert(pidx >= 0);
	assert(didx >= 0);
	opencl_remove_active_device(pidx, didx);
#endif
	return;
}

void cvtx_info_init(void)
{
	int nchar;
	const int initial_alloc = 1024 * 16;
	cvtx_info_string = (char*)malloc(initial_alloc); /* Lots of space... */
	char comp_name_buff[128];
	if (cvtx_info_string != NULL) {
		compiler_name_string(comp_name_buff);
		nchar = sprintf(cvtx_info_string,
			"cvortex version: %d.%d.%d\n"
			"compiler: %s\n"
			"using OpenMP: %s\n"
			"using OpenCL: %s\n",
			CVORTEX_VERSION_MAJOR, CVORTEX_VERSION_MINOR, CVORTEX_VERSION_PATCH,
			comp_name_buff,
#ifdef CVTX_USING_OPENMP
			"TRUE",
#else
			"FALSE",
#endif
#ifdef CVTX_USING_OPENCL
			"TRUE"
#else
			"FALSE"
#endif
		);
		assert(nchar > 0);
		assert(nchar < initial_alloc);
		cvtx_info_string = (char*) realloc(cvtx_info_string, nchar + 1);
	}
}

void cvtx_info_finalise(void)
{
	free(cvtx_info_string);
	cvtx_info_string = NULL;
	return;
}

void compiler_name_string(char* buffer)
{
#ifdef __clang__
	sprintf(buffer,
		"clang %d.%d.%d",
		__clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
	sprintf(buffer,
		"GCC %d.%d.%d",
		__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
	sprintf(buffer,
		"MSVC %d", _MSC_FULL_VER);
#endif
}
