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
#include <cassert>
#include <stdio.h>
#include <stdlib.h>	/* Required for not CVTX_USING_OPENCL */
#include <string>
#include "opencl_acc.h"

static void cvtx_info_init(void);
static void cvtx_info_finalise(void);
static std::string compiler_name_string(void);

static std::string cvtx_info_string;

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

CVTX_EXPORT const char* cvtx_information() {
	return cvtx_info_string.c_str();
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

CVTX_EXPORT const char* cvtx_accelerator_name(int accelerator_id) {
#ifdef CVTX_USING_OPENCL
	const char *res = opencl_accelerator_name(accelerator_id);
#else
	char* res = NULL;
#endif
	return res;
}

CVTX_EXPORT int cvtx_accelerator_enabled(int accelerator_id) {
	int res = -1;
#ifdef CVTX_USING_OPENCL
	assert(opencl_is_init() == 1);
	int pidx, didx;
	std::tie(pidx, didx) = opencl_deindex_device(accelerator_id);
	res = opencl_device_in_active_list(pidx, didx);
#endif
	return res >= 0 ? 1 : 0;
}

CVTX_EXPORT void cvtx_accelerator_enable(int accelerator_id) {
#ifdef CVTX_USING_OPENCL
	assert(opencl_is_init() == 1);
	assert(accelerator_id < cvtx_num_accelerators());
	int pidx, didx;
	std::tie(pidx, didx) = opencl_deindex_device(accelerator_id);
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
	std::tie(pidx, didx) = opencl_deindex_device(accelerator_id);
	assert(pidx >= 0);
	assert(didx >= 0);
	opencl_remove_active_device(pidx, didx);
#endif
	return;
}

void cvtx_info_init(void)
{
	const int initial_alloc = 1024 * 16;
	std::string comp_name_buff;
	comp_name_buff = compiler_name_string();
	cvtx_info_string =
		"cvortex version: " + std::to_string(CVORTEX_VERSION_MAJOR) + "." +
		std::to_string(CVORTEX_VERSION_MINOR) + "." +
		std::to_string(CVORTEX_VERSION_PATCH) + "\n" +
		"compiler: " + comp_name_buff + "\n" +
#ifdef CVTX_USING_OPENMP
		"using OpenMP: TRUE\n" +
#else
		"using OpenMP: FALSE\n" +
#endif
#ifdef CVTX_USING_OPENCL
		"using OpenCL: TRUE\n";
#else
		"using OpenCL: FALSE\n";
#endif
}

void cvtx_info_finalise(void)
{
	cvtx_info_string = "";
	return;
}

std::string compiler_name_string(void)
{
	std::string str("Unknown compiler");
#ifdef __clang__
	str = "clang " + std::to_string(__clang_major__) + "." + 
		std::to_string(__clang_minor__)
		+ "." + std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
	str = "GCC " + std::to_string(__GNUC__) + "." + 
		std::to_string(__GNUC_MINOR__)	+ "." + 
		std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
	str = "MSVC " + std::to_string(_MSC_FULL_VER);
#endif
	return str;
}
