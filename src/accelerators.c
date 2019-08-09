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
#include <stdlib.h>	/* Required for not CVTX_USING_OPENCL */
#include "opencl_acc.h"

CVTX_EXPORT void cvtx_initialise() {
#ifdef CVTX_USING_OPENCL
	opencl_init();
#endif
}

CVTX_EXPORT void cvtx_finalise() {
#ifdef CVTX_USING_OPENCL
	opencl_finalise();
#endif
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
	int pidx, didx;
	opencl_deindex_device(accelerator_id, &pidx, &didx);
	opencl_add_active_device(pidx, didx);
#endif
	return;
}

CVTX_EXPORT void cvtx_accelerator_disable(int accelerator_id) {
#ifdef CVTX_USING_OPENCL
	assert(opencl_is_init() == 1);
	int pidx, didx;
	opencl_deindex_device(accelerator_id, &pidx, &didx);
	opencl_remove_active_device(pidx, didx);
#endif
	return;
}
