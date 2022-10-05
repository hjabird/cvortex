#include "opencl_acc.h"
/*============================================================================
opencl_acc.cpp

Handles the opencl context(s).

Copyright(c) 2019-2020 HJA Bird

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

#include <iostream>
#include <cassert>
#include <cmath>

#include "OclDeviceState.h"
#include "OclPlatformState.h"

struct OclActiveDevice {
	int platform_idx;
	int device_idx;
};

/* THIS is this owner of all of the platforms and devices ------------------*/
static struct {
	bool initialised;
	std::vector<OclPlatformState> platforms;	/* Owner of all OCL state */
	std::vector<OclActiveDevice> active_devices;	/* Devices in use. */
} ocl_state = { 
		false, 
		std::vector<OclPlatformState>(),
		std::vector<OclActiveDevice>() };

/* Returns number of platforms and loads them into the ocl_state. 
-1 for error.*/
static int load_platforms();

int opencl_init() {
	static int tried_init = 0;
	static int good = 0;
	if (tried_init == 0 || (tried_init==1 && ocl_state.initialised == 0)) {
		tried_init = 1;
		ocl_state.initialised = 1;
		ocl_state.platforms.clear();
		ocl_state.active_devices.clear();
		good = load_platforms() > 0 ? 1 : 0;
	}
	opencl_enable_default_accelerator();
	return good;
}

int opencl_is_init() {
	return ocl_state.initialised;
}

void opencl_finalise() {
	ocl_state.platforms.clear(); 
	ocl_state.active_devices.clear();
	ocl_state.initialised = false;
	assert(ocl_state.platforms.size() == 0);
}

int opencl_num_devices() {
	int count = -1;
	if (ocl_state.initialised == 1) {
		count = 0;
		for (OclPlatformState &platform : ocl_state.platforms)
		{
			if (platform.m_good) {
				count += platform.number_of_devices();
			}
		}
	}
	return count;
}

std::tuple<int, int>  opencl_deindex_device(int index) {
	int plat_idx = -1;
	int dev_idx = -1;
	int np, i, nd, acc;
	if (ocl_state.initialised == 1 && index >= 0) {
		np = (int) ocl_state.platforms.size();
		acc = 0;
		for (i = 0; i < np; ++i) {
			OclPlatformState& platform = ocl_state.platforms[i];
			if (!platform.m_good) { continue; }
			nd = platform.number_of_devices();
			if (acc + nd > index) {
				plat_idx = i;
				dev_idx = index - acc;
				break;
			}
			acc += nd;
		}
		/* If index > acc then index is invalid. */
	}
	return std::make_tuple(plat_idx, dev_idx);
}

int opencl_index_device(int plat_idx, int dev_idx) {
	int index = -1;
	int i, acc = 0;
	if (ocl_state.initialised == 1 && plat_idx < ocl_state.platforms.size()
		&& ocl_state.platforms[plat_idx].m_good
		&& ocl_state.platforms[plat_idx].number_of_devices() < dev_idx) {
		for (i = 0; i < plat_idx - 1; ++i) {
			acc += ocl_state.platforms[i].number_of_devices();
		}
		acc += dev_idx;
		index = acc;
	}
	return index;
}

int opencl_num_active_devices() {
	int num;
	assert(opencl_is_init() == 1);
	num = (int)ocl_state.active_devices.size();
	return num;
}

int opencl_add_active_device(int plat_idx, int dev_idx){
	int already_added = 0, retv = -1;
	OclActiveDevice td;
	if (ocl_state.initialised == 1) {
		/* Check we haven't already added this device. */
		already_added = opencl_device_in_active_list(plat_idx, dev_idx) >= 0 ? 1 : 0;
		if (!already_added) {
			td.device_idx = dev_idx;
			td.platform_idx = plat_idx;
			ocl_state.active_devices.emplace_back(td);
			retv = (int) ocl_state.active_devices.size();
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
	if (ocl_state.initialised == 1) {
		lindx = opencl_device_in_active_list(plat_idx, dev_idx);
		if (lindx >= 0) {
			auto& ads = ocl_state.active_devices;
			ads.erase(ads.begin() + lindx);
			retv = (int) ads.size();
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
		for (i = 0; i < ocl_state.active_devices.size(); ++i) {
			if (ocl_state.active_devices[i].platform_idx == plat_idx
				&& ocl_state.active_devices[i].device_idx == dev_idx) {
				break;
			}
		}
		if (i < ocl_state.active_devices.size()) {
			pos = i;
		}
	}
	return pos;
}

int opencl_enable_default_accelerator() {
	int nd, np;
	assert(ocl_state.initialised);
	/* For now we just select the first working device we find. */
	std::tie(np, nd) = opencl_deindex_device(0);
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

	int nd = (int) ocl_state.active_devices.size();
	int retv, didx, pidx;
	if (ad_idx < nd && ad_idx >= 0) {
		didx = ocl_state.active_devices[ad_idx].device_idx;
		pidx = ocl_state.active_devices[ad_idx].platform_idx;
		assert(pidx < ocl_state.platforms.size());
		assert(pidx >= 0);
		assert(didx < ocl_state.platforms[pidx].number_of_devices());
		assert(didx >= 0);
		if (!ocl_state.platforms[pidx].m_good) {
			retv = -1;
		}
		else
		{
			retv = 0;
			*program = ocl_state.platforms[pidx].program();
			*context = ocl_state.platforms[pidx].context();
			*queue = ocl_state.platforms[pidx].device(didx).queue();
		}
	}
	else
	{
		retv = -1;
	}
	return retv;
}

const char* opencl_accelerator_name(int lindex) {
	const char* res = NULL;
	int pidx, didx;
	assert(opencl_is_init() == 1);
	if (lindex >= 0 && lindex < opencl_num_devices()) {
		std::tie(pidx, didx) = opencl_deindex_device(lindex);
		assert(pidx >= 0);
		assert(pidx < ocl_state.platforms.size());
		assert(didx >= 0);
		assert(didx < ocl_state.platforms[pidx].number_of_devices());
		if (	pidx >= 0 && pidx < ocl_state.platforms.size() &&
				didx >= 0 && didx < ocl_state.platforms[pidx].number_of_devices()
			) {
			res = ocl_state.platforms[pidx].device(didx).name_ref().c_str();
		}
	}
	return res;
}

/* STATIC FUNCTIONS ---------------------------------------------------------*/
static int load_platforms() {
	assert(ocl_state.platforms.size() == 0);

	int retv, i;
	cl_int status;
	cl_uint num_platforms;
	std::vector<cl_platform_id> plats;
	status = clGetPlatformIDs(0, NULL, &num_platforms);
	plats.resize(num_platforms);
	status = clGetPlatformIDs(num_platforms, plats.data(), NULL);
	if (status != CL_SUCCESS) {
		retv = -1;
	}
	else
	{
		retv = num_platforms;
		ocl_state.platforms.clear();
		for (i = 0; i < (int)num_platforms; ++i) {
			ocl_state.platforms.emplace_back(plats[i]);
		}
		for (auto& platform : ocl_state.platforms) {
			platform.initialise();
		}
	}
	return retv;
}

#endif
