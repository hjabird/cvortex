#include "OclDeviceState.h"
/*============================================================================
OclDeviceState.c

Represents an OpenCL device.

Copyright(c) 2020 HJA Bird

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

#include <cassert>

OclDeviceState::OclDeviceState(cl_platform_id plat_id, cl_device_id dev_id)
	:	m_device_id(dev_id),
		m_device_queue(NULL),
		m_device_name(""),
		m_good(true),
		m_device_queue_initialised(false),
		m_device_info_initialised(false),
		m_device_driver_version(""),
		m_device_compute_units(-1)
{
}

OclDeviceState::~OclDeviceState()
{
	if (m_device_queue != NULL) {
		clReleaseCommandQueue(m_device_queue);
	}
	if (m_device_id != NULL) {
		clReleaseDevice(m_device_id);
	}
}

OclDeviceState::OclDeviceState(OclDeviceState&& orig) noexcept
	: m_good(orig.m_good), 
	m_device_queue_initialised(orig.m_device_queue_initialised),
	m_device_id(orig.m_device_id),
	m_device_name(orig.m_device_name),
	m_device_queue(orig.m_device_queue)
{
	orig.m_good = false;
	orig.m_device_info_initialised = false;
	orig.m_device_id = NULL;
	orig.m_device_name = "";
	orig.m_device_queue = NULL;
}

void OclDeviceState::initialise_device_info()
{
	size_t str_len;
	char txt[1024];
	cl_uint uint_val;
	/* Device name */
	clGetDeviceInfo(m_device_id, CL_DEVICE_NAME, 0, NULL, &str_len);
	str_len = str_len < 1023 ? str_len : 1023;
	clGetDeviceInfo(m_device_id, CL_DEVICE_NAME, str_len,
		txt, NULL);
	m_device_name = txt;
	/* Driver version */
	clGetDeviceInfo(m_device_id, CL_DRIVER_VERSION, 0, NULL, &str_len);
	str_len = str_len < 1023 ? str_len : 1023;
	clGetDeviceInfo(m_device_id, CL_DRIVER_VERSION, str_len,
		txt, NULL);
	m_device_driver_version = txt;
	/* Number of compute units */
	clGetDeviceInfo(m_device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
		&uint_val, NULL);
	m_device_compute_units = uint_val;
	m_device_info_initialised = true;
}

void OclDeviceState::initialise_device_queue(cl_context context)
{
	assert(m_good == true);
	assert(context != NULL);
	cl_int status;
	m_device_queue = clCreateCommandQueue(
		context, m_device_id,
		(cl_command_queue_properties)NULL, &status);
	if (status != CL_SUCCESS) {
		m_good = 0;
	}
	m_device_queue_initialised = true;
}

const cl_device_id OclDeviceState::device_id()
{
	return m_device_id;
}

const cl_command_queue OclDeviceState::queue()
{
	return m_device_queue;
}

std::string& OclDeviceState::name_ref()
{
	return m_device_name;
}

#endif /*CVTX_USING_OPENCL*/

