#include "OclPlatformState.h"
/*============================================================================
OclPlatformState.h

Represents an OpenCL platform.

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
#include <iostream>
#include "opencl_acc.h"

OclPlatformState::OclPlatformState(cl_platform_id plat_id)
	: m_platform(plat_id),
	m_platform_name(), m_program(NULL), m_context(NULL),
	m_program_build_log(), m_devices(), m_good(false), 
	m_initialised_platform_and_program(false),
	m_initialised_devices(false)
{
	assert(plat_id != NULL);
	m_good = true;
	size_t str_len;
	char tmp[1024];
	/* Platform name */
	clGetPlatformInfo(m_platform, CL_PLATFORM_NAME, 0, NULL, &str_len);
	str_len = str_len < 1023 ? str_len : 1023;
	clGetPlatformInfo(m_platform, CL_PLATFORM_NAME,
		str_len, tmp, NULL);
	m_platform_name = std::string(tmp);
}

OclPlatformState::~OclPlatformState()
{
	if (m_program != NULL) {
		clReleaseProgram(m_program);
	}
	if (m_context != NULL) { 
		clReleaseContext(m_context); 
	}
	m_devices.clear();
	/* These'll happen automatically
	platform_name.clear();
	program_build_log.clear(); */
}

OclPlatformState::OclPlatformState(OclPlatformState&& orig) noexcept
	: m_good(orig.m_good),
	m_initialised_platform_and_program(orig.m_initialised_platform_and_program),
	m_platform(orig.m_platform),
	m_platform_name(orig.m_platform_name),
	m_program(orig.m_program),
	m_context(orig.m_context),
	m_program_build_log(orig.m_program_build_log),
	m_initialised_devices(orig.m_initialised_devices),
	m_devices(std::move(orig.m_devices))
{
	orig.m_good = false;
	orig.m_initialised_platform_and_program = false;
	orig.m_platform = NULL;
	orig.m_platform_name.clear();
	orig.m_context = NULL;
	orig.m_program_build_log = "";
	orig.m_initialised_devices = false;
	orig.m_devices.clear();
}

int OclPlatformState::initialise()
{
	assert(m_platform != NULL);
	assert(m_good == true);
	assert(m_program_build_log.size() == 0);
	assert(m_context == NULL);
	assert(m_program == NULL);

	/* Step 1. Find the devices. */
	find_devices();
	/* Step 2. Create a context. */
	if (create_context() == false) { return 0; }
	/* Step 3. Create device queues. */
	create_device_queues();
	/* Step 4. Compile the program. */
	build_program();
	return m_good;
}

void OclPlatformState::find_devices()
{
	assert(m_platform != NULL);
	if (m_initialised_devices == true) {
		return;
	}

	int retv, i;
	cl_int status;
	cl_uint num_devices;
	std::vector<cl_device_id> device_ids;
	/* We're only interested in GPUs - cpus work better with OpenMP */
	status = clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
	device_ids.resize(num_devices);
	status = clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_GPU, 
		(cl_uint)device_ids.size(), device_ids.data(), NULL);
	if (status != CL_SUCCESS) {
		retv = -1;
	}
	else
	{
		retv = num_devices;
		/*Loop over devices*/
		for (i = 0; i < device_ids.size(); ++i) {
			m_devices.emplace_back(m_platform, device_ids[i]);
		}
		for (auto& device : m_devices) {
			device.initialise_device_info();
		}
	}
	m_initialised_devices = true;
}

void OclPlatformState::create_device_queues()
{
	for (auto& device : m_devices) {
		device.initialise_device_queue(m_context);
	}
}

bool OclPlatformState::create_context()
{
	cl_int status;
	std::vector<cl_device_id> device_ids;
	for (auto& device : m_devices) {
		assert(device.device_id() != NULL);
		if (device.m_good) { device_ids.push_back(device.device_id()); }
	}
	m_context = clCreateContext(
		NULL, (cl_uint)device_ids.size(), device_ids.data(), NULL, NULL, &status);
	if (status != CL_SUCCESS) {
		m_good = 0;
		return m_good;
	}
	return m_good;
}

void OclPlatformState::build_program()
{
	cl_int status;
	std::string compile_options;
	char* tmp1;
	const char* tmp2;
	std::string program_source =
#		include "nbody.cl"
		;	/* Including in source makes it easier to distribute a shared lib. */
	std::vector<cl_device_id> device_ids;
	for (auto& device : m_devices) {
		assert(device.device_id() != NULL);
		if (device.m_good) { device_ids.push_back(device.device_id()); }
	}

	/* -cl-fast-relaxed-math is too dangerous - it ruins our NaNs on Nvidia/ */
	compile_options += " -D CVTX_CL_WORKGROUP_SIZE=" +
		std::to_string(CVTX_WORKGROUP_SIZE);
	compile_options += " -D CVTX_CL_LOG2_WORKGROUP_SIZE=" +
		std::to_string((int)log2(CVTX_WORKGROUP_SIZE));
	tmp2 = program_source.c_str();
	m_program = clCreateProgramWithSource(
		m_context, 1, (const char**)&tmp2, NULL, &status);
	status = clBuildProgram(m_program, (cl_uint)device_ids.size(),
		device_ids.data(), compile_options.c_str(), NULL, NULL);
	if (status != CL_SUCCESS) {
		m_good = 0;
	}
	/* It can be useful to have the buildlog even for good builds. */
	size_t length;
	status = clGetProgramBuildInfo(
		m_program, device_ids[0], CL_PROGRAM_BUILD_LOG, 0,
		NULL, &length);
	tmp1 = (char*)malloc(sizeof(char) * (length + 1));
	status = clGetProgramBuildInfo(
		m_program, device_ids[0], CL_PROGRAM_BUILD_LOG, length,
		tmp1, &length);
	m_program_build_log = tmp1;
#ifdef _DEBUG
	if (!m_good) {
		std::cout << "ERROR:\tFailed to build CVortex OpenCL kernel.\n"
			<< "\tOn platform: " << m_platform_name
			<< "\n\tGives build log:\n\n" << m_program_build_log << std::endl;
	}
#endif
	/* If the platform isn't `good` its almost certainly a failing of the library
	that ought to be fixed. */
	assert(m_good);
	free(tmp1);
	return;
}

int OclPlatformState::number_of_devices()
{
	return (int) m_devices.size();
}

OclDeviceState& OclPlatformState::device(int i)
{
	return m_devices[i];
}

const cl_program OclPlatformState::program()
{
	return m_program;
}

const cl_context OclPlatformState::context()
{
	return m_context;
}

#endif /*CVTX_USING_OPENCL*/
