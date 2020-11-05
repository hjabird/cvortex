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
#ifndef CVTX_OCLPLATFORMSTATE_H
#define CVTX_OCLPLATFORMSTATE_H

#include <string>
#include <vector>
#include <CL/cl.h>

#include "OclDeviceState.h"

class OclPlatformState {
public:
	bool m_good;
protected:
	bool m_initialised_platform_and_program;
	cl_platform_id m_platform;
	std::string m_platform_name;
	cl_program m_program;
	cl_context m_context;
	std::string m_program_build_log;
	bool m_initialised_devices;
	std::vector<OclDeviceState> m_devices;

public:
	/* Constructor doesn't create program, context or devices - use build program. */
	// OclPlatformState();
	OclPlatformState(cl_platform_id);
	~OclPlatformState();
	OclPlatformState(const OclPlatformState&) = delete;
	OclPlatformState& operator=(const OclPlatformState&) = delete;
	OclPlatformState(OclPlatformState&&) noexcept;

	int initialise();
	int number_of_devices();
	OclDeviceState& device(int i);
	const cl_program program();
	const cl_context context();
protected:
	void find_devices();
	void create_device_queues();
	bool create_context();
	void build_program();
};

#endif
#endif
