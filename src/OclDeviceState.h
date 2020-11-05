/*============================================================================
OclDeviceState.h

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
#ifndef CVTX_OCLDEVICESTATE_H
#define CVTX_OCLDEVICESTATE_H

#include <string>
#include <CL/cl.h>

class OclDeviceState {
public:
	bool m_good; 
protected:
	bool m_device_queue_initialised;
	bool m_device_info_initialised;
	cl_device_id m_device_id;
	std::string m_device_name;
	cl_command_queue m_device_queue;
	std::string m_device_driver_version;
	int m_device_compute_units;

public:
	OclDeviceState(cl_platform_id, cl_device_id);
	~OclDeviceState();
	OclDeviceState(const OclDeviceState&) = delete;
	OclDeviceState& operator=(const OclDeviceState&) = delete;
	OclDeviceState(OclDeviceState&&) noexcept;

	void initialise_device_info();
	void initialise_device_queue(cl_context context);
	const cl_device_id device_id();
	const cl_command_queue queue();
	std::string& name_ref();
};

#endif
#endif
