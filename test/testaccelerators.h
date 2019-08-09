#ifndef CVTX_TEST_ACCELERATORS_H
#define CVTX_TEST_ACCELERATORS_H

/*============================================================================
testparticle.h

Test functionality of vortex particle & methods.

Copyright(c) 2018-2019 HJA Bird

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
#include "../include/cvortex/libcvtx.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int testAccelerators(){
    SECTION("Accelerators");
    int i, status;

    TEST(cvtx_num_accelerators() >= 0);
	/* This is test system dependent. 
	Who has more that 3 gpus though? */
	TEST(cvtx_num_accelerators() < 4);
	if(cvtx_num_accelerators() == 0){
		printf("WARNING: cvtx found no accelerators. "
			"If this build is using CVTX_USE_OPENCL \n"
			"and you're expecting to find GPUs, "
			"something is wrong!\n");
	}
	for(i = 0; i < cvtx_num_accelerators(); ++i){
		TEST(cvtx_accelerator_name(i) != NULL);
		status = cvtx_accelerator_enabled(i);
		TEST((status == 1) || (status == 0));
		if(status == 1){
			cvtx_accelerator_disable(i);
			TEST(cvtx_accelerator_enabled(i) == 0);
			cvtx_accelerator_enable(i);
		} 
		else
		{
			cvtx_accelerator_enable(i);
			TEST(cvtx_accelerator_enabled(i) == 0);
			cvtx_accelerator_disable(i);
		} 
	}

    return 0;
}


#endif /* CVTX_TEST_ACCELERATORS_H */
