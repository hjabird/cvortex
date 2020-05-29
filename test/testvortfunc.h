#ifndef CVTX_TEST_VORTFUNC_H
#define CVTX_TEST_VORTFUNC_H

/*============================================================================
testvortfunc.h

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


int testVortFunc(){
    SECTION("VortFunc");
    /* Singular kernel */
    cvtx_VortFunc vfs = cvtx_VortFunc_singular();
    TEST(vfs.g_3D(0.1f) == 1.);
    TEST(vfs.g_3D(3.f) == 1.);
    TEST(vfs.zeta_3D(0.1f) == 0.);
    TEST(vfs.zeta_3D(10.f) == 0.);

    cvtx_VortFunc vfw = cvtx_VortFunc_winckelmans();
    TEST(vfw.g_3D(0.f) == 0.);
    TEST(fabsf(vfw.g_3D(1.f) - (float)0.61872) < 0.00001);
    TEST(fabsf(vfw.g_3D(10.f) - (float)0.9998168) < 0.0000001);
    TEST(vfw.zeta_3D(0.f) == 7.5);
    TEST(fabs(vfw.zeta_3D(10.f) - 7.2433e-7) < 1e-11);

    cvtx_VortFunc vfp = cvtx_VortFunc_planetary();
    TEST(vfp.g_3D(0.f) == 0.);
    TEST(vfp.g_3D(1.f) == 1.);
    TEST(vfp.g_3D(10.f) == 1.);
    TEST(vfp.zeta_3D(0.99f) == 3.);
    TEST(vfp.zeta_3D(1.01f) == 0.);

    cvtx_VortFunc vfg = cvtx_VortFunc_gaussian();
    TEST(vfg.g_3D(0.f) == 0.f);
	TEST(fabs(vfg.g_3D(0.5f) - 0.030859595f) < 1e-6);
    TEST(fabs(vfg.g_3D(1.f) - 0.198748043f) < 1e-6);
	TEST(fabs(vfg.g_3D(2.f) - 0.738535870f) < 1e-6);
	TEST(fabs(vfg.g_3D(4.f) - 0.998866015f) < 1e-6);
	TEST(fabs(vfg.g_3D(6.f) - 0.999999925f) < 1e-6);
	TEST(fabs(vfg.g_3D(8.f) - 0.999999999f) < 1e-6);
    TEST(vfg.g_3D(10.f) == 1.f);
    TEST(fabs(vfg.zeta_3D(1.f) - 0.483941449f) < 1e-6);
    TEST(fabs(vfg.zeta_3D(0.5f) - 0.70413065f) < 1e-6);
    return 0;
}

#endif /* CVTX_TEST_VORTFUNC_H */
