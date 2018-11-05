#ifndef CVTX_TEST_VORTFUNC_H
#define CVTX_TEST_VORTFUNC_H

/*============================================================================
testparticle.h

Test functionality of vortex particle & methods.

Copyright(c) 2018 HJA Bird

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
#include "../include/cvortex/VortFunc.h"

#include <math.h>


int testVortFunc(){
    SECTION("VortFunc");
    /* Singular kernel */
    cvtx_VortFunc vfs = cvtx_VortFunc_singular();
    TEST(vfs.g_fn(0.1) == 1.);
    TEST(vfs.g_fn(3) == 1.);
    TEST(vfs.zeta_fn(0.1) == 0.);
    TEST(vfs.zeta_fn(10) == 0.);

    cvtx_VortFunc vfw = cvtx_VortFunc_winckelmans();
    TEST(vfw.g_fn(0.) == 0.);
    TEST(fabsf(vfw.g_fn(1.) - 0.61872) < 0.00001);
    TEST(fabsf(vfw.g_fn(10.) - 0.9998168) < 0.0000001);
    TEST(vfw.zeta_fn(0) == 7.5);
    TEST(fabs(vfw.zeta_fn(10) - 7.2433e-7) < 1e-11);

    return 0;
}

#endif /* CVTX_TEST_VORTFUNC_H */