#ifndef CVTX_CPP_TYPES_H
#define CVTX_CPP_TYPES_H
#include "libcvtx.h"
/*============================================================================
UIntKey64.hpp

uint32 based key for working with 2D grids.

Copyright(c) HJA Bird

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

namespace bsv {

class v3f : public bsv_V3f {
public:
    v3f();
    v3f(float x_, float y_, float z_) {
        x[0] = x_;
        x[1] = y_;
        x[2] = z_;
    }
};

class v2f : public bsv_V2f {
public:
    inline v2f();
    inline v2f(float x_, float y_) {
        x[0] = x_;
        x[1] = y_;
    }
};
}   // namespace bsv




#endif  // CVTX_CPP_TYPES_H




