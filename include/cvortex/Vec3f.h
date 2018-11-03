#ifndef CVTX_VEC3F_H
#define CVTX_VEC3F_H
/*============================================================================
Vec3f.h

Basic representation of a vector in 3d space using floats.

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
#include <stdbool.h>

typedef struct cvtx_Vec3f{
	float x[3];
} cvtx_Vec3f;

float cvtx_Vec3f_abs(const cvtx_Vec3f self);
float cvtx_Vec3f_dot(const cvtx_Vec3f self, const cvtx_Vec3f other);
cvtx_Vec3f cvtx_Vec3f_cross(const cvtx_Vec3f self, const cvtx_Vec3f other);
cvtx_Vec3f cvtx_Vec3f_plus(const cvtx_Vec3f self, const cvtx_Vec3f other);
cvtx_Vec3f cvtx_Vec3f_minus(const cvtx_Vec3f self, const cvtx_Vec3f other);
cvtx_Vec3f cvtx_Vec3f_uminus(const cvtx_Vec3f self);
cvtx_Vec3f cvtx_Vec3f_mult(const cvtx_Vec3f self, const float multiplier);
cvtx_Vec3f cvtx_Vec3f_div(const cvtx_Vec3f self, const float div);
cvtx_Vec3f cvtx_Vec3f_zero(void);
bool cvtx_Vec3f_isequal(const cvtx_Vec3f self, const cvtx_Vec3f other);
bool cvtx_Vec3f_isnequal(const cvtx_Vec3f self, const cvtx_Vec3f other);

#endif /* CVTX_VEC3F_H */