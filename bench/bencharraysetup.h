#ifndef CVTX_BENCHARRAYSETUP_H
#define CVTX_BENCHARRAYSETUP_H
/*============================================================================
bencharraysetup.h

Set up big random arrays for IO of benchmark functions.

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
#include "libcvtx.h"
#include <bsv/bsv.h>

void create_particles_3D(int np, float maxf, float vol);
void create_particles_3D_outarr(int np);
void destroy_particles_3D();

void create_particles_2D(int np, float maxf, float area);
void create_particles_2D_outarr(int np);
void destroy_particles_2D();

void create_V3f_arr(int n, float maxf);
void create_V3f_arr2(int n, float maxf);
void destroy_V3f_arr();
void destroy_V3f_arr2();

void create_V2f_arr(int n, float maxf);
void create_V2f_arr2(int n, float maxf);
void destroy_V2f_arr();
void destroy_V2f_arr2();

cvtx_P3D** particle_3D_pptr(void);
cvtx_P3D* oparticle_3D_ptr(void);
cvtx_P3D* particle_3D_ptr(void);

cvtx_P2D** particle_2D_pptr(void);
cvtx_P2D* oparticle_2D_ptr(void);
cvtx_P2D* particle_2D_ptr(void);

bsv_V3f* v3f_arr(void);
bsv_V3f* v3f_arr2(void);
bsv_V2f* v2f_arr(void);
bsv_V2f* v2f_arr2(void);

#endif
