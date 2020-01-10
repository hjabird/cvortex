#ifndef CVTX_UINTKEY_H
#define CVTX_UINTKEY_H
#include "libcvtx.h"
/*============================================================================
uintkey.h

uint32 based keys in 2D and 3D for working with grids

Copyright(c) 2019-2020 HJA Bird

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

#include <stdint.h>

/* A coordinate on a grid in 2D. */
typedef struct UInt32Key2D {
	uint32_t xk;
	uint32_t yk;
} UInt32Key2D;

/* A coordinate on a grid in 3D. */
typedef struct UInt32Key3D {
	uint32_t xk;
	uint32_t yk;
	uint32_t zk;
} UInt32Key3D;

void minmax_xy_posn(
	const cvtx_P2D **array_start, const int nparticles,
	float *xmin, float *xmax, float *ymin, float *ymax);

void minmax_xyz_posn(
	const cvtx_P3D **array_start, const int nparticles,
	float *xmin, float *xmax, float *ymin, float *ymax,
	float *zmin, float *zmax);

void sort_perm_UInt32Key2D(
	UInt32Key2D *gridkeys,
	unsigned int* key_start, size_t num_items);

void sort_perm_UInt32Key3D(
	UInt32Key3D *gridkeys,
	unsigned int* key_start, size_t num_items);

/* Return a location on a grid with origin minx and miny of a 
2D particle. */
struct UInt32Key2D g_P2D_gridkey2D(
	const cvtx_P2D *particle,
	float grid_density,
	float minx, float miny);

/* Return a location on a grid with origin minx and miny of a
2D particle. */
struct UInt32Key3D g_P3D_gridkey3D(
	const cvtx_P3D *particle,
	float grid_density,
	float minx, float miny, float minz);

#endif /* CVTX_UINTKEY_H */
