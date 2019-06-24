#ifndef CVTX_GRIDKEY_H
#define CVTX_GRIDKEY_H
#include "libcvtx.h"
/*============================================================================
gridkey.h

Helper functions for working with grids.

Copyright(c) 2019 HJA Bird

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

/* A coordinate on a grid in 2D. */
struct Gridkey2D {
	unsigned int xk;
	unsigned int yk;
};

/* A coordinate on a grid in 3D. */
struct Gridkey3D {
	unsigned int xk;
	unsigned int yk;
	unsigned int zk;
};

void minmax_xy_posn(
	const cvtx_P2D **array_start, const int nparticles,
	float *xmin, float *xmax, float *ymin, float *ymax);

void minmax_xyz_posn(
	const cvtx_P3D **array_start, const int nparticles,
	float *xmin, float *xmax, float *ymin, float *ymax,
	float *zmin, float *zmax);

/* Return a location on a grid with origin minx and miny of a 
2D particle. */
struct Gridkey2D g_P2D_gridkey2D(
	const cvtx_P2D *particle,
	float grid_density,
	float minx, float miny);

/* Return a location on a grid with origin minx and miny of a
2D particle. */
struct Gridkey3D g_P3D_gridkey3D(
	const cvtx_P3D *particle,
	float grid_density,
	float minx, float miny, float minz);

/*
Takes two indexes, p1 & p2 that index an array of Gridkey2D s.
Array is of Gridkey2D is given by context.
Compares an Gridkey2D by x key then y key. */
int comp_Gridkey2D_by_idx(void* context, const void* p1, const void* p2);

/*
Takes two indexes, p1 & p2 that index an array of Gridkey2D s.
Array is of Gridkey3D is given by context.
Compares an Gridkey3D by x key then y key then z key. */
int comp_Gridkey3D_by_idx(void* context, const void* p1, const void* p2);


#endif // !CVTX_GRIDKEY_H
