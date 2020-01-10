#include "uintkey.h"
/*============================================================================
uintkey.c

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

#include <assert.h>
#include <stdlib.h>

void minmax_xy_posn(
	const cvtx_P2D **array_start, const int nparticles,
	float *xmin, float *xmax, float *ymin, float *ymax) {
	assert(array_start != NULL);
	assert(nparticles >= 0);
	float txmin, txmax, tymin, tymax;
	int i;

	if (nparticles > 0) {
		bsv_V2f coord = array_start[0]->coord;
		txmin = txmax = coord.x[0];
		tymin = tymax = coord.x[1];
	}
	else {
		txmin = txmax = tymin = tymax = 0.f;
	}
	for (i = 1; i < nparticles; ++i) {
		assert(array_start[i] != NULL);
		float x, y;
		x = array_start[i]->coord.x[0];
		y = array_start[i]->coord.x[1];
		txmin = txmin > x ? x : txmin;
		txmax = txmax < x ? x : txmax;
		tymin = tymin > y ? y : tymin;
		tymax = tymax < y ? y : tymax;
	}
	if (xmin != NULL) { *xmin = txmin; }
	if (xmax != NULL) { *xmax = txmax; }
	if (ymin != NULL) { *ymin = tymin; }
	if (ymax != NULL) { *ymax = tymax; }
	return;
}

void minmax_xyz_posn(
	const cvtx_P3D **array_start, const int nparticles,
	float *xmin, float *xmax, float *ymin, float *ymax,
	float *zmin, float *zmax) {
	assert(array_start != NULL);
	assert(nparticles >= 0);
	float txmin, txmax, tymin, tymax, tzmin, tzmax;
	int i;

	if (nparticles > 0) {
		txmin = txmax = array_start[0]->coord.x[0];
		tymin = tymax = array_start[0]->coord.x[1];
		tzmin = tzmax = array_start[0]->coord.x[2];
	}
	else {
		txmin = txmax = tymin = tymax = tzmin = tzmax = 0.f;
	}
	for (i = 1; i < nparticles; ++i) {
		assert(array_start[i] != NULL);
		float x, y, z;
		x = array_start[i]->coord.x[0];
		y = array_start[i]->coord.x[1];
		z = array_start[i]->coord.x[2];
		txmin = txmin > x ? x : txmin;
		txmax = txmax < x ? x : txmax;
		tymin = tymin > y ? y : tymin;
		tymax = tymax < y ? y : tymax;
		tzmin = tzmin > z ? z : tzmin;
		tzmax = tzmax < z ? z : tzmax;
	}
	if (xmin != NULL) { *xmin = txmin; }
	if (xmax != NULL) { *xmax = txmax; }
	if (ymin != NULL) { *ymin = tymin; }
	if (ymax != NULL) { *ymax = tymax; }
	if (zmin != NULL) { *zmin = tzmin; }
	if (zmax != NULL) { *zmax = tzmax; }
	return;
}

struct UInt32Key2D g_P2D_gridkey2D(
	const cvtx_P2D *particle,
	float grid_density,
	float minx, float miny) {
	assert(particle != NULL);
	assert(grid_density > 0.f);

	struct UInt32Key2D ret;
	float x, y;
	x = particle->coord.x[0];
	y = particle->coord.x[1];
	x = (x - minx) / grid_density;
	y = (y - miny) / grid_density;
	assert(x >= 0);
	assert(y >= 0);
	ret.xk = (unsigned int)(roundf(x));
	ret.yk = (unsigned int)(roundf(y));
	return ret;
}

struct UInt32Key3D g_P3D_gridkey3D(
	const cvtx_P3D *particle,
	float grid_density,
	float minx, float miny, float minz) {
	assert(particle != NULL);
	assert(grid_density > 0.f);

	struct UInt32Key3D ret;
	float x, y, z;
	x = particle->coord.x[0];
	y = particle->coord.x[1];
	z = particle->coord.x[2];
	x = (x - minx) / grid_density;
	y = (y - miny) / grid_density;
	z = (z - minz) / grid_density;
	assert(x >= 0);
	assert(y >= 0);
	assert(z >= 0);
	ret.xk = (unsigned int)(roundf(x));
	ret.yk = (unsigned int)(roundf(y));
	ret.zk = (unsigned int)(roundf(z));
	return ret;
}

