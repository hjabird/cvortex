#include "gridkey.h"
/*============================================================================
gridkey.c

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

#include <assert.h>

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
		tzmin = tzmax = array_start[0]->coord.x[3];
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

struct Gridkey2D g_P2D_gridkey2D(
	const cvtx_P2D *particle,
	float grid_density,
	float minx, float miny) {
	assert(particle != NULL);
	assert(grid_density > 0.f);

	struct Gridkey2D ret;
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

struct Gridkey3D g_P3D_gridkey3D(
	const cvtx_P3D *particle,
	float grid_density,
	float minx, float miny, float minz) {
	assert(particle != NULL);
	assert(grid_density > 0.f);

	struct Gridkey3D ret;
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

int comp_Gridkey2D_by_idx(void* context, const void* p1, const void* p2) {
	unsigned int x1, x2, y1, y2, pp1, pp2;
	int ret;
	pp1 = *(unsigned int*)p1;
	pp2 = *(unsigned int*)p2;
	x1 = ((struct Gridkey2D*)context)[pp1].xk;
	x2 = ((struct Gridkey2D*)context)[pp2].xk;
	y1 = ((struct Gridkey2D*)context)[pp1].yk;
	y2 = ((struct Gridkey2D*)context)[pp2].yk;
	if (x1 != x2) {
		ret = x1 < x2 ? -1 : 1;
	}
	else {
		ret = (y1 == y2 ? 0 : (y1 < y2 ? -1 : 1));
	}
	return ret;
}

int comp_Gridkey3D_by_idx(void* context, const void* p1, const void* p2) {
	unsigned int x1, x2, y1, y2, z1, z2, pp1, pp2;
	int ret;
	pp1 = *(unsigned int*)p1;
	pp2 = *(unsigned int*)p2;
	x1 = ((struct Gridkey3D*)context)[pp1].xk;
	x2 = ((struct Gridkey3D*)context)[pp2].xk;
	y1 = ((struct Gridkey3D*)context)[pp1].yk;
	y2 = ((struct Gridkey3D*)context)[pp2].yk;
	z1 = ((struct Gridkey3D*)context)[pp1].zk;
	z2 = ((struct Gridkey3D*)context)[pp2].zk;
	if (x1 != x2) {
		ret = x1 < x2 ? -1 : 1;
	}
	else {
		if (y1 != y2) {
			ret = y1 < y2 ? -1 : 1;
		}
		else {
			ret = z1 < z2 ? -1 : 1;
		}
	}
	return ret;
}
