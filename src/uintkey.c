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
#ifdef CVTX_USING_OPENMP
#	include <omp.h>
#endif

/* STATIC DECLARATIONS -----------------------------------------------------*/

/*
Get the permutation of the indecies needed to sort an array.
START:	UI		= [3, 2, 6, 4]
		KEYS	= [0, 0, 0, 0]
END:	UI		= [3, 2, 6, 4]
		KEYS	= [1, 0, 3, 2]
UI is assumed to be array of unsigned integers of
uibytes size.
Uses parallel radix-8 sorting method.

ui_start is an array of uibytes * num_items bytes long.
key_start is an array of unsigned ints num_items long.
*/
static void sort_perm_multibyte_radix8(
	unsigned char* ui_start, size_t uibytes,
	unsigned int* key_start, size_t num_items);

static int sort_perm_uint32key2d_quicksort(
	UInt32Key2D* ui_start,
	unsigned int* key_start, size_t num_items);

static int sort_perm_uint32key3d_quicksort(
	UInt32Key3D* ui_start,
	unsigned int* key_start, size_t num_items);


/*	DEFINITIONS ------------------------------------------------------------*/

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

void sort_perm_UInt32Key2D(
	UInt32Key2D *gridkeys,
	unsigned int* key_start, size_t num_items) {
	if (num_items < 100000) {
		sort_perm_uint32key2d_quicksort(gridkeys,
			key_start, num_items);
	}
	else {
		sort_perm_multibyte_radix8(
			(unsigned char*)gridkeys, sizeof(UInt32Key2D),
			key_start, num_items);
	}
}

void sort_perm_UInt32Key3D(
	UInt32Key3D *gridkeys,
	unsigned int* key_start, size_t num_items) {
	if (num_items < 100000) {
		sort_perm_uint32key3d_quicksort(
			gridkeys, key_start, num_items);
	}
	else {
		sort_perm_multibyte_radix8(
			(unsigned char *)gridkeys, sizeof(UInt32Key3D),
			key_start, num_items);
	}
}

UInt32Key2D g_P2D_gridkey2D(
	const cvtx_P2D *particle,
	float grid_density,
	float minx, float miny) {
	assert(particle != NULL);
	assert(grid_density > 0.f);

	UInt32Key2D ret;
	float x, y;
	x = particle->coord.x[0];
	y = particle->coord.x[1];
	x = (x - minx) / grid_density;
	y = (y - miny) / grid_density;
	assert(x >= 0);
	assert(y >= 0);
	ret.k.x = (unsigned int)(roundf(x));
	ret.k.y = (unsigned int)(roundf(y));
	return ret;
}

UInt32Key3D g_P3D_gridkey3D(
	const cvtx_P3D *particle,
	float grid_density,
	float minx, float miny, float minz) {
	assert(particle != NULL);
	assert(grid_density > 0.f);

	UInt32Key3D ret;
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
	ret.k.x = (unsigned int)(roundf(x));
	ret.k.y = (unsigned int)(roundf(y));
	ret.k.z = (unsigned int)(roundf(z));
	return ret;
}

void sort_perm_multibyte_radix8(
	unsigned char* ui_start, size_t uibytes,
	unsigned int* key_start, size_t num_items) {
	/* A parallel radix sort where only the permutation of
	the sort is recorded.

	key_start[:] is changed. ui_start[:] is not.

	During sorting, ordering is obtained from the working array ("wa")
	and the new order output into the output array ("oa"). On each pass
	the working array and output array are swapped.

	The radix sort works 1 byte at a time by:
	[loop] for each byte:
		[parallel]	count number of each value (ui_start[])
		[parallel]	combine counts and compute offsets
		[parallel]	reorder index data using offset
	[end loop]
	[serial]	make sure data is in the key_start array.
	*/

	assert(uibytes > 0);
	assert(num_items >= 0);
	assert(ui_start != NULL);
	assert(key_start != NULL);

	/* Buffer (buffer) swaps with input array
	as working array (wa) or output array (oa) */
	unsigned int *buffer = NULL, *wa = NULL, *oa = NULL;
	unsigned int n_para = 0x1 << sizeof(char) * 8;
#ifdef CVTX_USING_OPENMP
	unsigned int nthreads = omp_get_num_procs();
	omp_set_dynamic(0);
	omp_set_num_threads(nthreads);
#else
	unsigned int nthreads = 1;
#endif
	unsigned int *counts, *offsets, info_size = sizeof(unsigned int) * n_para;
	/* swap = what are we writing the result of this iter into? */
	unsigned int bit = 0, byte = 0, swap = 0, uini = (unsigned int)num_items;
	buffer = malloc(num_items * sizeof(unsigned int));
	counts = malloc(info_size * nthreads);
	offsets = malloc(info_size * nthreads);

	/* Number permutation array. */
	int ci;
	for (ci = 0; ci < (int)uini; ++ci) {
		key_start[ci] = ci;
	}

	for (byte = 0; byte < uibytes; ++byte) {
		memset(counts, 0, info_size * nthreads);
		memset(offsets, 0, info_size * nthreads);
		wa = swap ? buffer : key_start;
		oa = swap ? key_start : buffer;
		int i = 0, j = 0, m = 0, n = 0, threadid;

		/* Counting pass */
#pragma omp parallel for private(m, n, j)
		for (threadid = 0; threadid < (int)nthreads; ++threadid) {
			m = (uini / nthreads) * threadid;
			n = threadid == nthreads - 1 ?
				uini : (uini / nthreads) * (threadid + 1);
			unsigned int k;
			for (j = m; j < n; ++j) {
				k = ui_start[wa[j] * uibytes + byte];
				counts[k + n_para * threadid]++;
			}
		}
		/* Compute offsets */
#pragma omp parallel for private(n, j, i)
		for (m = 0; m < (int)nthreads; ++m) {
			for (n = 0; n < (int)n_para; ++n) {
				for (j = 0; j < (int)nthreads; ++j) {
					for (i = 0; i < n; ++i) {
						offsets[n + m * n_para] +=
							counts[i + j * n_para];
					}
				}
				for (j = 0; j < m; ++j) {
					offsets[n + m * n_para] +=
						counts[n + j * n_para];
				}
			}
		}
		/* Reorder pass*/
#pragma omp parallel for private(m, n, j)
		for (threadid = 0; threadid < (int)nthreads; ++threadid) {
			m = (uini / nthreads) * threadid;
			n = threadid == nthreads - 1 ?
				uini : (uini / nthreads) * (threadid + 1);
			for (j = m; j < n; ++j) {
				unsigned int k = ui_start[wa[j] * uibytes + byte]
					+ n_para * threadid;
				oa[offsets[k]] = wa[j];
				++offsets[k];
			}
		}
		swap = swap ? 0 : 1;
	}
	/* If we wrote our solution into the buffer, we need to copy
	it back. */
	if (!swap) {
		memcpy(key_start, buffer, num_items * sizeof(unsigned int));
	}
	free(buffer);
	free(counts);
	free(offsets);
#ifdef CVTX_USING_OPENMP
	omp_set_dynamic(1);
#endif
	return;
}

static int sort_perm_uint32key2d_quicksort(
	UInt32Key2D* ui_start,
	unsigned int* key_start, size_t num_items) {
	
	const int stack_max = 1024;
	int i, good = 1, stack_pos = 0;	
	size_t *highs, *lows, high, low, partition;
	uint64_t piv;
	unsigned int tmp;
	highs = (size_t*)malloc(sizeof(size_t) * stack_max);
	lows = (size_t*)malloc(sizeof(size_t) * stack_max);

	for (i = 0; i < (int)num_items; ++i) {
		key_start[i] = i;
	}

	highs[stack_pos] = num_items-1;
	lows[stack_pos] = 0;
	while (1) {
		if (stack_pos < 0) { break; }
		/* We're using the Hoare partition scheme 
		See the internet for an explanation. */
		high = highs[stack_pos];
		low = lows[stack_pos];
		assert(high > low);
		piv = ui_start[key_start[low + (high-low) / 2]].v;
		low--; high++;
		while (1) {
			do { high--; } while (ui_start[key_start [high]].v > piv);
			do { low++; } while (ui_start[key_start[low]].v < piv);
			if (low < high) { 
				tmp = key_start[low];
				key_start[low] = key_start[high];
				key_start[high] = tmp;
			}
			else {
				partition = high;
				break;
			}
		}
		if (stack_pos + 1> stack_max) {
			assert(0);
			good = 0;
			break;
		}
		/* Put the larger "side" on top of the stack to work on first. */
		if (highs[stack_pos] - partition - 1 > partition - lows[stack_pos]) {
			/* highs[stack_pos] = highs[stack_pos]; */
			highs[stack_pos + 1] = partition;
			lows[stack_pos + 1] = lows[stack_pos];
			lows[stack_pos] = partition+1;
		}
		else {
			/* lows[stack_pos] = lows[stack_pos]; */
			highs[stack_pos+1] = highs[stack_pos];
			highs[stack_pos] = partition;
			lows[stack_pos+1] = partition+1;
		}
		/* If the small interval we just put on top is 0 we can ignore it. */
		if (highs[stack_pos+1] - lows[stack_pos+1] > 0) { 
			stack_pos += 1;
		}
	}
	free(highs);
	free(lows);
	return good;
}

static int sort_perm_uint32key3d_quicksort(
	UInt32Key3D* ui_start,
	unsigned int* key_start, size_t num_items) {

	const int stack_max = 1024;
	int i, good = 1, stack_pos = 0;
	size_t* highs, * lows, high, low, partition;
	UInt32Key3D piv, cmpv;
	unsigned int tmp;
	highs = (size_t*)malloc(sizeof(size_t) * stack_max);
	lows = (size_t*)malloc(sizeof(size_t) * stack_max);

	for (i = 0; i < (int)num_items; ++i) {
		key_start[i] = i;
	}

	highs[stack_pos] = num_items - 1;
	lows[stack_pos] = 0;
	while (1) {
		if (stack_pos < 0) { break; }
		/* We're using the Hoare partition scheme
		See the internet for an explanation. */
		high = highs[stack_pos];
		low = lows[stack_pos];
		assert(high > low);
		piv = ui_start[key_start[low + (high - low) / 2]];
		low--; high++;
		while (1) {
			do { 
				high--; 
				cmpv = ui_start[key_start[high]];
			} while (!(cmpv.v.lo < piv.v.lo && cmpv.v.up < piv.v.up));
			do { 
				low++;
				cmpv = ui_start[key_start[low]];
			} while (!(cmpv.v.lo > piv.v.lo && cmpv.v.up > piv.v.up));
			if (low < high) {
				tmp = key_start[low];
				key_start[low] = key_start[high];
				key_start[high] = tmp;
			}
			else {
				partition = high;
				break;
			}
		}
		if (stack_pos + 1 > stack_max) {
			assert(0);
			good = 0;
			break;
		}
		/* Put the larger "side" on top of the stack to work on first. */
		if (highs[stack_pos] - partition - 1 > partition - lows[stack_pos]) {
			/* highs[stack_pos] = highs[stack_pos]; */
			highs[stack_pos + 1] = partition;
			lows[stack_pos + 1] = lows[stack_pos];
			lows[stack_pos] = partition + 1;
		}
		else {
			/* lows[stack_pos] = lows[stack_pos]; */
			highs[stack_pos + 1] = highs[stack_pos];
			highs[stack_pos] = partition;
			lows[stack_pos + 1] = partition + 1;
		}
		/* If the small interval we just put on top is 0 we can ignore it. */
		if (highs[stack_pos + 1] - lows[stack_pos + 1] > 0) {
			stack_pos += 1;
		}
	}
	free(highs);
	free(lows);
	return good;
}

