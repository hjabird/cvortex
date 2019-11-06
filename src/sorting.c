#include "sorting.h"
/*============================================================================
sorting.c

Sorting methods.

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
#include <stdlib.h>
#include <string.h>

#ifdef CVTX_USING_OPENMP
#	include <omp.h>
#endif

void sort_uintkey_by_uivar_radix(
	unsigned char* ui_start, size_t uibytes,
	unsigned int* key_start, size_t num_items) {

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
	buffer = malloc(num_items * uibytes);
	counts = malloc(info_size * nthreads);
	offsets = malloc(info_size * nthreads);

	for (byte = 0; byte < uibytes; ++byte) {
		memset(counts, 0, info_size * nthreads);
		memset(offsets, 0, info_size * nthreads);
		wa = swap ? buffer : key_start;
		oa = swap ? key_start : buffer;
		int i = 0, j = 0, m = 0, n = 0;

		/* Counting pass */
#pragma omp parallel for private(m, n, j)
		for (i = 0; i < (int)nthreads; ++i) {
#ifdef CVTX_USING_OPENMP
			unsigned int threadid = omp_get_thread_num();
#else
			unsigned int threadid = 0;
#endif
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
		for (i = 0; i < (int)nthreads; ++i) {
#ifdef CVTX_USING_OPENMP
			unsigned int threadid = omp_get_thread_num();
#else
			unsigned int threadid = 0;
#endif
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
		memcpy(key_start, buffer, num_items);
	}
	free(buffer);
	free(counts);
	free(offsets);
#ifdef CVTX_USING_OPENMP
	omp_set_dynamic(1);
#endif
	return;
}
