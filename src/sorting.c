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

#define CACHELINE_SIZE 64	/* Bytes */

void sort_uintkey_by_uivar_radix(
	unsigned char *ui_start, size_t uibytes,
	unsigned int *key_start, size_t num_items) {

	assert(uibytes > 0);
	assert(num_items >= 0);
	assert(ui_start != NULL);
	assert(key_start != NULL);

	/* Buffer (buffer) swaps with input array
	as working array (wa) or output array (oa) */
	unsigned int *buffer = NULL, *wa=NULL, *oa=NULL;
	/* Bit = 0/1 counts and offsets*/
	unsigned int bc0 = 0, bc1 = 0, bc0o = 0, bc1o = 0;
	/* swap = what are we writing the result of this iter into?
	True:	Work from buffer into key_start
	False:	Work from key_start into buffer
	*/
	unsigned int bit = 0, byte = 0, swap = 0, uini = (unsigned int)num_items;
	size_t i, j;
	buffer = malloc(num_items * uibytes);

	for (byte = 0; byte < uibytes; ++byte) {
		for (bit = 0; bit < 8; ++bit) {
			bc1 = bc0 = 0;
			wa = swap ? buffer : key_start;
			oa = swap ? key_start : buffer;
			unsigned char mask = 0x1 << bit;
			for (i = 0; i < uini; ++i) {
				j = wa[i] * uibytes + byte;
				ui_start[j] & mask ? ++bc1 : ++bc0;
			}
			assert(bc0 + bc1 == uini);
			bc1o = bc0o = 0;
			for (i = 0; i < uini; ++i) {
				j = wa[i] * uibytes + byte;
				if (ui_start[j] & mask) {
					oa[bc1o + bc0] = wa[i];
					++bc1o;
				}
				else {
					oa[bc0o] = wa[i];
					++bc0o;
				}
			}
			swap = swap ? 0 : 1;
		}
	}
	/* If we wrote our solution into the buffer, we need to copy
	it back. */
	if (!swap) {
		memcpy(key_start, buffer, num_items);
	}
	free(buffer);
	return;
}

void sort_uintkey_by_uivar_radix_p(
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
	unsigned int *counts, *offsets, info_size = sizeof(unsigned int) * n_para;
	/* swap = what are we writing the result of this iter into? */
	unsigned int bit = 0, byte = 0, swap = 0, uini = (unsigned int)num_items;
	buffer = malloc(num_items * uibytes);
	counts = malloc(info_size);
	offsets = malloc(info_size);

	for (byte = 0; byte < uibytes; ++byte) {
		memset(counts, 0, info_size);
		memset(offsets, 0, info_size);
		wa = swap ? buffer : key_start;
		oa = swap ? key_start : buffer;
		size_t i = 0, j = 0;
		for (i = 0; i < uini; ++i) {
			counts[ui_start[wa[i] * uibytes + byte]]++;
		}
		for (i = 0; i < n_para-1; ++i) {
			offsets[i + 1] = offsets[i] + counts[i];
			assert(offsets[i + 1] >= offsets[i]);
		}
		for (i = 0; i < uini; ++i) {
			j = ui_start[wa[i] * uibytes + byte];
			oa[offsets[j]] = wa[i];
			++offsets[j];
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
	return;
}
