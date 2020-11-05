#ifndef CVTX_ARRAY_METHODS_H
#define CVTX_ARRAY_METHODS_H
#include "libcvtx.h"
/*============================================================================
array_methods.h

Functions to work on arrays:
- A radix sort permutation for uints of 8*n bytes.

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
void sort_perm_multibyte_radix8(
	unsigned char* ui_start, size_t uibytes,
	unsigned int* key_start, size_t num_items);

void minmax_xyz_posn(
	const cvtx_P3D** array_start, const int nparticles,
	bsv_V3f *min, bsv_V3f *max);

bsv_V3f mean_xyz_posn(
	const cvtx_P3D** array_start, const int nparticles);

void minmax_xy_posn(
	const cvtx_P2D** array_start, const int nparticles,
	bsv_V2f* min, bsv_V2f* max);

bsv_V2f mean_xy_posn(
	const cvtx_P2D** array_start, const int nparticles);

#endif
