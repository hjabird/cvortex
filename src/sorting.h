#ifndef CVTX_SORTING_H
#define CVTX_SORTING_H
/*============================================================================
sorting.h

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

#include <stdlib.h>

#ifdef _WIN32
#	define qsort_r qsort_s
#elif defined(linux)
#	define qsort_r qsort_r
#endif

/*
"Sort" an array of indices by the variables they refer to.
START:	UI		= [3, 2, 6, 4]
		KEYS	= [0, 1, 2, 3]
END:	UI		= [3, 2, 6, 4]
		KEYS	= [1, 0, 3, 2]
UI is assumed to be array of unsigned integers of
uibytes size.
Uses radix sorting method.

ui_start is an array of uibytes * num_items bytes long.
key_start is an array of unsigned ints num_items long.
*/
void sort_uintkey_by_uivar_radix(
	unsigned char* ui_start, size_t uibytes,
	unsigned int* key_start, size_t num_items);

/* Parallel variant of sort_uintkey_by_uivar_radix(..) */
void sort_uintkey_by_uivar_radix_p(
	unsigned char* ui_start, size_t uibytes,
	unsigned int* key_start, size_t num_items);

#endif
