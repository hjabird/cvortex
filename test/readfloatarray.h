#ifndef CVTX_TEST_READFLOATARRAY_H
#define CVTX_TEST_READFLOATARRAY_H
/*============================================================================
readfloatarray.h

Read an array of floats in from a file for the purpose of testing.

FILE FORMAT:
[Line 1] ftype = 0 or 1 as unsigned char: 0 if ascii, 1 if binary
ASCII:(ftype == 0)
	[Line 2] total number of floats in data as ascii = N
	[Line 3] float 1
	[Line 4] float 2
	...
	[Line N+2] float N
	EOF.
BINARY:(ftype == 1)
	FLOAT32 num_floats in data following = N
	N * FLOAT32 data
	EOF

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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int readfloatarray(char* filename, float* array, int max_floats){
	assert(max_floats > 0);
	assert(array != NULL);
	FILE* f = fopen(filename, "rb");
	unsigned char ftype;
	int status, i;
	int nfloats;
	float nfloatsf, ndimsf;

	if(f == NULL){
		printf("Could not open test reference %s.\n", filename);
		assert(0);
		return -1;
	}

	/* Establish file type */
	ftype = getc(f);
	if(ftype == '0'){
		ftype = 0;
	}
	if(ftype == '1'){
		ftype = 1;
	}
	if(!(ftype == 1) && !(ftype == 0)){
		printf("Badly formatted test reference file %s. "
			"Incorrect ASCII/BINARY flag.\n", filename);
		return -1;
	}
	/* Now try to process. */
	if(ftype == 0)
	{	/* The ASCII version */
		if(!fscanf(f, "%f", &nfloatsf)){
			nfloatsf = -1.4f;
		}
		nfloats = (int)nfloatsf;
		if(	(nfloats < 0 ) ||
			(nfloats - nfloatsf != 0.f) ||
			(nfloats > max_floats)){
			printf("Badly formatted test reference file %s. "
				"Incorrect number of floats value.\n", filename);
			return -1;
		}

		for(i = 0; i < nfloats; ++i){
			if(feof(f)){
				printf("Badly formatted test reference file %s. "
				"Incorrect number of floats.\n", filename);
				return -1;
			}
			if(!fscanf(f, "%f", array+i)){
				printf("Badly formatted test reference file %s. "
				"Bad float number %i.\n", filename, i);
				return -1;
			};
		}
	}
	else
	{	/* The BINARY version */
		if(fread(&nfloatsf, sizeof(float), 1, f) != sizeof(float)){
			nfloatsf = -1.4f;
		}
		nfloats = (int)nfloatsf;
		if((nfloats < 0 ) ||
			(nfloats - nfloatsf != 0.f) ||
			(nfloats > max_floats)){
			printf("Badly formatted test reference file %s. "
				"Incorrect number of floats value.\n", filename);
			return -1;
		}

		status = (int)fread(array, sizeof(float), nfloats, f);		
		if(status != nfloats * sizeof(float)){
			printf("Badly formatted test reference file %s. "
				"Incorrect number of floats value.\n", filename);
			return -1;
		}			
	}
	/* DONE! */
	fclose(f);
	return nfloats;
}

#endif /* CVTX_TEST_READFLOATARRAY_H */

