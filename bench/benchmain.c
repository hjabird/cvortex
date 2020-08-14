/*============================================================================
benchmain.c

A dodgy self contained benchmarking system for cvortex.

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
#include <string.h>
#include <time.h>

#include "libcvtx.h"
#include "benchtools.h"

#include "benchinitilisation.h"
#include "benchredistribution.h"
#include "benchP3D.h"
#include "benchP2D.h"

int main(int argc, char* argv[]){
	if (!parse_command_args(argc, argv)) {
		return 0;
	};

	BENCH("init cold", bench_first_initialisation, 1, 1);
	cvtx_initialise();	/* If already run cold-init this does nothing. */
	BENCH("init reinit", bench_reinitialisation, 6, 1);
	run_redistribution_tests();
	run_P3D_bench();
	run_P2D_bench();

	cvtx_finalise();
	return 0;
}

