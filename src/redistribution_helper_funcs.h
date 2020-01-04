#ifndef CVTX_REDISTRIBUTION_HELPER_FUNCS_H
#define CVTX_REDISTRIBUTION_HELPER_FUNCS_H
#include "libcvtx.h"
/*============================================================================
redistribution_helper_funcs.h

Functions that are common to redistributing vortex particles in both 2D 
and 3D.

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

/* Get the min, max and mean of an array of floats. NULL
args for min/max/mean means that they aren't calculated. */
void farray_info(
	float* strs, int n_inpt_partices,
	float* mean, float* min, float* max);

/* Compute the approximate threshold to for removal of vortex particles. 
Strs gives abs strengths of vortex particles. We want to remove the weakest 
of the vortex particles. We want to have n_desired_particles or fewer remaining.
This will try to do that within 1% of n_desired_particles(always fewer). */
float get_strength_threshold(float* strs,
	int n_inpt_particles, int n_desired_particles);

#endif /*CVTX_REDISTRIBUTION_HELPER_FUNCS_H*/
