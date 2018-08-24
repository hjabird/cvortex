#ifndef CVTX_PARTICLE_KERNEL_FUNCTIONS_H
#define CVTX_PARTICLE_KERNEL_FUNCTIONS_H
/*============================================================================
ParticleKernelFunctions.h

Common functions used to regularise vortex particles.

Copyright(c) 2018 HJA Bird

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

typedef struct {
	float(*reduction_factor_fn)(float rho);
	float(*vorticity_fraction_fn)(float rho);
	void(*combined_fn)(float rho, float* reduction, float* vort_frac);
} cvtx_ParticleKernalFunctions;

const cvtx_ParticleKernalFunctions cvtx_ParticleKernalFunctions_singular(void);
const cvtx_ParticleKernalFunctions cvtx_ParticleKernalFunctions_winckelmans(void);

#endif /* CVTX_PARTICLE_KERNEL_FUNCTIONS_H */
