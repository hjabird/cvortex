#ifndef CVTX_TEST_SAMECPUGPURES_H
#define CVTX_TEST_SAMECPUGPURES_H

/*============================================================================
testsamecpugpuresult.h

Test that the GPU and the CPU obtain the same solutions.

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
#include "../include/cvortex/libcvtx.h"

#include <math.h>
#include <stdlib.h>

int testSameCpuGpuRes() {
	SECTION("SameCpuGpuRes");
	const int num_obj = 1000; /* Hopefully enough for the GPU to kick in. */
	float max_float = 10;
	int i;
	int good;
	float reg_rad = 0.3f; /* Big enough we should have some overlapping particles */

	bsv_V3f *pmes, *presult, *presult2;
	bsv_V2f *pvel;
	cvtx_P3D *particles, **pparticles;
	cvtx_VortFunc func;
	float tmpp, tmpm;
	particles = malloc(sizeof(cvtx_P3D) * num_obj);
	pparticles = malloc(sizeof(cvtx_P3D*) * num_obj);
	pmes = malloc(sizeof(bsv_V3f) * num_obj);
	presult = malloc(sizeof(bsv_V3f) * num_obj);
	presult2 = malloc(sizeof(bsv_V3f) * num_obj);

	for (i = 0; i < num_obj; ++i) {
		particles[i].coord.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		particles[i].coord.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		particles[i].coord.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
		particles[i].vorticity.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		particles[i].vorticity.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		particles[i].vorticity.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
		particles[i].volume = (float)rand() / (float)(RAND_MAX / 0.01);
		pparticles[i] = &(particles[i]);
	}

	if (cvtx_num_accelerators() > 0) {
		/* Singular */
		func = cvtx_VortFunc_singular();
		cvtx_accelerator_enable(0);
		cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult, &func, reg_rad);
		cvtx_accelerator_disable(0);
		cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult2, &func, reg_rad);
		good = 1;
		for (i = 0; i < num_obj; ++i) {
			tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
			tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
			if (tmpm / tmpp > 1e-5) {
				good = 0;
				break;
			}
		}
		TEST(good);
		cvtx_accelerator_enable(0);
		cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
		cvtx_accelerator_disable(0);
		cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
		good = 1;
		for (i = 0; i < num_obj; ++i) {
			tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
			tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
			if (tmpm/tmpp > 1e-5) {
				good = 0;
				break;
			}
		}
		TEST(good);
		/* (No viscous method) */

		/* Planetary */
		func = cvtx_VortFunc_planetary();
		cvtx_accelerator_enable(0);
		cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult, &func, reg_rad);
		cvtx_accelerator_disable(0);
		cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult2, &func, reg_rad);
		good = 1;
		for (i = 0; i < num_obj; ++i) {
			tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
			tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
			if (tmpm / tmpp > 1e-5) {
				good = 0;
				break;
			}
		}
		TEST(good);
		cvtx_accelerator_enable(0);
		cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
		cvtx_accelerator_disable(0);
		cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
		good = 1;
		for (i = 0; i < num_obj; ++i) {
			tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
			tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
			if (tmpm / tmpp > 1e-5) {
				good = 0;
				break;
			}
		}
		TEST(good);
		/* No viscous method. */

		/* Gaussian */
		func = cvtx_VortFunc_gaussian();
		cvtx_accelerator_enable(0);
		cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult, &func, reg_rad);
		cvtx_accelerator_disable(0);
		cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult2, &func, reg_rad);
		good = 1;
		for (i = 0; i < num_obj; ++i) {
			tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
			tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
			if (tmpm / tmpp > 1e-5) {
				good = 0;
				break;
			}
		}
		TEST(good);
		cvtx_accelerator_enable(0);
		cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
		cvtx_accelerator_disable(0);
		cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
		good = 1;
		for (i = 0; i < num_obj; ++i) {
			tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
			tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
			if (tmpm / tmpp > 1e-5) {
				good = 0;
				break;
			}
		}
		TEST(good);
		cvtx_accelerator_enable(0);
		cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad, 0.1f);
		cvtx_accelerator_disable(0);
		cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad, 0.1f);
		good = 1;
		for (i = 0; i < num_obj; ++i) {
			tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
			tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
			if (tmpm / tmpp > 1e-5) {
				good = 0;
				break;
			}
		}
		TEST(good);

		/* Winckelmans */
		func = cvtx_VortFunc_gaussian();
		cvtx_accelerator_enable(0);
		cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult, &func, reg_rad);
		cvtx_accelerator_disable(0);
		cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult2, &func, reg_rad);
		good = 1;
		for (i = 0; i < num_obj; ++i) {
			tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
			tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
			if (tmpm / tmpp > 1e-5) {
				good = 0;
				break;
			}
		}
		TEST(good);
		cvtx_accelerator_enable(0);
		cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
		cvtx_accelerator_disable(0);
		cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
		good = 1;
		for (i = 0; i < num_obj; ++i) {
			tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
			tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
			if (tmpm / tmpp > 1e-5) {
				good = 0;
				break;
			}
		}
		TEST(good);
		cvtx_accelerator_enable(0);
		cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad, 0.1f);
		cvtx_accelerator_disable(0);
		cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad, 0.1f);
		good = 1;
		for (i = 0; i < num_obj; ++i) {
			tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
			tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
			if (tmpm / tmpp > 1e-5) {
				good = 0;
				break;
			}
		}
		TEST(good);
	}
	free(particles);
	free(pparticles);
	free(pmes);
	free(presult);
	free(presult2);
	return 0;
}

#endif /* CVTX_TEST_SAMECPUGPURES_H */
