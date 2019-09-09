#ifndef CVTX_TEST_SAMECPUGPURESSINGLE_H
#define CVTX_TEST_SAMECPUGPURESSINGLE_H

/*============================================================================
testsamecpugpuresultsingle.h

Test that the GPU and the CPU obtain the same solutions where we only have
1 object with actual vorticity (ie. adding stuff together shouldn'd do 
anything funny).

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

int testSameCpuGpuResSingle() {
	SECTION("Same CPU/GPU Result - only a single particle with any vorticity");
	const int num_obj = 1000; /* Hopefully enough for the GPU to kick in. */
	float max_float = 10;
	float rel_acc = 1e-5f;
	int i;
	int repeat, max_repeats=5;	/* Random input can change the results... */
	int good;
	float reg_rad = 0.3f; /* Big enough we should have some overlapping particles */

	bsv_V3f *pmes, *presult, *presult2;
	bsv_V2f *p2mes, *p2dres, *p2dres2;
	cvtx_P3D *particles, **pparticles;
	cvtx_P2D *p2ds, **pp2ds;
	cvtx_F3D *fils, **pfils;
	cvtx_VortFunc func;
	float tmpp, tmpm, *fres, *fres2;
	particles = malloc(sizeof(cvtx_P3D) * num_obj);
	pparticles = malloc(sizeof(cvtx_P3D*) * num_obj);
	pmes = malloc(sizeof(bsv_V3f) * num_obj);
	presult = malloc(sizeof(bsv_V3f) * num_obj);
	presult2 = malloc(sizeof(bsv_V3f) * num_obj);
	p2dres = malloc(sizeof(bsv_V2f) * num_obj);
	p2dres2 = malloc(sizeof(bsv_V2f) * num_obj);
	p2mes = malloc(sizeof(bsv_V2f) * num_obj);
	p2ds = malloc(sizeof(cvtx_P2D) * num_obj);
	pp2ds = malloc(sizeof(cvtx_P2D*) * num_obj);
	fils = malloc(sizeof(cvtx_F3D) * num_obj);
	pfils = malloc(sizeof(cvtx_F3D*) * num_obj);
	fres = malloc(sizeof(float) * num_obj);
	fres2 = malloc(sizeof(float) * num_obj);

	for (repeat = 0; repeat < max_repeats; ++repeat) {
		/* 3D PROBLEMS!!!!! */
		particles[0].coord.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		particles[0].coord.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		particles[0].coord.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
		pmes[0] = particles[0].coord;
		particles[0].vorticity.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		particles[0].vorticity.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		particles[0].vorticity.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
		particles[0].volume = (float)rand() / (float)(RAND_MAX / 0.01);
		pparticles[0] = &(particles[0]);
		for (i = 1; i < num_obj; ++i) {
			particles[i].coord.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			particles[i].coord.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			particles[i].coord.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
			pmes[i] = particles[i].coord;
			particles[i].vorticity.x[0] = 0.f;
			particles[i].vorticity.x[1] = 0.f;
			particles[i].vorticity.x[2] = 0.f;
			particles[i].volume = (float)rand() / (float)(RAND_MAX / 0.01);
			pparticles[i] = &(particles[i]);
		}
		fils[0].start.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].start.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].start.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].end.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].end.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].end.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].strength = (float)rand() / (float)(RAND_MAX / max_float);
		pfils[0] = &(fils[0]);
		for (i = 1; i < num_obj; ++i) {
			fils[i].start.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].start.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].start.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].end.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].end.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].end.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].strength = 0.f;;
			pfils[i] = &(fils[i]);
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
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M vel singular (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M dvort singular (single object with vorticity)");
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
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M vel planetary (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M dvort planetary (single object with vorticity)");
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
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M vel guassian (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M dvort gaussian (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad, 0.1f);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M visc dvort gaussian (single object with vorticity)");

			/* Winckelmans */
			func = cvtx_VortFunc_winckelmans();
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M vel winckelmans (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M dvort winckelmans (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad, 0.1f);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M visc dvort winckelmans (single object with vorticity)");

			/* Vortex filaments */
			cvtx_accelerator_enable(0);
			cvtx_F3D_M2M_vel(pfils, num_obj, pmes, num_obj, presult);
			cvtx_accelerator_disable(0);
			cvtx_F3D_M2M_vel(pfils, num_obj, pmes, num_obj, presult2);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "F3D M2M vel (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_F3D_M2M_dvort(pfils, num_obj, pparticles, num_obj, presult);
			cvtx_accelerator_disable(0);
			cvtx_F3D_M2M_dvort(pfils, num_obj, pparticles, num_obj, presult2);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "F3D M2M dvort (single object with vorticity)");
		}


		/* 2D PROBLEMS!!!!! */
		p2ds[0].coord.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		p2ds[0].coord.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		p2ds[0].vorticity = (float)rand() / (float)(RAND_MAX / max_float);
		p2ds[0].area = (float)rand() / (float)(RAND_MAX / 0.01);
		pp2ds[0] = &(p2ds[0]);
		for (i = 1; i < num_obj; ++i) {
			p2ds[i].coord.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			p2ds[i].coord.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			p2ds[i].vorticity = 0.f;
			p2ds[i].area = (float)rand() / (float)(RAND_MAX / 0.01);
			pp2ds[i] = &(p2ds[i]);
		}
		if (cvtx_num_accelerators() > 0) {
			/* Singular */
			func = cvtx_VortFunc_singular();
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(p2dres[i], p2dres2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(p2dres[i], p2dres2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M vel singular (single object with vorticity)");

			/* Planetary */
			func = cvtx_VortFunc_planetary();
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(p2dres[i], p2dres2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(p2dres[i], p2dres2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M vel planetary (single object with vorticity)");

			/* Gaussian */
			func = cvtx_VortFunc_gaussian();
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(p2dres[i], p2dres2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(p2dres[i], p2dres2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M vel gaussian (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_visc_dvort(pp2ds, num_obj, pp2ds, num_obj, fres, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_visc_dvort(pp2ds, num_obj, pp2ds, num_obj, fres2, &func, reg_rad, 0.1f);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = fabsf(fres[i] - fres2[i]);
				tmpp = fabsf(fres[i] + fres2[i]);
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M visc dvort gaussian (single object with vorticity)");

			/* Winckelmans */
			func = cvtx_VortFunc_winckelmans();
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(p2dres[i], p2dres2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(p2dres[i], p2dres2[i]));
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M vel winckelmans (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_visc_dvort(pp2ds, num_obj, pp2ds, num_obj, fres, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_visc_dvort(pp2ds, num_obj, pp2ds, num_obj, fres2, &func, reg_rad, 0.1f);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = fabsf(fres[i] - fres2[i]);
				tmpp = fabsf(fres[i] + fres2[i]);
				if (tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M visc dvort winckelmans (single object with vorticity)");
		}
	}
	free(particles);
	free(pparticles);
	free(pmes);
	free(presult);
	free(presult2);
	free(p2dres);
	free(p2dres2);
	free(p2ds);
	free(pp2ds); 
	free(p2mes);
	free(fils);
	free(pfils);
	free(fres);
	free(fres2);
	return 0;
}

#endif /* CVTX_TEST_SAMECPUGPURESSINGLE_H */
