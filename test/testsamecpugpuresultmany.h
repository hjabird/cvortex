#ifndef CVTX_TEST_SAMECPUGPURESMANY_H
#define CVTX_TEST_SAMECPUGPURESMANY_H

/*============================================================================
testsamecpugpuresultmany.h

Test that the GPU and the CPU obtain the same solutions for lots of particles
to lots of particles, where all particles have vorticity.

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

int testSameCpuGpuResMany() {
	SECTION("Same CPU/GPU Result - Many particles with vorticity");
	const int num_obj = 1000; /* Hopefully enough for the GPU to kick in. */
	float max_float = 10;
	float rel_acc = 1e-5f;
	int i;
	int repeat, max_repeats=10;	/* Random input can change the results... */
	int good;
	float reg_rad = 0.3f; /* Big enough we should have some overlapping particles */

	bsv_V3f *pmes, *presult, *presult2;
	bsv_V2f *p2mes, *p2dres, *p2dres2;
	cvtx_P3D *particles, **pparticles;
	cvtx_P2D *p2ds, **pp2ds;
	cvtx_F3D *fils, **pfils;
	cvtx_VortFunc func;
	float tmpp, tmpm, *fres, *fres2, maxerr, aveerr;
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
		for (i = 0; i < num_obj; ++i) {
			particles[i].coord.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			particles[i].coord.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			particles[i].coord.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
			pmes[i] = particles[i].coord;
			particles[i].vorticity.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			particles[i].vorticity.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			particles[i].vorticity.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
			particles[i].volume = (float)rand() / (float)(RAND_MAX / 0.01);
			pparticles[i] = &(particles[i]);
		}
		for (i = 0; i < num_obj; ++i) {
			fils[i].start.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].start.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].start.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].end.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].end.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].end.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].strength = (float)rand() / (float)(RAND_MAX / max_float);
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
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M vel singular");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
			good = 1; 
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P3D M2M dvort singular");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }
			/* (No viscous method) */

			/* Planetary */
			func = cvtx_VortFunc_planetary();
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult2, &func, reg_rad);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P3D M2M vel planetary");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
			good = 1; 
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P3D M2M dvort planetary");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }
			/* No viscous method. */

			/* Gaussian */
			func = cvtx_VortFunc_gaussian();
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult2, &func, reg_rad);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P3D M2M vel gaussian");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P3D M2M dvort gaussian");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad, 0.1f);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P3D M2M visc dvort gaussian");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }

			/* Winckelmans */
			func = cvtx_VortFunc_winckelmans();
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(pparticles, num_obj, pmes, num_obj, presult2, &func, reg_rad);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P3D M2M vel winckelmans");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P3D M2M dvort winckelmans");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_visc_dvort(pparticles, num_obj, pparticles, num_obj, presult2, &func, reg_rad, 0.1f);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P3D M2M visc dvort winckelmans");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }

			/* Vortex filaments */
			cvtx_accelerator_enable(0);
			cvtx_F3D_M2M_vel(pfils, num_obj, pmes, num_obj, presult);
			cvtx_accelerator_disable(0);
			cvtx_F3D_M2M_vel(pfils, num_obj, pmes, num_obj, presult2);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "F3D M2M vel");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }
			cvtx_accelerator_enable(0);
			cvtx_F3D_M2M_dvort(pfils, num_obj, pparticles, num_obj, presult);
			cvtx_accelerator_disable(0);
			cvtx_F3D_M2M_dvort(pfils, num_obj, pparticles, num_obj, presult2);
			good = 1; 
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(presult[i], presult2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(presult[i], presult2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "F3D M2M dvort");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }
		}


		/* 2D PROBLEMS!!!!! */
		for (i = 0; i < num_obj; ++i) {
			p2ds[i].coord.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			p2ds[i].coord.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			p2ds[i].vorticity = (float)rand() / (float)(RAND_MAX / max_float);
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
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(p2dres[i], p2dres2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(p2dres[i], p2dres2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P2D M2M vel singular");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }

			/* Planetary */
			func = cvtx_VortFunc_planetary();
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres2, &func, reg_rad);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(p2dres[i], p2dres2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(p2dres[i], p2dres2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P2D M2M vel planetary");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }

			/* Gaussian */
			func = cvtx_VortFunc_gaussian();
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres2, &func, reg_rad);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(p2dres[i], p2dres2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(p2dres[i], p2dres2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P2D M2M vel gaussian"); 
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }

			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_visc_dvort(pp2ds, num_obj, pp2ds, num_obj, fres, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_visc_dvort(pp2ds, num_obj, pp2ds, num_obj, fres2, &func, reg_rad, 0.1f);
			good = 1;
			maxerr = aveerr = 0.f;
			for (i = 0; i < num_obj; ++i) {
				tmpm = fabsf(fres[i] - fres2[i]);
				tmpp = fabsf(fres[i] + fres2[i]);
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P2D M2M visc dvort gaussian");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr / num_obj, maxerr); }

			/* Winckelmans */
			func = cvtx_VortFunc_winckelmans();
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(pp2ds, num_obj, p2mes, num_obj, p2dres2, &func, reg_rad);
			good = 1;
			maxerr = aveerr = 0.f;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(p2dres[i], p2dres2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(p2dres[i], p2dres2[i]));
				if (tmpp > 2e-35f){
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P2D M2M vel winckelmans");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr/num_obj, maxerr); }
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_visc_dvort(pp2ds, num_obj, pp2ds, num_obj, fres, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_visc_dvort(pp2ds, num_obj, pp2ds, num_obj, fres2, &func, reg_rad, 0.1f);
			good = 1;
			maxerr = aveerr = 0.f;
			for (i = 0; i < num_obj; ++i) {
				tmpm = fabsf(fres[i] - fres2[i]); 
				tmpp = fabsf(fres[i] + fres2[i]);
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					maxerr = fabsf(tmpm / tmpp) > maxerr ? fabsf(tmpm / tmpp) : maxerr;
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "P2D M2M visc dvort winckelmans");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e\n", aveerr/num_obj, maxerr); }
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

#endif /* CVTX_TEST_SAMECPUGPURESMANY_H */
