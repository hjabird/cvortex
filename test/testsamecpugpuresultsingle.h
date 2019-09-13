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
	int repeat, max_repeats=10;	/* Random input can change the results... */
	int good;
	float reg_rad = 0.3f; /* Big enough we should have some overlapping particles */

	bsv_V3f *mes_pts3d, *res3d, *res3d2;
	bsv_V2f *mes_pts2d, *res2d, *res2d2;
	cvtx_P3D *p3ds, **p3ds_ptrs, *targ_p3ds, **targ_p3ds_ptrs;
	cvtx_P2D *p2ds, **p2ds_ptrs, *targ_p2ds, **targ_p2ds_ptrs;
	cvtx_F3D *fils, **fils_ptrs;
	cvtx_VortFunc func;
	float *fres, *fres2, maxerr, aveerr, mtmpp, mtmpm;
	double tmpp, tmpm;
	p3ds = malloc(sizeof(cvtx_P3D) * num_obj);
	p3ds_ptrs = malloc(sizeof(cvtx_P3D*) * num_obj);
	targ_p3ds = malloc(sizeof(cvtx_P3D) * num_obj);
	targ_p3ds_ptrs = malloc(sizeof(cvtx_P3D*) * num_obj);
	mes_pts3d = malloc(sizeof(bsv_V3f) * num_obj);
	res3d = malloc(sizeof(bsv_V3f) * num_obj);
	res3d2 = malloc(sizeof(bsv_V3f) * num_obj);
	res2d = malloc(sizeof(bsv_V2f) * num_obj);
	res2d2 = malloc(sizeof(bsv_V2f) * num_obj);
	mes_pts2d = malloc(sizeof(bsv_V2f) * num_obj);
	p2ds = malloc(sizeof(cvtx_P2D) * num_obj);
	p2ds_ptrs = malloc(sizeof(cvtx_P2D*) * num_obj); 
	targ_p2ds = malloc(sizeof(cvtx_P2D) * num_obj);
	targ_p2ds_ptrs = malloc(sizeof(cvtx_P2D*) * num_obj);
	fils = malloc(sizeof(cvtx_F3D) * num_obj);
	fils_ptrs = malloc(sizeof(cvtx_F3D*) * num_obj);
	fres = malloc(sizeof(float) * num_obj);
	fres2 = malloc(sizeof(float) * num_obj); 

	for (repeat = 0; repeat < max_repeats; ++repeat) {
		/* 3D PROBLEMS!!!!! */
		p3ds[0].coord.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		p3ds[0].coord.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		p3ds[0].coord.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
		mes_pts3d[0] = p3ds[0].coord;
		p3ds[0].vorticity.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		p3ds[0].vorticity.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		p3ds[0].vorticity.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
		p3ds[0].volume = (float)rand() / (float)(RAND_MAX / 0.01);
		p3ds_ptrs[0] = &(p3ds[0]);
		targ_p3ds[0] = p3ds[0];
		targ_p3ds_ptrs[0] = &(targ_p3ds[0]);
		for (i = 1; i < num_obj; ++i) {
			p3ds[i].coord.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			p3ds[i].coord.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			p3ds[i].coord.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
			mes_pts3d[i] = p3ds[i].coord;
			p3ds[i].vorticity.x[0] = 0.f;
			p3ds[i].vorticity.x[1] = 0.f;
			p3ds[i].vorticity.x[2] = 0.f;
			p3ds[i].volume = (float)rand() / (float)(RAND_MAX / 0.01);
			p3ds_ptrs[i] = &(p3ds[i]);
			targ_p3ds[i] = p3ds[i];
			targ_p3ds[i].vorticity.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			targ_p3ds[i].vorticity.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			targ_p3ds[i].vorticity.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
			targ_p3ds_ptrs[i] = &(targ_p3ds[i]);
		}
		fils[0].start.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].start.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].start.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].end.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].end.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].end.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
		fils[0].strength = (float)rand() / (float)(RAND_MAX / max_float);
		fils_ptrs[0] = &(fils[0]);
		for (i = 1; i < num_obj; ++i) {
			fils[i].start.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].start.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].start.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].end.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].end.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].end.x[2] = (float)rand() / (float)(RAND_MAX / max_float);
			fils[i].strength = 0.f;;
			fils_ptrs[i] = &(fils[i]);
		}
		if (cvtx_num_accelerators() > 0) {
			/* Singular */
			func = cvtx_VortFunc_singular();
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(p3ds_ptrs, num_obj, mes_pts3d, num_obj, res3d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(p3ds_ptrs, num_obj, mes_pts3d, num_obj, res3d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M vel singular (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M dvort singular (single object with vorticity)");
			/* (No viscous method) */

			/* Planetary */
			func = cvtx_VortFunc_planetary();
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(p3ds_ptrs, num_obj, mes_pts3d, num_obj, res3d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(p3ds_ptrs, num_obj, mes_pts3d, num_obj, res3d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M vel planetary (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M dvort planetary (single object with vorticity)");
			/* No viscous method. */

			/* Gaussian */
			func = cvtx_VortFunc_gaussian();
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(p3ds_ptrs, num_obj, mes_pts3d, num_obj, res3d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(p3ds_ptrs, num_obj, mes_pts3d, num_obj, res3d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M vel guassian (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M dvort gaussian (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_visc_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_visc_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d2, &func, reg_rad, 0.1f);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M visc dvort gaussian (single object with vorticity)");

			/* Winckelmans */
			func = cvtx_VortFunc_winckelmans();
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(p3ds_ptrs, num_obj, mes_pts3d, num_obj, res3d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(p3ds_ptrs, num_obj, mes_pts3d, num_obj, res3d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M vel winckelmans (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M dvort winckelmans (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_visc_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_visc_dvort(p3ds_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d2, &func, reg_rad, 0.1f);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P3D M2M visc dvort winckelmans (single object with vorticity)");

			/* Vortex filaments */
			cvtx_accelerator_enable(0);
			cvtx_F3D_M2M_vel(fils_ptrs, num_obj, mes_pts3d, num_obj, res3d);
			cvtx_accelerator_disable(0);
			cvtx_F3D_M2M_vel(fils_ptrs, num_obj, mes_pts3d, num_obj, res3d2);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					if (fabsf(tmpm / tmpp) > maxerr) {
						maxerr = fabsf(tmpm / tmpp);
						mtmpm = tmpm; mtmpp = tmpp;
					}
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "F3D M2M vel (single object with vorticity)");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e, Max Sum = %.2e, Max Diff = %.2e\n", aveerr / num_obj, maxerr, mtmpp, mtmpm); }
			cvtx_accelerator_enable(0);
			cvtx_F3D_M2M_dvort(fils_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d);
			cvtx_accelerator_disable(0);
			cvtx_F3D_M2M_dvort(fils_ptrs, num_obj, targ_p3ds_ptrs, num_obj, res3d2);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabsf(tmpm / tmpp);
					if (fabsf(tmpm / tmpp) > maxerr) {
						maxerr = fabsf(tmpm / tmpp);
						mtmpm = tmpm; mtmpp = tmpp;
					}
					if (tmpm / tmpp > rel_acc) {
						good = 0;
					}
				}
			}
			NAMED_TEST(good, "F3D M2M dvort (single object with vorticity)");
			if (!good) { printf("\tAve Err = %.2e Max Err = %.2e, Max Sum = %.2e, Max Diff = %.2e\n", aveerr / num_obj, maxerr, mtmpp, mtmpm); }
		}


		/* 2D PROBLEMS!!!!! */
		p2ds[0].coord.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
		p2ds[0].coord.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
		p2ds[0].vorticity = (float)rand() / (float)(RAND_MAX / max_float);
		p2ds[0].area = (float)rand() / (float)(RAND_MAX / 0.01);
		p2ds_ptrs[0] = &(p2ds[0]);
		targ_p2ds[0] = p2ds[0];
		targ_p2ds_ptrs[0] = &(targ_p2ds[0]);
		for (i = 1; i < num_obj; ++i) {
			p2ds[i].coord.x[0] = (float)rand() / (float)(RAND_MAX / max_float);
			p2ds[i].coord.x[1] = (float)rand() / (float)(RAND_MAX / max_float);
			p2ds[i].vorticity = 0.f;
			p2ds[i].area = (float)rand() / (float)(RAND_MAX / 0.01);
			p2ds_ptrs[i] = &(p2ds[i]);
			targ_p2ds[i] = p2ds[i];
			targ_p2ds[i].vorticity = (float)rand() / (float)(RAND_MAX / max_float);
			targ_p2ds_ptrs[i] = &(targ_p2ds[i]);
		}
		if (cvtx_num_accelerators() > 0) {
			/* Singular */
			func = cvtx_VortFunc_singular();
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(p2ds_ptrs, num_obj, mes_pts2d, num_obj, res2d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(p2ds_ptrs, num_obj, mes_pts2d, num_obj, res2d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(res2d[i], res2d2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(res2d[i], res2d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M vel singular (single object with vorticity)");

			/* Planetary */
			func = cvtx_VortFunc_planetary();
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(p2ds_ptrs, num_obj, mes_pts2d, num_obj, res2d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(p2ds_ptrs, num_obj, mes_pts2d, num_obj, res2d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(res2d[i], res2d2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(res2d[i], res2d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M vel planetary (single object with vorticity)");

			/* Gaussian */
			func = cvtx_VortFunc_gaussian();
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(p2ds_ptrs, num_obj, mes_pts2d, num_obj, res2d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(p2ds_ptrs, num_obj, mes_pts2d, num_obj, res2d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(res2d[i], res2d2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(res2d[i], res2d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M vel gaussian (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_visc_dvort(p2ds_ptrs, num_obj, targ_p2ds_ptrs, num_obj, fres, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_visc_dvort(p2ds_ptrs, num_obj, targ_p2ds_ptrs, num_obj, fres2, &func, reg_rad, 0.1f);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = fabsf(fres[i] - fres2[i]);
				tmpp = fabsf(fres[i] + fres2[i]);
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M visc dvort gaussian (single object with vorticity)");

			/* Winckelmans */
			func = cvtx_VortFunc_winckelmans();
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(p2ds_ptrs, num_obj, mes_pts2d, num_obj, res2d, &func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(p2ds_ptrs, num_obj, mes_pts2d, num_obj, res2d2, &func, reg_rad);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V2f_abs(bsv_V2f_minus(res2d[i], res2d2[i]));
				tmpp = bsv_V2f_abs(bsv_V2f_plus(res2d[i], res2d2[i]));
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M vel winckelmans (single object with vorticity)");
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_visc_dvort(p2ds_ptrs, num_obj, targ_p2ds_ptrs, num_obj, fres, &func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_visc_dvort(p2ds_ptrs, num_obj, targ_p2ds_ptrs, num_obj, fres2, &func, reg_rad, 0.1f);
			good = 1;
			for (i = 0; i < num_obj; ++i) {
				tmpm = fabsf(fres[i] - fres2[i]);
				tmpp = fabsf(fres[i] + fres2[i]);
				if (tmpp > 2e-35f && tmpm / tmpp > rel_acc) {
					good = 0;
					break;
				}
			}
			NAMED_TEST(good, "P2D M2M visc dvort winckelmans (single object with vorticity)");
		}
	}
	free(p3ds);
	free(p3ds_ptrs);
	free(mes_pts3d);
	free(res3d);
	free(res3d2);
	free(res2d);
	free(res2d2);
	free(p2ds);
	free(p2ds_ptrs); 
	free(mes_pts2d);
	free(fils);
	free(fils_ptrs);
	free(fres);
	free(fres2);
	free(targ_p3ds);
	free(targ_p3ds_ptrs); 
	free(targ_p2ds);
	free(targ_p2ds_ptrs);
	return 0;
}

#endif /* CVTX_TEST_SAMECPUGPURESSINGLE_H */
