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
	cvtx_P3D *p3ds, *targ_p3ds;
	cvtx_P2D *p2ds, *targ_p2ds;
	cvtx_F3D *fils;
	cvtx_VortFunc func;
	float *fres, *fres2, maxerr, aveerr, mtmpp, mtmpm;
	double tmpp, tmpm;
	p3ds = malloc(sizeof(cvtx_P3D) * num_obj);
	targ_p3ds = malloc(sizeof(cvtx_P3D) * num_obj);
	mes_pts3d = malloc(sizeof(bsv_V3f) * num_obj);
	res3d = malloc(sizeof(bsv_V3f) * num_obj);
	res3d2 = malloc(sizeof(bsv_V3f) * num_obj);
	res2d = malloc(sizeof(bsv_V2f) * num_obj);
	res2d2 = malloc(sizeof(bsv_V2f) * num_obj);
	mes_pts2d = malloc(sizeof(bsv_V2f) * num_obj);
	p2ds = malloc(sizeof(cvtx_P2D) * num_obj);
	targ_p2ds = malloc(sizeof(cvtx_P2D) * num_obj);
	fils = malloc(sizeof(cvtx_F3D) * num_obj);
	fres = malloc(sizeof(float) * num_obj);
	fres2 = malloc(sizeof(float) * num_obj); 

	for (repeat = 0; repeat < max_repeats; ++repeat) {
		/* 3D PROBLEMS!!!!! */
		p3ds[0].coord.x[0] = (float)mrand() / ((float)RAND_MAX / max_float);
		p3ds[0].coord.x[1] = (float)mrand() / ((float)RAND_MAX / max_float);
		p3ds[0].coord.x[2] = (float)mrand() / ((float)RAND_MAX / max_float);
		mes_pts3d[0] = p3ds[0].coord;
		p3ds[0].vorticity.x[0] = (float)mrand() / ((float)RAND_MAX / max_float) - max_float / 2.f;
		p3ds[0].vorticity.x[1] = (float)mrand() / ((float)RAND_MAX / max_float) - max_float / 2.f;
		p3ds[0].vorticity.x[2] = (float)mrand() / ((float)RAND_MAX / max_float) - max_float / 2.f;
		p3ds[0].volume = (float)mrand() / ((float)RAND_MAX / 0.01);
		for (i = 1; i < num_obj; ++i) {
			p3ds[i].coord.x[0] = (float)mrand() / ((float)RAND_MAX / max_float);
			p3ds[i].coord.x[1] = (float)mrand() / ((float)RAND_MAX / max_float);
			p3ds[i].coord.x[2] = (float)mrand() / ((float)RAND_MAX / max_float);
			mes_pts3d[i] = p3ds[i].coord;
			p3ds[i].vorticity.x[0] = 0.f;
			p3ds[i].vorticity.x[1] = 0.f;
			p3ds[i].vorticity.x[2] = 0.f;
			p3ds[i].volume = (float)mrand() / ((float)RAND_MAX / 0.01);
			targ_p3ds[i] = p3ds[i];
			targ_p3ds[i].vorticity.x[0] = (float)mrand() / ((float)RAND_MAX / max_float) - max_float / 2.f;
			targ_p3ds[i].vorticity.x[1] = (float)mrand() / ((float)RAND_MAX / max_float) - max_float / 2.f;
			targ_p3ds[i].vorticity.x[2] = (float)mrand() / ((float)RAND_MAX / max_float) - max_float / 2.f;
		}
		fils[0].start.x[0] = (float)mrand() / ((float)RAND_MAX / max_float);
		fils[0].start.x[1] = (float)mrand() / ((float)RAND_MAX / max_float);
		fils[0].start.x[2] = (float)mrand() / ((float)RAND_MAX / max_float);
		fils[0].end.x[0] = (float)mrand() / ((float)RAND_MAX / max_float);
		fils[0].end.x[1] = (float)mrand() / ((float)RAND_MAX / max_float);
		fils[0].end.x[2] = (float)mrand() / ((float)RAND_MAX / max_float);
		fils[0].strength = (float)mrand() / ((float)RAND_MAX / max_float) - max_float / 2.f;
		for (i = 1; i < num_obj; ++i) {
			fils[i].start.x[0] = (float)mrand() / ((float)RAND_MAX / max_float);
			fils[i].start.x[1] = (float)mrand() / ((float)RAND_MAX / max_float);
			fils[i].start.x[2] = (float)mrand() / ((float)RAND_MAX / max_float);
			fils[i].end.x[0] = (float)mrand() / ((float)RAND_MAX / max_float);
			fils[i].end.x[1] = (float)mrand() / ((float)RAND_MAX / max_float);
			fils[i].end.x[2] = (float)mrand() / ((float)RAND_MAX / max_float);
			fils[i].end = bsv_V3f_plus(fils[i].end, bsv_V3f_mult(fils[i].start, 0.01));
			fils[i].strength = 0.f;
		}
		if (cvtx_num_accelerators() > 0) {
			/* Singular */
			func = cvtx_VortFunc_singular;
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(p3ds, num_obj, mes_pts3d, num_obj, res3d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(p3ds, num_obj, mes_pts3d, num_obj, res3d2, func, reg_rad);
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
			cvtx_P3D_M2M_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d2, func, reg_rad);
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
			func = cvtx_VortFunc_planetary;
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(p3ds, num_obj, mes_pts3d, num_obj, res3d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(p3ds, num_obj, mes_pts3d, num_obj, res3d2, func, reg_rad);
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
			cvtx_P3D_M2M_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d2, func, reg_rad);
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
			func = cvtx_VortFunc_gaussian;
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(p3ds, num_obj, mes_pts3d, num_obj, res3d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(p3ds, num_obj, mes_pts3d, num_obj, res3d2, func, reg_rad);
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
			cvtx_P3D_M2M_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d2, func, reg_rad);
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
			cvtx_P3D_M2M_visc_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d, func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_visc_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d2, func, reg_rad, 0.1f);
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
			func = cvtx_VortFunc_winckelmans;
			cvtx_accelerator_enable(0);
			cvtx_P3D_M2M_vel(p3ds, num_obj, mes_pts3d, num_obj, res3d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_vel(p3ds, num_obj, mes_pts3d, num_obj, res3d2, func, reg_rad);
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
			cvtx_P3D_M2M_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d2, func, reg_rad);
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
			cvtx_P3D_M2M_visc_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d, func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P3D_M2M_visc_dvort(p3ds, num_obj, targ_p3ds, num_obj, res3d2, func, reg_rad, 0.1f);
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
			cvtx_F3D_M2M_vel(fils, num_obj, mes_pts3d, num_obj, res3d);
			cvtx_accelerator_disable(0);
			cvtx_F3D_M2M_vel(fils, num_obj, mes_pts3d, num_obj, res3d2);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabs(tmpm / tmpp);
					if (fabs(tmpm / tmpp) > maxerr) {
						maxerr = fabs(tmpm / tmpp);
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
			cvtx_F3D_M2M_dvort(fils, num_obj, targ_p3ds, num_obj, res3d);
			cvtx_accelerator_disable(0);
			cvtx_F3D_M2M_dvort(fils, num_obj, targ_p3ds, num_obj, res3d2);
			good = 1;
			maxerr = aveerr = 0.;
			for (i = 0; i < num_obj; ++i) {
				tmpm = bsv_V3f_abs(bsv_V3f_minus(res3d[i], res3d2[i]));
				tmpp = bsv_V3f_abs(bsv_V3f_plus(res3d[i], res3d2[i]));
				if (tmpp > 2e-35f) {
					aveerr += fabs(tmpm / tmpp);
					if (fabs(tmpm / tmpp) > maxerr) {
						maxerr = fabs(tmpm / tmpp);
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
		p2ds[0].coord.x[0] = (float)mrand() / ((float)RAND_MAX / max_float);
		p2ds[0].coord.x[1] = (float)mrand() / ((float)RAND_MAX / max_float);
		p2ds[0].vorticity = (float)mrand() / ((float)RAND_MAX / max_float) - max_float / 2.f;
		p2ds[0].area = (float)mrand() / ((float)RAND_MAX / 0.01);
		targ_p2ds[0] = p2ds[0];
		for (i = 1; i < num_obj; ++i) {
			p2ds[i].coord.x[0] = (float)mrand() / ((float)RAND_MAX / max_float);
			p2ds[i].coord.x[1] = (float)mrand() / ((float)RAND_MAX / max_float);
			p2ds[i].vorticity = 0.f;
			p2ds[i].area = (float)mrand() / ((float)RAND_MAX / 0.01);
			targ_p2ds[i] = p2ds[i];
			targ_p2ds[i].vorticity = (float)mrand() / ((float)RAND_MAX / max_float);
		}
		if (cvtx_num_accelerators() > 0) {
			/* Singular */
			func = cvtx_VortFunc_singular;
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(p2ds, num_obj, mes_pts2d, num_obj, res2d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(p2ds, num_obj, mes_pts2d, num_obj, res2d2, func, reg_rad);
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
			func = cvtx_VortFunc_planetary;
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(p2ds, num_obj, mes_pts2d, num_obj, res2d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(p2ds, num_obj, mes_pts2d, num_obj, res2d2, func, reg_rad);
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
			func = cvtx_VortFunc_gaussian;
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(p2ds, num_obj, mes_pts2d, num_obj, res2d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(p2ds, num_obj, mes_pts2d, num_obj, res2d2, func, reg_rad);
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
			cvtx_P2D_M2M_visc_dvort(p2ds, num_obj, targ_p2ds, num_obj, fres, func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_visc_dvort(p2ds, num_obj, targ_p2ds, num_obj, fres2, func, reg_rad, 0.1f);
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
			func = cvtx_VortFunc_winckelmans;
			cvtx_accelerator_enable(0);
			cvtx_P2D_M2M_vel(p2ds, num_obj, mes_pts2d, num_obj, res2d, func, reg_rad);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_vel(p2ds, num_obj, mes_pts2d, num_obj, res2d2, func, reg_rad);
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
			cvtx_P2D_M2M_visc_dvort(p2ds, num_obj, targ_p2ds, num_obj, fres, func, reg_rad, 0.1f);
			cvtx_accelerator_disable(0);
			cvtx_P2D_M2M_visc_dvort(p2ds, num_obj, targ_p2ds, num_obj, fres2, func, reg_rad, 0.1f);
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
	free(mes_pts3d);
	free(res3d);
	free(res3d2);
	free(res2d);
	free(res2d2);
	free(p2ds);
	free(mes_pts2d);
	free(fils);
	free(fres);
	free(fres2);
	free(targ_p3ds);
	free(targ_p2ds);
	return 0;
}

#endif /* CVTX_TEST_SAMECPUGPURESSINGLE_H */
