#include "bencharraysetup.h"
/*============================================================================
bencharraysetup.h

Set up big random arrays for IO of benchmark functions.

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
#include <stdlib.h>
#include "benchtools.h"

static cvtx_P3D* particles_3D = NULL;
static cvtx_P3D* oparticles_3D = NULL;
static cvtx_P3D** pparticles_3D = NULL;
static cvtx_P2D* particles_2D = NULL;
static cvtx_P2D* oparticles_2D = NULL;
static cvtx_P2D** pparticles_2D = NULL;

static bsv_V3f* m_v3f_arr1 = NULL;
static bsv_V3f* m_v3f_arr2 = NULL;
static bsv_V2f* m_v2f_arr1 = NULL;
static bsv_V2f* m_v2f_arr2 = NULL;
static float * m_f_arr1 = NULL;

/* Setup / teardown. */
void create_particles_3D(int np, float maxf, float vol) {
	particles_3D = malloc(sizeof(cvtx_P3D) * np);
	pparticles_3D = malloc(sizeof(cvtx_P3D*) * np);
	int i;
	for (i = 0; i < np; ++i) {
		particles_3D[i].coord.x[0] = mrandf(maxf);
		particles_3D[i].coord.x[1] = mrandf(maxf);
		particles_3D[i].coord.x[2] = mrandf(maxf);
		particles_3D[i].vorticity.x[0] = mrandf(maxf);
		particles_3D[i].vorticity.x[1] = mrandf(maxf);
		particles_3D[i].vorticity.x[2] = mrandf(maxf);
		particles_3D[i].volume = vol;
		pparticles_3D[i] = &(particles_3D[i]);
	}
	return;
}

void create_particles_3D_outarr(int np) {
	oparticles_3D = malloc(sizeof(cvtx_P3D) * np);
}

void destroy_particles_3D() {
	free(particles_3D);
	free(pparticles_3D);
	return;
}

void destroy_oparticles_3D() {
	free(oparticles_3D);
}

void create_particles_2D(int np, float maxf, float area) {
	particles_2D = malloc(sizeof(cvtx_P2D) * np);
	pparticles_2D = malloc(sizeof(cvtx_P2D*) * np);
	int i;
	for (i = 0; i < np; ++i) {
		particles_2D[i].coord.x[0] = mrandf(maxf);
		particles_2D[i].coord.x[1] = mrandf(maxf);
		particles_2D[i].vorticity = mrandf(maxf);
		particles_2D[i].area = area;
		pparticles_2D[i] = &(particles_2D[i]);
	}
	return;
}


void create_particles_2D_outarr(int np) {
	oparticles_2D = malloc(sizeof(cvtx_P2D) * np);
}

void destroy_particles_2D() {
	free(particles_2D);
	free(pparticles_2D);
	return;
}

void destroy_oparticles_2D() {
	free(oparticles_2D);
	return;
}

void create_V3f_arr(int n, float maxf) {
	m_v3f_arr1 = malloc(sizeof(bsv_V3f) * n);
	int i;
	for (i = 0; i < n; ++i) {
		m_v3f_arr1[i].x[0] = mrandf(maxf);
		m_v3f_arr1[i].x[1] = mrandf(maxf);
		m_v3f_arr1[i].x[2] = mrandf(maxf);
	}
	return;
}

void create_V3f_arr2(int n, float maxf) {
	m_v3f_arr2 = malloc(sizeof(bsv_V3f) * n);
	int i;
	for (i = 0; i < n; ++i) {
		m_v3f_arr2[i].x[0] = mrandf(maxf);
		m_v3f_arr2[i].x[1] = mrandf(maxf);
		m_v3f_arr2[i].x[2] = mrandf(maxf);
	}
	return;
}

void destroy_V3f_arr() {
	free(m_v3f_arr1);
	return;
}

void destroy_V3f_arr2() {
	free(m_v3f_arr2);
	return;
}

void create_V2f_arr(int n, float maxf) {
	m_v2f_arr1 = malloc(sizeof(bsv_V2f) * n);
	int i;
	for (i = 0; i < n; ++i) {
		m_v2f_arr1[i].x[0] = mrandf(maxf);
		m_v2f_arr1[i].x[1] = mrandf(maxf);
	}
	return;
}

void create_V2f_arr2(int n, float maxf) {
	m_v2f_arr2 = malloc(sizeof(bsv_V2f) * n);
	int i;
	for (i = 0; i < n; ++i) {
		m_v2f_arr2[i].x[0] = mrandf(maxf);
		m_v2f_arr2[i].x[1] = mrandf(maxf);
	}
	return;
}

void destroy_V2f_arr() {
	free(m_v2f_arr1);
	return;
}

void destroy_V2f_arr2() {
	free(m_v2f_arr2);
	return;
}

void create_f_arr(int n, float maxf) {
	m_f_arr1 = malloc(sizeof(float) * n);
	int i;
	for (i = 0; i < n; ++i) {
		m_f_arr1[i] = mrandf(maxf);
	}
	return;
}

void destroy_f_arr() {
	free(m_f_arr1);
	return;
}

cvtx_P3D** particle_3D_pptr(void) {
	return pparticles_3D;
}

cvtx_P3D* oparticle_3D_ptr(void) {
	return oparticles_3D;
}

cvtx_P3D* particle_3D_ptr(void) {
	return particles_3D;
}

cvtx_P2D** particle_2D_pptr(void) {
	return pparticles_2D;
}

cvtx_P2D* oparticle_2D_ptr(void) {
	return oparticles_2D;
}

cvtx_P2D* particle_2D_ptr(void) {
	return particles_2D;
}

bsv_V3f* v3f_arr(void) {
	return m_v3f_arr1;
}

bsv_V3f* v3f_arr2(void) {
	return m_v3f_arr2;
}

bsv_V2f* v2f_arr(void) {
	return m_v2f_arr1;
}

bsv_V2f* v2f_arr2(void) {
	return m_v2f_arr2;
}

float* f_arr(void) {
	return m_f_arr1;
}
