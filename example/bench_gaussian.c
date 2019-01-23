#include "../include/cvortex/Particle.h"
#include "../include/cvortex/VortFunc.h"
#include "../include/cvortex/LegacyVtk.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define NUM_PER_RING 256
#define NUM_RINGS 512
#define TOTAL_PARTICLES  NUM_PER_RING*NUM_RINGS

int main(int argc, char* argv[])
{
	double factor = sqrt(2.), ms_correction = 1000. / (double)CLOCKS_PER_SEC;
	long testsize = 8, this_testsize, repeats = 3, i, power;
	clock_t s, e, tvel, tdvort, tvdvort;
	tvel = 99999999;
	tdvort = tvel;
	tvdvort = tvel;
	float adj;
	const float pi = (float) 3.14159265359;
	float regularisation_rad = (float)(1.5 * pi / NUM_PER_RING);
	cvtx_Particle *m_particles, **m_particle_ptrs;
	m_particles = malloc(sizeof(cvtx_Particle) * TOTAL_PARTICLES);
	m_particle_ptrs = malloc(sizeof(cvtx_Particle*) * TOTAL_PARTICLES);
	for (i = 0; i < TOTAL_PARTICLES; i++) {
		m_particle_ptrs[i] = &m_particles[i];
		m_particles[i].coord.x[2] = (float)(floorf((float)i / NUM_PER_RING) * 1.5 * pi / NUM_PER_RING / 4);
		m_particles[i].coord.x[0] = cosf((float)(2 * pi * (float)i / NUM_PER_RING));
		m_particles[i].coord.x[1] = sinf((float)(2 * pi * (float)i / NUM_PER_RING));
		adj = 2 * (float)(i / NUM_PER_RING) / (TOTAL_PARTICLES / NUM_PER_RING) - 1;
		m_particles[i].vorticity.x[2] = 0;
		m_particles[i].vorticity.x[0] = adj * -sinf((float)(2 * pi * (float)i / NUM_PER_RING)) / NUM_PER_RING;
		m_particles[i].vorticity.x[1] = adj * cosf((float)(2 * pi * (float)i / NUM_PER_RING)) / NUM_PER_RING;
		m_particles[i].volume = (2 * pi / NUM_PER_RING) * (1 / TOTAL_PARTICLES);
	}


	cvtx_Vec3f *mes_pnts, *vels, *dvorts, *dvorts_visc;
	mes_pnts = malloc(sizeof(cvtx_Vec3f) * TOTAL_PARTICLES);
	vels = malloc(sizeof(cvtx_Vec3f) * TOTAL_PARTICLES);
	dvorts = malloc(sizeof(cvtx_Vec3f) * TOTAL_PARTICLES);
	dvorts_visc = malloc(sizeof(cvtx_Vec3f) * TOTAL_PARTICLES);
	cvtx_VortFunc vort_fn = cvtx_VortFunc_gaussian();
	printf("Benchmarking with Gaussian Kernel.\n");
	printf("\t\tMilliseconds\t\t\tNormalised (Time per interaction)\n");
	printf("Particles\tvel\tdvort\tvisc dvort\n");
	for (power = 0; testsize * pow(factor, power) < TOTAL_PARTICLES; ++power) {
		this_testsize = (long)(testsize * pow(factor, power));
		for (i = 0; i < repeats; ++i) {
			s = clock();
			cvtx_ParticleArr_Arr_ind_vel(
				(cvtx_Particle**)m_particle_ptrs, this_testsize,
				mes_pnts, this_testsize,
				vels, &vort_fn, regularisation_rad);
			e = clock();
			tvel = e - s < tvel ? e - s : tvel;
		}
		for (i = 0; i < repeats; ++i) {
			s = clock();
			cvtx_ParticleArr_Arr_ind_dvort(
				(cvtx_Particle**)m_particle_ptrs, this_testsize,
				(cvtx_Particle**)m_particle_ptrs, this_testsize,
				dvorts, &vort_fn, regularisation_rad);
			e = clock();
			tdvort = e - s < tdvort ? e - s : tdvort;
		}
		for (i = 0; i < repeats; ++i) {
			s = clock();
			cvtx_ParticleArr_Arr_visc_ind_dvort(
				 (cvtx_Particle**)m_particle_ptrs, this_testsize,
				 (cvtx_Particle**)m_particle_ptrs, this_testsize,
				 dvorts_visc, &vort_fn, regularisation_rad, 0.01f);
			e = clock();
			tvdvort = e - s < tvdvort ? e - s : tvdvort;
		}
		printf("%i:\t\t%.2e\t%.2e\t%.2e\t\t%.2e\t%.2e\t%.2e\n", this_testsize, tvel*ms_correction, tdvort*ms_correction, tvdvort*ms_correction,
				pow((double)sqrt(tvel)*ms_correction / this_testsize, 2), pow((double)sqrt(tdvort)*ms_correction / this_testsize, 2),
				pow((double)sqrt(tvdvort)*ms_correction / this_testsize, 2));

		tvel = 99999999;
		tdvort = tvel;
		tvdvort = tvel;
	}
	return 0;
}
