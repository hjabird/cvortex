#include "../include/cvortex/Particle.h"
#include "../include/cvortex/VortFunc.h"
#include "../include/cvortex/LegacyVtk.h"
#include <math.h>
#include <stdio.h>

#define NUM_PER_RING 60
#define NUM_RINGS 60
#define TOTAL_PARTICLES  NUM_PER_RING*NUM_RINGS
const int num_steps = 75;

int main(int argc, char* argv[])
{
    float dt = (float)0.025;
	float adj;
	const float pi = 3.14159265359;
	int i, step;
    cvtx_Particle m_particles[TOTAL_PARTICLES];
    cvtx_Particle *m_particle_ptrs[TOTAL_PARTICLES];
    for(i = 0; i < TOTAL_PARTICLES; i++){
        m_particle_ptrs[i] = &m_particles[i];
        m_particles[i].coord.x[2] = floorf((float)i / NUM_PER_RING) * 1.5 * pi / NUM_PER_RING / 4;
        m_particles[i].coord.x[0] = cosf((float)(2 * pi * (float) i / NUM_PER_RING));
        m_particles[i].coord.x[1] = sinf((float)(2 * pi * (float) i / NUM_PER_RING));
		adj = 2 * (float)(i / NUM_PER_RING) / (TOTAL_PARTICLES / NUM_PER_RING) - 1;
        m_particles[i].vorticity.x[2] = 0;
        m_particles[i].vorticity.x[0] = adj * -sinf((float)(2 * pi * (float) i / NUM_PER_RING)) / NUM_PER_RING;
        m_particles[i].vorticity.x[1] = adj * cosf((float)(2 * pi * (float) i / NUM_PER_RING)) / NUM_PER_RING;
        m_particles[i].radius = (float) (1.5 * pi / NUM_PER_RING);
    }


    cvtx_Vec3f mes_pnts[TOTAL_PARTICLES];
    cvtx_Vec3f vels[TOTAL_PARTICLES];
    cvtx_Vec3f dvorts[TOTAL_PARTICLES];
    cvtx_Vec3f dvorts_visc[TOTAL_PARTICLES];
    char file_name[128];
    for(step = 0; step < num_steps; ++step){
        for(i =0; i < TOTAL_PARTICLES; i++){
            mes_pnts[i] = m_particles[i].coord;
        }
        cvtx_VortFunc vort_fn = cvtx_VortFunc_gaussian();
        cvtx_ParticleArr_Arr_ind_vel(
            (cvtx_Particle**)m_particle_ptrs, TOTAL_PARTICLES,
            mes_pnts, TOTAL_PARTICLES,
            vels, &vort_fn);
        cvtx_ParticleArr_Arr_ind_dvort(
            (cvtx_Particle**)m_particle_ptrs, TOTAL_PARTICLES,
            (cvtx_Particle**)m_particle_ptrs, TOTAL_PARTICLES,
            dvorts, &vort_fn);
        cvtx_ParticleArr_Arr_visc_ind_dvort(
            (cvtx_Particle**)m_particle_ptrs, TOTAL_PARTICLES,
            (cvtx_Particle**)m_particle_ptrs, TOTAL_PARTICLES,
            dvorts_visc, &vort_fn, 1.f);
        for(i =0; i < TOTAL_PARTICLES; i++){
            m_particles[i].coord = cvtx_Vec3f_plus(
                m_particles[i].coord, 
                cvtx_Vec3f_mult(vels[i], dt));
            m_particles[i].vorticity = cvtx_Vec3f_plus(
                m_particles[i].vorticity, 
                cvtx_Vec3f_mult(dvorts[i], dt));
            m_particles[i].vorticity = cvtx_Vec3f_plus(
                m_particles[i].vorticity, 
                cvtx_Vec3f_mult(dvorts_visc[i], dt));
        }
        sprintf(file_name, "./output/particles_%i.vtk", step);
        if(cvtx_ParticleArr_to_vtk(file_name, m_particle_ptrs, TOTAL_PARTICLES)){
            printf("Failed to write to file.\n");
            break;
        } 
    }
    printf("%d interactions computed.\n", TOTAL_PARTICLES * TOTAL_PARTICLES * num_steps );
    return 0;
}
