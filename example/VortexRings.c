
#include "../include/cvortex/Particle.h"
#include "../include/cvortex/ParticleKernelFunctions.h"
#include "../include/cvortex/LegacyVtk.h"
#define NUM_PER_RING 200
#include <math.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
    float dt = (float)0.1;
    cvtx_Particle m_particles[NUM_PER_RING * 2];
    cvtx_Particle *m_particle_ptrs[NUM_PER_RING * 2];
    float z = 0;
    for(int i = 0; i < NUM_PER_RING; i++){
        m_particle_ptrs[i] = &m_particles[i];
        m_particles[i].coord.x[2] = z;
        m_particles[i].coord.x[0] = cosf((float)(2 * 3.141592 * (float) i / NUM_PER_RING));
        m_particles[i].coord.x[1] = sinf((float)(2 * 3.141592 * (float) i / NUM_PER_RING));
        m_particles[i].vorticity.x[2] = 0;
        m_particles[i].vorticity.x[0] = -sinf((float)(2 * 3.141592 * (float) i / NUM_PER_RING)) / NUM_PER_RING;
        m_particles[i].vorticity.x[1] = cosf((float)(2 * 3.141592 * (float) i / NUM_PER_RING)) / NUM_PER_RING;
        m_particles[i].radius = (float) (1.5 * 3.141592 / NUM_PER_RING);
    }
    z = 1;
    for(int i = NUM_PER_RING; i < NUM_PER_RING * 2; i++){
        m_particle_ptrs[i] = &m_particles[i];
        m_particles[i].coord.x[2] = z;
        m_particles[i].coord.x[0] = cosf((float)(2 * 3.141592 * (float) i / NUM_PER_RING));
        m_particles[i].coord.x[1] = sinf((float)(2 * 3.141592 * (float) i / NUM_PER_RING));
        m_particles[i].vorticity.x[2] = 0;
        m_particles[i].vorticity.x[0] = -sinf((float)(2 * 3.141592 * (float) i / NUM_PER_RING)) / NUM_PER_RING;
        m_particles[i].vorticity.x[1] = cosf((float)(2 * 3.141592 * (float) i / NUM_PER_RING)) / NUM_PER_RING;
        m_particles[i].radius = (float) (1.5 * 3.141592 / NUM_PER_RING);
    }

    cvtx_Vec3f mes_pnts[NUM_PER_RING*2];
    cvtx_Vec3f vels[NUM_PER_RING*2];
    cvtx_Vec3f dvorts[NUM_PER_RING*2];
    char file_name[128];
    for(int step = 0; step < 300; ++step){
        for(int i =0; i < NUM_PER_RING*2; i++){
            mes_pnts[i] = m_particles[i].coord;
        }
        cvtx_VortFunc vort_fn = cvtx_VortFunc_singular();
        cvtx_ParticleArr_Arr_ind_vel(
            m_particle_ptrs, NUM_PER_RING*2,
            mes_pnts, NUM_PER_RING*2,
            vels, &vort_fn);
        cvtx_ParticleArr_Arr_ind_dvort(
            m_particle_ptrs, NUM_PER_RING*2,
            m_particle_ptrs, NUM_PER_RING*2,
            dvorts, &vort_fn);
        for(int i =0; i < NUM_PER_RING*2; i++){
            m_particles[i].coord = cvtx_Vec3f_plus(
                m_particles[i].coord, 
                cvtx_Vec3f_mult(vels[i], dt));
            m_particles[i].vorticity = cvtx_Vec3f_plus(
                m_particles[i].vorticity, 
                cvtx_Vec3f_mult(dvorts[i], dt));
        }
        sprintf(file_name, "./output/particles_%i.vtk", step);
        cvtx_ParticleArr_to_vtk(file_name, m_particle_ptrs, NUM_PER_RING*2); 
    }
    return 0;
}
