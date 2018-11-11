
#include "../include/cvortex/Particle.h"
#include "../include/cvortex/VortFunc.h"
#include "../include/cvortex/LegacyVtk.h"
#define NUM_PER_RING 15
#include <math.h>
#include <stdio.h>

int main(int argc, char* argv[])
{
    float dt = (float)0.5;
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
    cvtx_Vec3f dvorts_visc[NUM_PER_RING*2];
    char file_name[128];
    for(int step = 0; step < 300; ++step){
        for(int i =0; i < NUM_PER_RING*2; i++){
            mes_pnts[i] = m_particles[i].coord;
        }
        cvtx_VortFunc vort_fn = cvtx_VortFunc_gaussian();
        cvtx_ParticleArr_Arr_ind_vel(
            (const cvtx_Particle**)m_particle_ptrs, NUM_PER_RING*2,
            mes_pnts, NUM_PER_RING*2,
            vels, &vort_fn);
        cvtx_ParticleArr_Arr_ind_dvort(
            (const cvtx_Particle**)m_particle_ptrs, NUM_PER_RING*2,
            (const cvtx_Particle**)m_particle_ptrs, NUM_PER_RING*2,
            dvorts, &vort_fn);
        cvtx_ParticleArr_Arr_visc_ind_dvort(
            (const cvtx_Particle**)m_particle_ptrs, NUM_PER_RING*2,
            (const cvtx_Particle**)m_particle_ptrs, NUM_PER_RING*2,
            dvorts_visc, &vort_fn, 0.01f);
        for(int i =0; i < NUM_PER_RING*2; i++){
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
        if(cvtx_ParticleArr_to_vtk(file_name, m_particle_ptrs, NUM_PER_RING*2)){
            printf("Failed to write to file.\n");
            break;
        } 
    }
    return 0;
}
