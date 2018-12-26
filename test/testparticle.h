#ifndef CVTX_TEST_PARTICLE_H
#define CVTX_TEST_PARTICLE_H

/*============================================================================
testparticle.h

Test functionality of vortex particle & methods.

Copyright(c) 2018 HJA Bird

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
#include "../include/cvortex/Particle.h"
#include "../include/cvortex/VortFunc.h"
#include "../include/cvortex/Vec3f.h"

#include <math.h>

int testParticle(){
    SECTION("Particle");
    /*                  Pos    Vort   rad   */
    cvtx_Particle p1 = {0,0,0, 1,0,0, 1};
    cvtx_Vec3f v0 = {0,0,0};
    cvtx_Vec3f vx = {1,0,0};
    cvtx_Vec3f vy = {0,1,0};
    cvtx_Vec3f vz = {0,0,1};
    cvtx_Vec3f vzbig = {0,0,100};

    cvtx_VortFunc vfw = cvtx_VortFunc_winckelmans();
    cvtx_VortFunc vfs = cvtx_VortFunc_singular();

    /* Test velocity is induced in the correct places */
    /* Winckelmans */
    TEST(cvtx_Vec3f_isequal(
        cvtx_Particle_ind_vel(&p1, v0, &vfw, 1),
        v0));
    TEST(cvtx_Vec3f_isequal(
        cvtx_Particle_ind_vel(&p1, vx, &vfw, 1),
        v0));
    TEST(!cvtx_Vec3f_isequal(
        cvtx_Particle_ind_vel(&p1, vy, &vfw, 1),
        v0));
    TEST(!cvtx_Vec3f_isequal(
        cvtx_Particle_ind_vel(&p1, vz, &vfw, 1),
        v0));
    /* Singular */
    TEST(cvtx_Vec3f_isequal(
        cvtx_Particle_ind_vel(&p1, v0, &vfs, 1),
        v0));
    TEST(cvtx_Vec3f_isequal(
        cvtx_Particle_ind_vel(&p1, vx, &vfs, 1),
        v0));
    TEST(!cvtx_Vec3f_isequal(
        cvtx_Particle_ind_vel(&p1, vy, &vfs, 1),
        v0));
    TEST(!cvtx_Vec3f_isequal(
        cvtx_Particle_ind_vel(&p1, vz, &vfs, 1),
        v0));

    /* Test induced in the correct direction */
    TEST(cvtx_Particle_ind_vel(&p1, vy, &vfw, 1).x[0] == 0. );
    TEST(cvtx_Particle_ind_vel(&p1, vy, &vfw, 1).x[1] == 0. );
    TEST(cvtx_Particle_ind_vel(&p1, vy, &vfw, 1).x[2] > 0. );
    TEST(cvtx_Particle_ind_vel(&p1, vz, &vfw, 1).x[0] == 0. );
    TEST(cvtx_Particle_ind_vel(&p1, vz, &vfw, 1).x[1] < 0. );
    TEST(cvtx_Particle_ind_vel(&p1, vz, &vfw, 1).x[2] == 0. );
    TEST(cvtx_Particle_ind_vel(&p1, vy, &vfs, 1).x[0] == 0. );
    TEST(cvtx_Particle_ind_vel(&p1, vy, &vfs, 1).x[1] == 0. );
    TEST(cvtx_Particle_ind_vel(&p1, vy, &vfs, 1).x[2] > 0. );
    TEST(cvtx_Particle_ind_vel(&p1, vz, &vfs, 1).x[0] == 0. );
    TEST(cvtx_Particle_ind_vel(&p1, vz, &vfs, 1).x[1] < 0. );
    TEST(cvtx_Particle_ind_vel(&p1, vz, &vfs, 1).x[2] == 0. );

    /* Test induced velocity is smaller at large distances */
    TEST(cvtx_Particle_ind_vel(&p1, vz, &vfw, 1).x[1] <
        cvtx_Particle_ind_vel(&p1, vzbig, &vfw, 1).x[1]);
    TEST(cvtx_Particle_ind_vel(&p1, vz, &vfs, 1).x[1] <
        cvtx_Particle_ind_vel(&p1, vzbig, &vfs, 1).x[1]);

    /* Test induced vorticity */
    cvtx_Particle p2 =  {0,0,0, 0,0,1, 1};
    cvtx_Particle pxz = {0,0,1, 1,0,0, 1};
    cvtx_Particle pyz = {0,0,1, 0,1,0, 1};
    cvtx_Particle pzz = {0,0,1, 0,0,1, 1};
    TEST(cvtx_Particle_ind_dvort(&p2, &pxz, &vfs, 1).x[0] == 0);
    TEST(cvtx_Particle_ind_dvort(&p2, &pxz, &vfs, 1).x[1] > 0);
    TEST(cvtx_Particle_ind_dvort(&p2, &pxz, &vfs, 1).x[2] == 0);

    TEST(cvtx_Particle_ind_dvort(&p2, &pyz, &vfs, 1).x[0] < 0);
    TEST(cvtx_Particle_ind_dvort(&p2, &pyz, &vfs, 1).x[1] == 0);
    TEST(cvtx_Particle_ind_dvort(&p2, &pyz, &vfs, 1).x[2] == 0);

    TEST(cvtx_Particle_ind_dvort(&p2, &pzz, &vfs, 1).x[0] == 0);
    TEST(cvtx_Particle_ind_dvort(&p2, &pzz, &vfs, 1).x[1] == 0);
    TEST(cvtx_Particle_ind_dvort(&p2, &pzz, &vfs, 1).x[2] == 0);
    
    return 0;
}


#endif /* CVTX_TEST_PARTICLE_H */
