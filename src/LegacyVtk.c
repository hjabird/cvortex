#include "../include/cvortex/LegacyVtk.h"
/*============================================================================
LegacyVtk.c

Make a vtk legacy file from a collection of vortex particles.

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

#include <stdio.h>

int cvtx_ParticleArr_to_vtk(
    char* path, 
    cvtx_Particle **particles, 
    int num_particles)
{
    FILE *file;
    file = fopen(path, "wb");
    if(file == NULL){
        printf("Could not open file at path :%s\n", path);
        return 1;
    }
    fprintf(file, "# vtk DataFile Version 2.0\n");
    fprintf(file, "Vortex particles\n");
    fprintf(file, "ASCII\n");
    fprintf(file, "DATASET UNSTRUCTURED_GRID\nPOINTS %i float\n", num_particles);
    for(int i = 0; i < num_particles; ++i){
        fprintf(file, "%f %f %f\n", 
            particles[i]->coord.x[0],
            particles[i]->coord.x[1],
            particles[i]->coord.x[2]);
    }
    fprintf(file, "\nCELLS %i %i\n", num_particles, 2 * num_particles);
    for(int i = 0; i < num_particles; ++i){
        fprintf(file, "1 %i\n", i);
    }
    fprintf(file, "\nCELL_TYPES %i\n", num_particles);
    for(int i = 0; i < num_particles; ++i){
        fprintf(file, "1\n");
    }
    fprintf(file, "\nCELL_DATA %i\n", num_particles);
    fprintf(file, "VECTORS Vorticity float\n");
    for(int i = 0; i < num_particles; ++i){
        fprintf(file, "%f %f %f\n", 
            particles[i]->vorticity.x[0],
            particles[i]->vorticity.x[1],
            particles[i]->vorticity.x[2]);
    }
    return 0;
}