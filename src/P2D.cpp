#include "libcvtx.h"
/*============================================================================
P2D.cpp

Vortex particle in 2D with CPU based code.

Copyright(c) HJA Bird

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

#include <string>

#include "GridParticleQuadtree.h"
#include "UIntKey64.hpp"
#include "array_methods.h"
#include "redistribution_helper_funcs.h"
#include "vortex_kernels.h"
#include "particle_core_functions.hpp"

#ifdef CVTX_USING_OPENCL
#include "ocl_P2D.h"
#endif
#ifdef CVTX_USING_OPENMP
#include <omp.h>
#endif

#define NG_FOR_REDUCING_PARICLES 64

namespace cvtx {

/* Induced velocity at multiple points in a 2D field due to
a single particle. */
template <cvtx_VortFunc VortFunc>
void P2D_S2M_vel(const cvtx_P2D &self, const bsv_V2f *mes_start,
                 const int num_mes, bsv_V2f *result_array,
                 float regularisation_radius) {
  float recip_reg_rad = 1.f / std::abs(regularisation_radius);
#pragma omp parallel for
  for (int i = 0; i < num_mes; ++i) {
    result_array[i] = core::vel<VortFunc>(
        self, mes_start[i], recip_reg_rad);
  }
}

/* Induced velocity at a single point in a 2D field due to
multiple particles. */
template <cvtx_VortFunc VortFunc>
bsv_V2f P2D_M2S_vel(const cvtx_P2D *array_start, const int num_particles,
                    const bsv_V2f mes_point, float regularisation_radius) {
  float rx = 0, ry = 0;
  float recip_reg_rad = 1.f / std::abs(regularisation_radius);
#pragma omp parallel for reduction(+ : rx, ry)
  for (int i = 0; i < num_particles; ++i) {
    bsv_V2f vel = core::vel<VortFunc>(array_start[i], mes_point, recip_reg_rad);
    rx += vel.x[0];
    ry += vel.x[1];
  }
  return {rx, ry};
}

namespace open_mp {

template <cvtx_VortFunc VortFunc>
void P2D_M2M_vel(const cvtx_P2D *array_start, const int num_particles,
                 const bsv_V2f *mes_start, const int num_mes,
                 bsv_V2f *result_array, float regularisation_radius) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < num_mes; ++i) {
    result_array[i] = cvtx::P2D_M2S_vel<VortFunc>(
        array_start, num_particles, mes_start[i], regularisation_radius);
  }
  return;
}

void P2D_M2M_vel(const cvtx_P2D *array_start, const int num_particles,
                 const bsv_V2f *mes_start, const int num_mes,
                 bsv_V2f *result_array, const cvtx_VortFunc kernel,
                 float regularisation_radius) {
  switch (kernel) {
  case cvtx_VortFunc_singular:
    return P2D_M2M_vel<cvtx_VortFunc_singular>(array_start, num_particles, mes_start,
                                        num_mes, result_array,
                                        regularisation_radius);
  case cvtx_VortFunc_planetary:
    return P2D_M2M_vel<cvtx_VortFunc_planetary>(
        array_start, num_particles, mes_start,
                                         num_mes, result_array,
                                         regularisation_radius);
  case cvtx_VortFunc_winckelmans:
    return P2D_M2M_vel<cvtx_VortFunc_winckelmans>(
        array_start, num_particles,
        mes_start, num_mes, result_array,
        regularisation_radius);
  case cvtx_VortFunc_gaussian:
    return P2D_M2M_vel<cvtx_VortFunc_gaussian>(array_start, num_particles,
                                               mes_start,
                                               num_mes, result_array,
                                               regularisation_radius); 
  default:
    break;
  }
  return;
}

} // namespace open_mp
} // namespace cvtx

CVTX_EXPORT bsv_V2f cvtx_P2D_S2S_vel(const cvtx_P2D *self,
                                     const bsv_V2f mes_point,
                                     const cvtx_VortFunc kernel,
                                     float regularisation_radius) {
  bsv_V2f ret;
  float recip_rad = 1.f / std::abs(regularisation_radius);
  switch (kernel) {
  case cvtx_VortFunc_singular:
      return cvtx::core::vel<cvtx_VortFunc_singular>(*self, mes_point,
                                                      recip_rad);
  case cvtx_VortFunc_planetary:
    return cvtx::core::vel<cvtx_VortFunc_planetary>(*self, mes_point,
                                                    recip_rad);
  case cvtx_VortFunc_winckelmans:
    return cvtx::core::vel<cvtx_VortFunc_winckelmans>(*self, mes_point,
                                                      recip_rad);
  case cvtx_VortFunc_gaussian:
    return cvtx::core::vel<cvtx_VortFunc_gaussian>(*self, mes_point,
                                                   recip_rad);
  default:
    return {0.f, 0.f};
  }
}

CVTX_EXPORT void cvtx_P2D_S2M_vel(const cvtx_P2D *self,
                                  const bsv_V2f *mes_start, const int num_mes,
                                  bsv_V2f *result_array,
                                  const cvtx_VortFunc kernel,
                                  float regularisation_radius) {
  switch (kernel) {
  case cvtx_VortFunc_singular:
    cvtx::P2D_S2M_vel<cvtx_VortFunc_singular>(
        *self, mes_start, num_mes, result_array, regularisation_radius);
    break;
  case cvtx_VortFunc_planetary:
    cvtx::P2D_S2M_vel<cvtx_VortFunc_planetary>(
        *self, mes_start, num_mes, result_array, regularisation_radius);
    break;
  case cvtx_VortFunc_winckelmans:
    cvtx::P2D_S2M_vel<cvtx_VortFunc_winckelmans>(
        *self, mes_start, num_mes, result_array, regularisation_radius);
    break;
  case cvtx_VortFunc_gaussian:
    cvtx::P2D_S2M_vel<cvtx_VortFunc_gaussian>(
        *self, mes_start, num_mes, result_array, regularisation_radius);
    break;
  default:
    assert(false && "Unexpected kernel!");
  }
  return;
}

CVTX_EXPORT bsv_V2f cvtx_P2D_M2S_vel(const cvtx_P2D *array_start,
                                     const int num_particles,
                                     const bsv_V2f mes_point,
                                     const cvtx_VortFunc kernel,
                                     float regularisation_radius) {
  switch (kernel) {
  case cvtx_VortFunc_singular:
    return cvtx::P2D_M2S_vel<cvtx_VortFunc_singular>(
        array_start, num_particles, mes_point, regularisation_radius);
  case cvtx_VortFunc_planetary:
    return cvtx::P2D_M2S_vel<cvtx_VortFunc_planetary>(
        array_start, num_particles, mes_point, regularisation_radius);
  case cvtx_VortFunc_winckelmans:
    return cvtx::P2D_M2S_vel<cvtx_VortFunc_winckelmans>(
        array_start, num_particles, mes_point, regularisation_radius);
  case cvtx_VortFunc_gaussian:
    return cvtx::P2D_M2S_vel<cvtx_VortFunc_gaussian>(
        array_start, num_particles, mes_point, regularisation_radius);
  default:
    return {0.f, 0.f};
  }
}

CVTX_EXPORT void cvtx_P2D_M2M_vel(const cvtx_P2D *array_start,
                                  const int num_particles,
                                  const bsv_V2f *mes_start, const int num_mes,
                                  bsv_V2f *result_array,
                                  const cvtx_VortFunc kernel,
                                  float regularisation_radius) {
#ifdef CVTX_USING_OPENCL
  if (cvtx::vkernel::opencl_kernel_name_ext(kernel) == nullptr ||
      opencl_brute_force_P2D_M2M_vel(array_start, num_particles, mes_start,
                                     num_mes, result_array, kernel,
                                     regularisation_radius) != 0)
#endif
  {
    cvtx::open_mp::P2D_M2M_vel(array_start, num_particles, mes_start, num_mes,
                               result_array, kernel, regularisation_radius);
  }
  return;
}

/* Visous vorticity exchange methods ----------------------------------------*/


CVTX_EXPORT float cvtx_P2D_S2S_visc_dvort(const cvtx_P2D *self,
                                          const cvtx_P2D *induced_particle,
                                          const cvtx_VortFunc kernel,
                                          float regularisation_radius,
                                          float kinematic_visc) {
  switch (kernel) {
  case cvtx_VortFunc_gaussian:
      return cvtx::core::visc_dvort<cvtx_VortFunc_gaussian>(
        self, induced_particle, regularisation_radius, kinematic_visc);
  case cvtx_VortFunc_winckelmans:
    return cvtx::core::visc_dvort<cvtx_VortFunc_winckelmans>(
        self, induced_particle, regularisation_radius, kinematic_visc);
  default:
    assert(false && "Invalid kernel choice.");
    return 0.f;
  }
}

CVTX_EXPORT void
cvtx_P2D_S2M_visc_dvort(const cvtx_P2D *self, const cvtx_P2D *induced_start,
                        const int num_induced, float *result_array,
                        const cvtx_VortFunc kernel, float regularisation_radius,
                        float kinematic_visc) {
  int i;
#pragma omp parallel for
  for (i = 0; i < num_induced; ++i) {
    result_array[i] = cvtx_P2D_S2S_visc_dvort(
        self, induced_start + i, kernel, regularisation_radius, kinematic_visc);
  }
  return;
}

CVTX_EXPORT float cvtx_P2D_M2S_visc_dvort(const cvtx_P2D *array_start,
                                          const int num_particles,
                                          const cvtx_P2D *induced_particle,
                                          const cvtx_VortFunc kernel,
                                          float regularisation_radius,
                                          float kinematic_visc) {
  float dvort = 0.;
  assert(num_particles >= 0);
#pragma omp parallel for reduction(+ : dvort)
  for (int i = 0; i < num_particles; ++i) {
    dvort += cvtx_P2D_S2S_visc_dvort(array_start + i, induced_particle, kernel,
                                     regularisation_radius, kinematic_visc);
  }
  return dvort;
}

void cpu_brute_force_P2D_M2M_visc_dvort(
    const cvtx_P2D *array_start, const int num_particles,
    const cvtx_P2D *induced_start, const int num_induced, float *result_array,
    const cvtx_VortFunc kernel, float regularisation_radius,
    float kinematic_visc) {
  for (int i = 0; i < num_induced; ++i) {
    result_array[i] =
        cvtx_P2D_M2S_visc_dvort(array_start, num_particles, induced_start + i,
                                kernel, regularisation_radius, kinematic_visc);
  }
  return;
}

CVTX_EXPORT void
cvtx_P2D_M2M_visc_dvort(const cvtx_P2D *array_start, const int num_particles,
                        const cvtx_P2D *induced_start, const int num_induced,
                        float *result_array, const cvtx_VortFunc kernel,
                        float regularisation_radius, float kinematic_visc) {
#ifdef CVTX_USING_OPENCL
  if (num_particles < 256 || num_induced < 256 ||
      cvtx::vkernel::opencl_kernel_name_ext(kernel) == nullptr ||
      opencl_brute_force_P2D_M2M_visc_dvort(
          array_start, num_particles, induced_start, num_induced, result_array,
          kernel, regularisation_radius, kinematic_visc) != 0)
#endif
  {
    cpu_brute_force_P2D_M2M_visc_dvort(
        array_start, num_particles, induced_start, num_induced, result_array,
        kernel, regularisation_radius, kinematic_visc);
  }
  return;
}

/* Particle redistribution -------------------------------------------------*/
static int cvtx_remove_particles_under_str_threshold_2d(
    cvtx_P2D *io_arr, float *strs, int n_inpt_partices, float threshold,
    int max_keepable_particles);

CVTX_EXPORT int cvtx_P2D_redistribute_on_grid(
    const cvtx_P2D *input_array_start, const int n_input_particles,
    cvtx_P2D *output_particles, /* input is &(*cvtx_P2D) to write to */
    int max_output_particles,   /* Set to resultant num particles.   */
    const cvtx_RedistFunc *redistributor, const float grid_density,
    float negligible_vort) {

  assert(n_input_particles >= 0);
  assert(max_output_particles >= 0);
  assert(grid_density > 0.f);
  assert(negligible_vort >= 0.f);
  assert(negligible_vort < 1.f);
  size_t n_created_particles;
  int grid_radius;
  bsv_V2f min, mean; /* Bounds of the particle box.		*/
  const float recip_grid_density = 1.f / grid_density;
  /* For particle removal: */
  float min_keepable_particle;
#ifdef CVTX_USING_OPENMP
  unsigned int nthreads = omp_get_num_procs();
  omp_set_dynamic(0);
  omp_set_num_threads(nthreads);
#else
  unsigned int nthreads = 1;
#endif

  /* Generate grid keys for existing particles. */
  minmax_xy_posn(input_array_start, n_input_particles, &min, NULL);
  mean = mean_xy_posn(input_array_start, n_input_particles);
  grid_radius = (int)roundf(redistributor->radius);
  min =
      bsv_V2f_minus(min, bsv_V2f_mult({1.f, 1.f}, grid_radius * grid_density));
  bsv_V2f dcorner = bsv_V2f_div(bsv_V2f_minus(mean, min), grid_density);
  dcorner.x[0] = roundf(dcorner.x[0]) + 5;
  dcorner.x[1] = roundf(dcorner.x[1]) + 5;
  min = bsv_V2f_minus(mean, bsv_V2f_mult(dcorner, grid_density));

  /* Create an octtree on a grid and add all the new particles to it.
  We spread the work across multiple threads and then merge the results. */
  std::vector<GridParticleQuadtree> ptree(nthreads);
#pragma omp parallel for schedule(static)
  for (long long threadid = 0; threadid < nthreads; threadid++) {
    std::vector<UIntKey64> key_buffer;
    std::vector<float> str_buffer;
    size_t key_buffer_sz = UIntKey64::num_nearby_keys(grid_radius);
    key_buffer.resize(key_buffer_sz);
    str_buffer.resize(key_buffer_sz);
    size_t istart, iend; /* Particles for this thread. */
    istart = threadid * (n_input_particles / nthreads);
    iend = threadid == nthreads - 1
               ? n_input_particles
               : (threadid + 1) * (n_input_particles / nthreads);
    for (long long i = istart; i < (long long)iend; ++i) {
      bsv_V2f tparticle_pos = (input_array_start + i)->coord;
      float tparticle_str = (input_array_start + i)->vorticity;
      UIntKey64 key = UIntKey64::nearest_key_min((input_array_start + i)->coord,
                                                 recip_grid_density, min);
      key.nearby_keys(grid_radius, key_buffer.data(), key_buffer.size());
      for (size_t j = 0; j < key_buffer_sz; ++j) {
        bsv_V2f npos = key_buffer[j].to_position_min(grid_density, min);
        float U, W, vortfrac;
        bsv_V2f dx = bsv_V2f_minus(tparticle_pos, npos);
        U = std::abs(dx.x[0] * recip_grid_density);
        W = std::abs(dx.x[1] * recip_grid_density);
        vortfrac = redistributor->func(U) * redistributor->func(W);
        str_buffer[j] = tparticle_str * vortfrac;
      }
      ptree[threadid].add_particles(key_buffer, str_buffer);
    }
  }
  for (int threadid = 1; threadid < (int)nthreads; threadid++) {
    ptree[0].merge_in(ptree[threadid]);
  }
  ptree.resize(1);
  GridParticleQuadtree &tree = ptree[0];
  /* Go back to array of particles. */
  n_created_particles = tree.number_of_particles();
  std::vector<UIntKey64> new_particle_keys(n_created_particles);
  std::vector<float> new_particle_strs(n_created_particles);
  std::vector<cvtx_P2D> new_particles(n_created_particles);
  tree.flatten_tree(new_particle_keys.data(), new_particle_strs.data(),
                    n_created_particles);
  float np_vol = grid_density * grid_density;
  for (int i = 0; i < n_created_particles; ++i) {
    new_particles[i].area = np_vol;
    new_particles[i].vorticity = new_particle_strs[i];
    new_particles[i].coord =
        new_particle_keys[i].to_position_min(grid_density, min);
  }
  new_particle_keys.clear();
  new_particle_strs.clear();
  /* Remove particles with neglidgible vorticity. */
  std::vector<float> strengths(n_created_particles);
#pragma omp parallel for
  for (long long i = 0; i < n_created_particles; ++i) {
    strengths[i] = std::abs(new_particles[i].vorticity);
  }
  farray_info(strengths.data(), n_created_particles, &min_keepable_particle,
              NULL, NULL);
  min_keepable_particle = min_keepable_particle * negligible_vort;
  n_created_particles = cvtx_remove_particles_under_str_threshold_2d(
      new_particles.data(), strengths.data(), n_created_particles,
      min_keepable_particle, n_created_particles);
  new_particles.resize(n_created_particles);
  /* The strengths are modified to keep total vorticity constant. */
#pragma omp parallel for
  for (long long i = 0; i < n_created_particles; ++i) {
    strengths[i] = std::abs(new_particles[i].vorticity);
  }
  /* Now to handle what we return to the caller */
  if (output_particles != NULL) {
    if (n_created_particles > max_output_particles) {
      min_keepable_particle = get_strength_threshold(
          strengths.data(), n_created_particles, max_output_particles);
      n_created_particles = cvtx_remove_particles_under_str_threshold_2d(
          new_particles.data(), strengths.data(), n_created_particles,
          min_keepable_particle, max_output_particles);
    }
    /* And now make an array to return to our caller. */
    memcpy(output_particles, new_particles.data(),
           sizeof(cvtx_P2D) * n_created_particles);
  }
  return n_created_particles;
}

int cvtx_remove_particles_under_str_threshold_2d(cvtx_P2D *io_arr, float *strs,
                                                 int n_inpt_partices,
                                                 float min_keepable_str,
                                                 int max_keepable_particles) {

  float vorticity_deficit;
  int n_output_particles;
  int i, j;

  j = 0;
  vorticity_deficit = 0.f;

  for (i = 0; i < n_inpt_partices; ++i) {
    if (strs[i] > min_keepable_str && i < max_keepable_particles) {
      io_arr[j] = io_arr[i];
      ++j;
    } else {
      /* For vorticity conservation. */
      vorticity_deficit = io_arr[i].vorticity + vorticity_deficit;
    }
  }

  n_output_particles = j;
  vorticity_deficit = vorticity_deficit / (float)n_output_particles;
  for (i = 0; i < n_output_particles; ++i) {
    io_arr[i].vorticity = io_arr[i].vorticity + vorticity_deficit;
  }
  return n_output_particles;
}
