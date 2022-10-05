#include "libcvtx.h"
/*============================================================================
VortFunc.c

Common functions used to regularise vortex particles.

If we have a vorticity field omega(x) comprised of particles with vorticity
alpha_i, regularisation zeta(rho), where rho is the distance between x_i
and measurement point x divided by regularisation distance sigma then
omega(x) = sum( zeta(rho_i) * alpha_i )

These regularisation function are given by zeta, but exclude the 4*pi
part - that constant is in the evaluation bits.

The velocity includes a function g(rho) defined by
zeta(rho) = 1/rho^2 * dg/drho
again excluding the 4 pi bit.

For the particle strenght exchange schmeme another funcion eta(rho) is
required.
eta(rho) = -1/rho * (dzeta/drho)

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
#ifndef CVTX_VORTEX_KERNELS_H
#define CVTX_VORTEX_KERNELS_H

#include <assert.h>
#include <cmath>

#include "fast_maths.hpp"

#define SQRTF_2_OVER_PI 0.7978845608028654f
#define RECIP_SQRTF_2 0.7071067811865475f

namespace cvtx {
namespace vkernel {

/* Interface: Vortex equations */
template <cvtx_VortFunc VortFunc> inline float zeta_fn(float rho);
template <cvtx_VortFunc VortFunc> inline float g_2D(float rho);
template <cvtx_VortFunc VortFunc> inline float g_3D(float rho);
template <cvtx_VortFunc VortFunc> inline float eta_2D(float rho);
template <cvtx_VortFunc VortFunc> inline float eta_3D(float rho);

/* Interface: Vortex traits */
template <cvtx_VortFunc VortFunc> constexpr bool has_eta_fns() { return false; }
template <cvtx_VortFunc VortFunc> constexpr const char *opencl_kernel_name_ext();

/* --------------------------------------------------------------------------*/
/* SPECIALISATIONS ----------------------------------------------------------*/

/* Specialisations for the singular kernel */
template <> inline float g_2D<cvtx_VortFunc_singular>(float) { return 1.f; }
template <> inline float g_3D<cvtx_VortFunc_singular>(float) { return 1.f; }
template <> inline float zeta_fn<cvtx_VortFunc_singular>(float) { return 0.f; }
/* No eta functions for singular kernels */
template <> constexpr const char *opencl_kernel_name_ext<cvtx_VortFunc_singular>() {
  const char *str = "singular";
  return str;
}

/* Winckelmans kernels */
template <> inline float g_3D<cvtx_VortFunc_winckelmans>(float rho) {
  float a, b, c, d;
  a = (rho * rho) + 2.5f;
  b = a * rho * (rho * rho);
  c = (rho * rho) + 1.f;
  d = b / std::sqrt(maths::pow<5>(c));
  return d;
}
template <> inline float zeta_fn<cvtx_VortFunc_winckelmans>(float rho) {
  float a, b, c;
  a = rho * rho + 1.f;
  b = 1.f / std::sqrt(maths::pow<7>(a));
  c = 7.5f * b;
  return c;
}
template <> inline float g_2D<cvtx_VortFunc_winckelmans>(float rho) {
  float num, denom;
  num = maths::pow<4>(rho) + maths::pow<2>(rho) * 2.f;
  denom = maths::pow<4>(rho) + 2.f * maths::pow<2>(rho) + 1.f;
  return num / denom;
}
template <> constexpr bool has_eta_fns<cvtx_VortFunc_winckelmans>() {
  return true;
}
template <> inline float eta_2D<cvtx_VortFunc_winckelmans>(float rho) {
  float a, a2, c;
  a = rho * rho + 1.f;
  a2 = 1.f / (a * a);
  c = 24.f * std::exp(4.f * a * (a2 * a2));
  return c * (a2 * a2);
}
template <> inline float eta_3D<cvtx_VortFunc_winckelmans>(float rho) {
  float a, b, c;
  a = 52.5f;
  b = rho * rho + 1.f;
  c = 1.f / std::sqrt(maths::pow<9>(b));
  return a * c;
}
template <> constexpr const char *opencl_kernel_name_ext<cvtx_VortFunc_winckelmans>() {
  const char *str = "winckelmans";
  return str;
}

/* Planetary kernels */
template <>
inline float g_3D<cvtx_VortFunc_planetary>(float rho) {
  return rho < 1.f ? maths::pow<3>(rho) : 1.f;
}
template <> inline float zeta_fn<cvtx_VortFunc_planetary>(float rho) {
  return rho < 1.f ? 3.f : 0.f;
}
template <> inline float g_2D<cvtx_VortFunc_planetary>(float rho) {
  return rho < 1.f ? rho * rho : 1.f;
}
/* No eta functions for exponential kernels */
template <> constexpr const char *opencl_kernel_name_ext<cvtx_VortFunc_planetary>() {
  const char *str = "planetary";
  return str;
}

/* Gaussian kernels */
template <> inline float g_3D<cvtx_VortFunc_gaussian>(float rho) {
  /* = 1 to 8sf for rho ~>6. Taylor expansion otherwise */
  float ret;
  if (rho > 6.f) {
    ret = 1.f;
  } else {
    /* Approximate erf using Abramowitz and Stegan 1.7.26 */
    float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f;
    float a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
    float rho_sr2 = rho * RECIP_SQRTF_2;
    float t = 1.f / (1.f + p * rho_sr2);
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t2 * t2;
    float t5 = t3 * t2;
    float erf = 1.f - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) *
                          std::exp(-rho_sr2 * rho_sr2);
    float term2 = rho * SQRTF_2_OVER_PI * std::exp(-rho_sr2 * rho_sr2);
    ret = erf - term2;
  }
  return ret;
}
template <> inline float zeta_fn<cvtx_VortFunc_gaussian>(float rho) {
  return SQRTF_2_OVER_PI * std::exp(-rho * rho * 0.5f);
}
template <> inline float g_2D<cvtx_VortFunc_gaussian>(float rho) {
  return 1.f - std::exp(-rho * rho * 0.5f);
}
template <> constexpr bool has_eta_fns<cvtx_VortFunc_gaussian>() {
  return true;
}
template <> inline float eta_2D<cvtx_VortFunc_gaussian>(float rho) {
  return std::exp(-rho * rho * 0.5f);
}
template <> inline float eta_3D<cvtx_VortFunc_gaussian>(float rho) {
  return zeta_fn<cvtx_VortFunc_gaussian>(rho);
}
template <> constexpr const char *opencl_kernel_name_ext<cvtx_VortFunc_gaussian>() {
  const char *str = "gaussian";
  return str;
}

inline const char *opencl_kernel_name_ext(cvtx_VortFunc kernel){
  switch(kernel) {
  case cvtx_VortFunc_singular:
    return opencl_kernel_name_ext<cvtx_VortFunc_singular>();
  case cvtx_VortFunc_planetary:
    return opencl_kernel_name_ext<cvtx_VortFunc_planetary>();
  case cvtx_VortFunc_winckelmans:
    return opencl_kernel_name_ext<cvtx_VortFunc_winckelmans>();
  case cvtx_VortFunc_gaussian:
    return opencl_kernel_name_ext<cvtx_VortFunc_gaussian>();
  default:
    return nullptr;  
  }
}

} // namespace vkernel
} // namespace cvtx

#endif // CVTX_VORTEX_KERNELS_H
