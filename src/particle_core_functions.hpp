#ifndef CVTX_PARTICLE_CORE_FUNCTIONS_H
#define CVTX_PARTICLE_CORE_FUNCTIONS_H
#include "libcvtx.h"
/*============================================================================
Particle core functions.hpp

The core functions for induced velocity, vortex stretching, viscous effects,
and vorticity.

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

#include <Eigen/Dense>

namespace cvtx {
namespace core {

constexpr float pi_f = 3.14159265359f;

// 2D -------------------------------------------------------------------------
template <cvtx_VortFunc VortFunc>
inline bsv_V2f vel(const cvtx_P2D &self, const bsv_V2f &mes_point,
            float recip_reg_rad) {
  bsv_V2f rad, ret;
  float radd, rho, g, cor;
  rad = bsv_V2f_minus(mes_point, self.coord);
  radd = bsv_V2f_abs(rad);
  rho = radd * recip_reg_rad;
  g = vkernel::g_2D<VortFunc>(rho);
  cor = self.vorticity / (radd * radd);
  ret.x[0] = g * rad.x[1] * cor;
  ret.x[1] = g * -rad.x[0] * cor;
  ret = bsv_V2f_isequal(self.coord, mes_point) ? bsv_V2f_zero() : ret;
  return bsv_V2f_mult(ret, 1.f / (2.f * pi_f));
}

template<cvtx_VortFunc VortFunc, int NumPoints>
inline Eigen::Matrix<float, NumPoints, 2> vel(
    const Eigen::Matrix<float, 1, 2> particleCoord,
                const float particleVort,
                const Eigen::Matrix<float, NumPoints, 2>& mesCoord,
                float recipRegRad) {
  using column_array_t = Eigen::Array<float, NumPoints, 1>;
  using matrix_t = Eigen::Matrix<float, NumPoints, 2>;
  matrix_t coordDiff = mesCoord.rowwise() - particleCoord;
  column_array_t radius = coordDiff.rowwise().norm().array();
  column_array_t rho = radius * recipRegRad;
  column_array_t g =
      rho.unaryExpr([](float rho) { return vkernel::g_2D<VortFunc>(rho); });
  column_array_t cor = (g / radius.square()) * particleVort;
  // Where cor is NaN, set it to zero.
  cor = (!(radius > 0)).select(0.f, cor);
  // Rotate the coord: {x[1], -x[0]}
  coordDiff.col(0).swap(coordDiff.col(1));
  coordDiff.col(1) *= -1;
  matrix_t ret = (coordDiff.array().colwise() * cor).matrix();
  return ret * (1.f / (2.f * pi_f));
}

template <cvtx_VortFunc VortFunc>
inline float visc_dvort(const cvtx_P2D *self, const cvtx_P2D *induced_particle,
                        float regularisation_radius, float kinematic_visc) {
  bsv_V2f rad;
  float radd, rho, ret, t1, t2, t22, t21, t211, t212;
  if (bsv_V2f_isequal(self->coord, induced_particle->coord)) {
    ret = 0.f;
  } else {
    rad = bsv_V2f_minus(self->coord, induced_particle->coord);
    radd = bsv_V2f_abs(rad);
    rho = std::abs(radd / regularisation_radius);
    t1 = 2 * kinematic_visc / (regularisation_radius * regularisation_radius);
    t211 = self->vorticity * induced_particle->area;
    t212 = -induced_particle->vorticity * self->area;
    t21 = t211 + t212;
    t22 = vkernel::eta_2D<VortFunc>(rho);
    t2 = t21 * t22;
    ret = t2 * t1;
  }
  return ret;
}

// 3D ---------------------------------------------------------------------------
template <cvtx_VortFunc VortFunc>
inline bsv_V3f vel(const cvtx_P3D &self, const bsv_V3f mes_point,
                       float recip_reg_rad) {
  bsv_V3f rad, num, ret;
  float cor, den, rho, radd;
  rad = bsv_V3f_minus(mes_point, self.coord);
  radd = bsv_V3f_abs(rad);
  rho = radd * recip_reg_rad; /* Assume positive. */
  cor = -vkernel::g_3D<VortFunc>(rho);
  den = 1.f / (radd * radd * radd);
  num = bsv_V3f_cross(rad, self.vorticity);
  ret = bsv_V3f_mult(num, cor * den);
  ret = bsv_V3f_isequal(self.coord, mes_point) ? bsv_V3f_zero() : ret;
  return bsv_V3f_mult(ret, 1.f / (4.f * pi_f));
}

template <cvtx_VortFunc VortFunc, int NumPoints>
inline Eigen::Matrix<float, NumPoints, 3> vel(const Eigen::Matrix<float, 1, 3> particleCoord,
                const Eigen::Matrix<float, 1, 3> particleVort,
                const Eigen::Matrix<float, NumPoints, 3> &mesCoord,
                float recipRegRad) {
  using column_array_t = Eigen::Array<float, NumPoints, 1>;
  using matrix_t = Eigen::Matrix<float, NumPoints, 3>;
  matrix_t coordDiff = mesCoord.rowwise() - particleCoord;
  column_array_t radius = coordDiff.rowwise().norm().array();
  column_array_t rho = radius * recipRegRad;
  column_array_t g =
      rho.unaryExpr([](float rho) { return -vkernel::g_3D<VortFunc>(rho); });
  column_array_t cor = g / radius.cube();
  // Where cor is NaN, set it to zero.
  cor = (!(radius > 0)).select(0.f, cor);
  matrix_t mat = coordDiff.rowwise().cross(particleVort);
  matrix_t ret = (mat.array().colwise() * cor).matrix();
  return ret * (1.f / (4.f * pi_f));
}

template <cvtx_VortFunc VortFunc>
inline bsv_V3f dvort(const cvtx_P3D &self, const cvtx_P3D &induced_particle,
                  float regularisation_radius) {
  bsv_V3f ret, rad, cross_om, t2, t21, t21n, t22;
  float g, f, radd, rho, t1, t21d, t221, t222, t223;
  rad = bsv_V3f_minus(induced_particle.coord, self.coord);
  radd = bsv_V3f_abs(rad);
  rho = std::abs(radd / regularisation_radius);
  g = vkernel::g_3D<VortFunc>(rho);
  f = vkernel::zeta_fn<VortFunc>(rho);
  cross_om = bsv_V3f_cross(induced_particle.vorticity, self.vorticity);
  t1 = 1.f / (4.f * pi_f * regularisation_radius * regularisation_radius *
              regularisation_radius);
  t21n = bsv_V3f_mult(cross_om, g);
  t21d = rho * rho * rho;
  t21 = bsv_V3f_div(t21n, t21d);
  t221 = -1.f / (radd * radd);
  t222 = (3 * g) / t21d - f;
  t223 = bsv_V3f_dot(rad, cross_om);
  t22 = bsv_V3f_mult(rad, t221 * t222 * t223);
  t2 = bsv_V3f_plus(t21, t22);
  ret = bsv_V3f_mult(t2, t1);
  ret = bsv_V3f_isequal(self.coord, induced_particle.coord) ? bsv_V3f_zero()
                                                            : ret;
  return ret;
}

template <cvtx_VortFunc VortFunc, int NumPoints>
inline Eigen::Matrix<float, NumPoints, 3> dvort(
    const Eigen::Matrix<float, 1, 3> particleCoord,
    const Eigen::Matrix<float, 1, 3> particleVort,
    const Eigen::Matrix<float, NumPoints, 3> &inducedCoords, 
    const Eigen::Matrix<float, NumPoints, 3> &inducedVorts,
    float regularisationRadius) {
  using column_array_t = Eigen::Array<float, NumPoints, 1>;
  using matrix_t = Eigen::Matrix<float, NumPoints, 3>;
  matrix_t coordDiff = inducedCoords.rowwise() - particleCoord;
  column_array_t radius = coordDiff.rowwise().norm().array();
  column_array_t rho = radius * (1.f / regularisationRadius);
  column_array_t g =
      rho.unaryExpr([](float rho) { return vkernel::g_3D<VortFunc>(rho); });
  column_array_t f =
      rho.unaryExpr([](float rho) { return vkernel::zeta_fn<VortFunc>(rho); });
  matrix_t cross_om = inducedVorts.rowwise().cross(particleVort);
  float t1 = 1.f / (4.f * pi_f * maths::pow<3>(regularisationRadius));
  matrix_t t21 = cross_om.array().colwise() * (g * rho.cube().inverse());
  column_array_t t221 = -1.f / radius.square();
  column_array_t t222 = (3 * g) / rho.cube() - f;
  // Dot product:
  column_array_t t223 = (coordDiff.array() * cross_om.array()).rowwise().sum();
  matrix_t t22 = coordDiff.array().colwise() * (t221 * t222 * t223);
  matrix_t t2 = t21 + t22;
  matrix_t ret = (t2.array() * t1).matrix();
  for (int i{0}; i < ret.cols(); ++i) {
    ret.col(i) = (!(radius > 0)).select(0.f, ret.col(i));
  }
  return ret;
}

template <cvtx_VortFunc VortFunc>
inline bsv_V3f visc_dvort(const cvtx_P3D &self, const cvtx_P3D &induced_particle,
                       float regularisation_radius, float kinematic_visc) {
  bsv_V3f ret, rad, t211, t212, t21, t2;
  float radd, rho, t1, t22;
  rad = bsv_V3f_minus(self.coord, induced_particle.coord);
  radd = bsv_V3f_abs(rad);
  rho = std::abs(radd / regularisation_radius);
  t1 = 2 * kinematic_visc / (regularisation_radius * regularisation_radius);
  t211 = bsv_V3f_mult(self.vorticity, induced_particle.volume);
  t212 = bsv_V3f_mult(induced_particle.vorticity, -1 * self.volume);
  t21 = bsv_V3f_plus(t211, t212);
  t22 = vkernel::eta_3D<VortFunc>(rho);
  t2 = bsv_V3f_mult(t21, t22);
  ret = bsv_V3f_mult(t2, t1);
  ret = bsv_V3f_isequal(self.coord, induced_particle.coord) ? bsv_V3f_zero()
                                                            : ret;
  return ret;
}

template <cvtx_VortFunc VortFunc>
inline bsv_V3f vort(const cvtx_P3D &self, const bsv_V3f mes_point,
                 float regularisation_radius) {
  bsv_V3f rad, ret;
  float radd, coeff, divisor;
  rad = bsv_V3f_minus(self.coord, mes_point);
  radd = bsv_V3f_abs(rad);
  coeff = vkernel::zeta_fn<VortFunc>(radd / regularisation_radius);
  divisor = 4.f * pi_f * regularisation_radius * regularisation_radius *
            regularisation_radius;
  coeff = coeff / divisor;
  ret = bsv_V3f_mult(self.vorticity, coeff);
  return ret;
}
}  // namespace core
}  // namespace cvtx




#endif // CVTX_PARTICLE_CORE_FUNCTIONS_H
