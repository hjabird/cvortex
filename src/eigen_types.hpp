#include "libcvtx.h"
/*============================================================================
VortFunc.c

Fast good-enough maths functions.

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
#ifndef CVTX_EIGEN_TYPES_H
#define CVTX_EIGEN_TYPES_H

#include <Eigen/Dense>
#include <tuple>

namespace cvtx {
namespace detail {

/** Get a value rounded up to the nearest BlockSize multiple.
* Eg. padded_size<4>(4) == 4; padded_size<4>(5) == 8.
* padded_size<8>(35) == 40, padded_size<8>(41) == 48.
* @param BlockSize the multiple.
**/
template<int BlockSize>
constexpr int padded_size(const int n) {
  return (n / BlockSize + (n % BlockSize ? 1 : 0)) * BlockSize;
}

/** An array based representation of Vortex particles.
* @tparam Dims is the dimensionality of the problem.
* @tparam DataOrder is the data ordering of the arrays.
**/
template <int Dims>
struct particle_array;

template <>
struct particle_array<2> {
  Eigen::Matrix<float, Eigen::Dynamic, 2> coords;
  Eigen::Matrix<float, Eigen::Dynamic, 1> vorts;
  Eigen::Matrix<float, Eigen::Dynamic, 1> volumes;
};

template <>
struct particle_array<3> {
  Eigen::Matrix<float, Eigen::Dynamic, 3> coords;
  Eigen::Matrix<float, Eigen::Dynamic, 3> vorts;
  Eigen::Matrix<float, Eigen::Dynamic, 1> volumes;
};

/// Is a type bsv_V2f or V3f? 
template <class T>
struct is_bsv_vec : public std::false_type {};
template<>
struct is_bsv_vec<bsv_V2f> : public std::true_type {};
template <>
struct is_bsv_vec<bsv_V3f> : public std::true_type {};

/// Is a cvortex particle type?
template <class T>
struct is_cvortex_particle : public std::false_type {};
template <>
struct is_cvortex_particle<cvtx_P2D> : public std::true_type {};
template <>
struct is_cvortex_particle<cvtx_P3D> : public std::true_type {};

/// Dimensionality of type
template <typename T>
struct native_dimensions {};
template <>
struct native_dimensions<cvtx_P2D> {
  static constexpr int dims = 2;
};
template <>
struct native_dimensions<cvtx_P3D> {
  static constexpr int dims = 3;
};
template <>
struct native_dimensions<bsv_V2f> {
  static constexpr int dims = 2;
};
template <>
struct native_dimensions<bsv_V3f> {
  static constexpr int dims = 3;
};

template<typename T>
struct eigen_equiv;

template <>
struct eigen_equiv<float*> {
  using type = Eigen::Matrix<float, Eigen::Dynamic, 1>;
};

template <>
struct eigen_equiv<bsv_V2f*> {
  using type = Eigen::Matrix<float, Eigen::Dynamic, 2>;
};

template<>
struct eigen_equiv<bsv_V3f*> {
  using type = Eigen::Matrix<float, Eigen::Dynamic, 3>;
};

template <>
struct eigen_equiv<cvtx_P2D*> {
  using type = particle_array<2>;
};

template <>
struct eigen_equiv<cvtx_P3D*> {
  using type = particle_array<3>;
};

/** Convert a type to its eigen equivalent array type.
* 
* @tparam BlockSize Make the output array a multiple of BlockSize.
* @tparam T is the type that is becoming eigen-ified.
* @tparam Stride is the stride of the data in bytes.
**/
// template <int BlockSize = 1, typename T, size_t Stride = sizeof(T)>
// inline typename eigen_equiv<T*>::type to_eigen(const T*, const int);

/// Specialization for bsv_V2f and bsv_V3f.
template <int BlockSize = 1, typename T, size_t Stride = sizeof(T)>
inline typename std::enable_if_t<is_bsv_vec<T>::value,
                                 typename eigen_equiv<T*>::type>
to_eigen(
    const T* arr, const int n) {
  int nRows = padded_size<BlockSize>(n);
  static constexpr int dims = native_dimensions<T>::dims;
  typename eigen_equiv<T*>::type mat(nRows, dims);
  const char* voidArr = reinterpret_cast<const char*>(arr);
  for (int i{0}; i < n; ++i) {
    for (int j{0}; j < dims; ++j) {
      mat(i, j) = reinterpret_cast<const T*>(voidArr + i * Stride)->x[j];
    }
  }
  for (int i{n}; i < nRows; ++i) {
    for (int j{0}; j < dims; ++j) {
      mat(i, j) = 0.f;
    }
  }
    return mat;
}

/// Specialization for floats.
template <int BlockSize = 1, size_t Stride = sizeof(float)>
inline typename eigen_equiv<float*>::type
to_eigen(const float* arr, const int n) {
  int nRows = padded_size<BlockSize>(n);
  typename eigen_equiv<float*>::type mat(nRows, 1);
  const char* voidArr = reinterpret_cast<const char*>(arr);
  for (int i{0}; i < n; ++i) {
      mat(i) = *reinterpret_cast<const float*>(voidArr + i * Stride);
  }
  for (int i{n}; i < nRows; ++i) {
      mat(i) = 0.f;
  }
  return mat;
}

/// Specialization for cvtx_P2D
template <int BlockSize = 1, size_t Stride = sizeof(cvtx_P2D)>
inline particle_array<2> to_eigen(const cvtx_P2D* arr, const int n) {
  particle_array<2> ret;
  const char* voidArr{reinterpret_cast<const char*>(arr)};
  ret.coords = to_eigen<BlockSize, bsv_V2f, Stride>(reinterpret_cast<const bsv_V2f*>(voidArr), n);
  ret.vorts = to_eigen<BlockSize, Stride>(
      reinterpret_cast<const float*>(voidArr + offsetof(cvtx_P2D, vorticity)), n);
  ret.volumes = to_eigen<BlockSize, Stride>(
      reinterpret_cast<const float*>(voidArr + offsetof(cvtx_P2D, area)), n);
  return ret;
}

template <int BlockSize = 1, size_t Stride = sizeof(cvtx_P3D)>
inline particle_array<3> to_eigen(const cvtx_P3D* arr, const int n) {
  particle_array<3> ret;
  const char* voidArr{reinterpret_cast<const char*>(arr)};
  ret.coords = to_eigen<BlockSize, bsv_V3f, Stride>(reinterpret_cast<const bsv_V3f*>(voidArr), n);
  ret.vorts = to_eigen<BlockSize, bsv_V3f, Stride>(
      reinterpret_cast<const bsv_V3f*>(voidArr + offsetof(cvtx_P3D, vorticity)), n);
  ret.volumes =
      to_eigen<BlockSize, Stride>(
      reinterpret_cast<const float*>(voidArr + offsetof(cvtx_P3D, volume)), n);
  return ret;
}

template <int BlockSize = 1>
inline auto to_eigen(const cvtx_F3D* arr, const int n) {
  int nRows = padded_size<BlockSize>(n);
    Eigen::Matrix<float, Eigen::Dynamic, 3> starts(nRows, 3);
    Eigen::Matrix<float, Eigen::Dynamic, 3> ends(nRows, 3);
    Eigen::Matrix<float, Eigen::Dynamic, 1> strengths(nRows);
    for (int i{0}; i < n; ++i) {
        for (int j{0}; j < 3; ++j) {
            starts(i, j) = arr[i].start.x[j];
            ends(i, j) = arr[i].end.x[j];
        }
        strengths(i) = arr[i].strength;
    }
    for (int i{n}; i < nRows; ++i) {
      for (int j{0}; j < 3; ++j) {
        starts(i, j) = 0.f;
        ends(i, j) = 1.f;
      }
      strengths(i) = 0.f;
    }
    return std::make_tuple(starts, ends, strengths);
}

inline void to_array(Eigen::Matrix<float, Eigen::Dynamic, 3>& input,
                bsv_V3f* arr, const int n) {
    assert(input.rows() >= n);
    for (int i{0}; i < n; ++i) {
        for (int j{0}; j < 3; ++j) {
            arr[i].x[j] = input(i, j);
        }
    }
}

inline void to_array(Eigen::Matrix<float, Eigen::Dynamic, 2>& input,
                bsv_V2f* arr, const int n) {
    assert(input.rows() >= n);
    for (int i{0}; i < n; ++i) {
        for (int j{0}; j < 2; ++j) {
            arr[i].x[j] = input(i, j);
        }
    }
}

inline void to_array(Eigen::Matrix<float, Eigen::Dynamic, 1>& input,
                float* arr, const int n) {
    assert(input.rows() >= n);
    for (int i{0}; i < n; ++i) {
        arr[i] = input(i);
    }
}

} // detail
} // cvtx
#endif  // CVTX_EIGEN_TYPES_H
