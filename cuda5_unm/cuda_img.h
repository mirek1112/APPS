// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************

#pragma once

#include <opencv2/core/mat.hpp>

// Structure definition for exchanging data between Host and Device
struct CudaImg
{
  uint3 m_size;             // size of picture
  union {
      void   *m_p_void;     // data of picture
      uchar1 *m_p_uchar1;   // data of picture
      uchar3 *m_p_uchar3;   // data of picture
      uchar4 *m_p_uchar4;   // data of picture
  };

  // Access uchar3 pixel
  __device__ __host__ uchar3& at(int y, int x) {
    return m_p_uchar3[y * m_size.x + x];
  }

  // Access uchar4 pixel (e.g. with alpha channel)
  template<typename T>
  __device__ __host__ T& at(int y, int x);
};

// Template specialization for uchar4
template<>
__device__ __host__ inline uchar4& CudaImg::at<uchar4>(int y, int x) {
    return m_p_uchar4[y * m_size.x + x];
}
