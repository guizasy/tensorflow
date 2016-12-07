/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>

#include "tensorflow/core/kernels/resample_op_gpu.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace {

static __device__ __forceinline__ float bicubicCoeff(float x_) {
  float x = fabsf(x_);
  if (x <= 1.0f)     return x * x * (1.5f * x - 2.5f) + 1.0f;
  else if (x < 2.0f) return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
  else               return 0.0f;
}

static __device__ __forceinline__ float triangleCoeff(float x) {
  if (-1.0f <= x && x < 0.0f) return x+1.0f;
  if (0.0f <= x && x <= 1.0f) return 1.0f-x;
  return 0.0f;
}

template <typename T>
__global__ void ResampleNHWC(const int nthreads, const T* bottom_data,
                                          const int in_height, const int in_width,
                                          const int channels, const int out_height,
                                          const int out_width, T* top_data,
                                          const float fx, const float fy,
                                          const float ax, const float ay,
                                          const int rx, const int ry,
                                          const int kernel_width, const int bottom_offset,
                                          const bool antialias) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // first get the location
    int n = index;
    int c = n % channels;
    n /= channels;
    int out_x = n % out_width;
    n /= out_width;
    int out_y = n % out_height;
    n /= out_height;

    float x_in = out_x * fx + fy / 2.0f - 0.5f;
    float y_in = out_y * fy + fx / 2.0f - 0.5f;

    int x_in_round = round(x_in);
    int y_in_round = round(y_in);

    T sum = 0.0;
    T wsum = 0.0;
    const T* bottom_data_n = bottom_data + n * bottom_offset;

    for(int y = y_in_round-ry; y <= y_in_round+ry; y++) {
      if (y<0 || y>=in_height) continue;
      for(int x = x_in_round-rx; x <= x_in_round+rx; x++) {
        if (x<0 || x>=in_width) continue;

        float dx = x_in - x;
        float dy = y_in - y;
        float w;
        // bicubic
        if (kernel_width == 4)
          w = ax*bicubicCoeff(ax*dx)
            * ay*bicubicCoeff(ay*dy);
        else
          w = ax*triangleCoeff(ax*dx)
            * ay*triangleCoeff(ay*dy);
        const int idx = (y * in_width + x) * channels + c;
        sum += w * ldg(bottom_data_n + idx);
        wsum += w;
      }
    }
    top_data[index] = (!wsum) ? 0.0 : (sum / wsum);
  }
}

template <typename T>
__global__ void ResampleBackwardNHWC(
                                   const int nthreads, const T* top_diff,
                                   const int in_height, const int in_width,
                                   const int channels, const int out_height,
                                   const int out_width, T* bottom_diff,
                                   const float fx, const float fy,
                                   const float ax, const float ay,
                                   const int rx, const int ry,
                                   const int kernel_width, const int bottom_offset,
                                   const bool antialias) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int in_x = n % in_width;
    n /= in_width;
    int in_y = n % in_height;
    n /= in_height;

    float x_in = in_x * fx + fy / 2.0f - 0.5f;
    float y_in = in_y * fy + fx / 2.0f - 0.5f;

    int x_in_round = round(x_in);
    int y_in_round = round(y_in);

    T wsum = 0.0;
    T* bottom_diff_n = bottom_diff + n * bottom_offset;

    // two for loops, because needs to compute the sum
    for(int y = y_in_round-ry; y <= y_in_round+ry; y++) {
      if (y<0 || y>=out_height) continue;
      for(int x = x_in_round-rx; x <= x_in_round+rx; x++) {
        if (x<0 || x>=out_width) continue;

        float dx = x_in - x;
        float dy = y_in - y;
        float w;
        // bicubic
        if (kernel_width == 4)
          w = ax*bicubicCoeff(ax*dx)
            * ay*bicubicCoeff(ay*dy);
        else
          w = ax*triangleCoeff(ax*dx)
            * ay*triangleCoeff(ay*dy);
        wsum += w;
      }
    }

    if (wsum) {
      for(int y = y_in_round-ry; y <= y_in_round+ry; y++) {
        if (y<0 || y>=out_height) continue;
        for(int x = x_in_round-rx; x <= x_in_round+rx; x++) {
          if (x<0 || x>=out_width) continue;

          float dx = x_in - x;
          float dy = y_in - y;
          float w;
          // bicubic
          if (kernel_width == 4)
            w = ax*bicubicCoeff(ax*dx)
              * ay*bicubicCoeff(ay*dy);
          else
            w = ax*triangleCoeff(ax*dx)
              * ay*triangleCoeff(ay*dy);
          const int idx = (y * out_width + x) * channels + c;
          T wp = w / wsum;
          CudaAtomicAdd(bottom_diff_n + idx, wp * ldg(top_diff + index));
        }
      }
    }
  }
}

}  // namespace

template <typename T>
bool Resample(const T* bottom_data, const int batch,
                           const int in_height, const int in_width,
                           const int channels, const int out_height,
                           const int out_width, T* top_data,
                           const float fx, const float fy,
                           const int kernel_width, const bool antialias,
                           const Eigen::GpuDevice& d) {
  const int output_size = batch * channels * out_height * out_width;
  CudaLaunchConfig config = GetCudaLaunchConfig(output_size, d);

  float ax = 1.0f / (antialias ? fx : 1.0f);
  float ay = 1.0f / (antialias ? fy : 1.0f);
  int rx = (fx < 1.0f) ? 2 : ceil(static_cast<float>(kernel_width)/ax);
  int ry = (fy < 1.0f) ? 2 : ceil(static_cast<float>(kernel_width)/ay);
  const int bottom_offset = in_height * in_width * channels;

  ResampleNHWC<T>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
      output_size, bottom_data, in_height, in_width, channels, out_height,
      out_width, top_data, fx, fy, ax, ay, rx, ry, kernel_width, bottom_offset, antialias);
  return d.ok();
}

#define DECLARE_GPU_SPEC(T)                                                        \
  template bool Resample(const T* bottom_data, const int batch,       \
                               const int in_height, const int in_width,            \
                               const int channels, const int out_height,           \
                               const int out_width, T* top_data,               \
                               const float fx, const float fy,                 \
                               const int kernel_width, const bool antialias,   \
                               const Eigen::GpuDevice& d);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

template <typename T>
bool ResampleBackward(const T* top_diff, const int batch,
                                   const int in_height, const int in_width,
                                   const int channels, const int out_height,
                                   const int out_width, T* bottom_diff,
                                   const float fx, const float fy,
                                   const int kernel_width, const bool antialias,
                                   const Eigen::GpuDevice& d) {
  const int output_size = batch * channels * out_height * out_width;
  CudaLaunchConfig output_config = GetCudaLaunchConfig(output_size, d);
  SetZero<<<output_config.block_count,
            output_config.thread_per_block, 0, d.stream()>>>(output_size, bottom_diff);

  float ax = 1.0f / (antialias ? fx : 1.0f);
  float ay = 1.0f / (antialias ? fy : 1.0f);
  int rx = (fx < 1.0f) ? 2 : ceil(static_cast<float>(kernel_width)/ax);
  int ry = (fy < 1.0f) ? 2 : ceil(static_cast<float>(kernel_width)/ay);
  const int bottom_offset = output_size / batch;

  const int input_size = batch * channels * in_height * in_width;
  CudaLaunchConfig input_config = GetCudaLaunchConfig(input_size, d);
  ResampleBackwardNHWC<T><<<
      input_config.block_count, input_config.thread_per_block, 0, d.stream()>>>(
      input_config.virtual_thread_count, top_diff, in_height, in_width,
      channels, out_height, out_width, bottom_diff, 
      fx, fy, ax, ay, rx, ry, kernel_width, bottom_offset, antialias);
  return d.ok();
}

#define DECLARE_GPU_SPEC(T)                                                           \
  template bool ResampleBackward(const T* top_diff, const int batch,     \
                               const int in_height, const int in_width,               \
                               const int channels, const int out_height,              \
                               const int out_width, T* bottom_diff,               \
                               const float fx, const float fy,                    \
                               const int kernel_width, const bool antialias,      \
                               const Eigen::GpuDevice& d);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
