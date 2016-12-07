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

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_CORE_KERNELS_ROI_POOLING_OP_GPU_H_
#define TENSORFLOW_CORE_KERNELS_ROI_POOLING_OP_GPU_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

template <typename T>
bool RoiPoolForwardGPU(
    const T* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const T* bottom_rois,
    T* top_data, int* argmax_data, const Eigen::GpuDevice& d);

template <typename T>
bool RoiPoolBackwardGPU(const T* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const T* bottom_rois,
    T* bottom_diff, const int* argmax_data, const Eigen::GpuDevice& d);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_ROI_POOLING_OP_GPU_H_
