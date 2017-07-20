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

#ifndef TENSORFLOW_KERNELS_OPTICAL_FLOW_TO_HSV_OP_H_
#define TENSORFLOW_KERNELS_OPTICAL_FLOW_TO_HSV_OP_H_

// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS
#include <math.h>
#include <memory>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class OpticalFlowToHSVOp : public OpKernel {
 public:
  explicit OpticalFlowToHSVOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(
        context, context->GetAttr("saturate_magnitude", &saturate_magnitude_));
  }

  void ComputeColor(const float& fx, const float& fy, float* pix) {
    float rad = std::sqrt(fx * fx + fy * fy);
    float ang = std::atan2(fy, fx) / M_PI / 2.0f + 0.5f;
    pix[0] = std::max(std::min(ang, 1.0f), 0.0f);
    pix[1] = 1.0f;
    pix[2] = std::max(std::min(rad, 1.0f), 0.0f);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    // just need to get the size of the input
    int64 batch_size = input.dim_size(0);
    int64 height = input.dim_size(1);
    int64 width = input.dim_size(2);
    OP_REQUIRES(
        context, input.dim_size(3) == 2,
        errors::InvalidArgument("the flow field must have a channel size of 2",
                                input.shape().DebugString()));
    OP_REQUIRES_OK(
        context, context->allocate_output(
                     0, TensorShape({batch_size, height, width, 3}), &output));
    if (!context->status().ok()) return;

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();

    for (int b = 0; b < batch_size; ++b) {
      float max_mag = 0.0;
      if (saturate_magnitude_ > 0.0) {
        max_mag = saturate_magnitude_;
      } else {
        T max_rad = 0.0;
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            T fx = input_data(b, y, x, 0);
            T fy = input_data(b, y, x, 1);
            if (Eigen::numext::isnan(fx) || Eigen::numext::isnan(fy)) {
              continue;
            }
            T rad = fx * fx + fy * fy;
            max_rad = std::max(rad, max_rad);
          }
        }
        if (max_rad == 0.0) max_rad = 1.0;
        max_mag = static_cast<float>(std::sqrt(max_rad));
      }
      // then do the actual computation
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const float fx = input_data(b, y, x, 0);
          const float fy = input_data(b, y, x, 1);
          if (Eigen::numext::isnan(fx) || Eigen::numext::isnan(fy)) {
            output_data(b, y, x, 0) = 0.0;
            output_data(b, y, x, 1) = 0.0;
            output_data(b, y, x, 2) = 0.0;
            continue;
          }
          float pix[3];
          ComputeColor(fx / max_mag, fy / max_mag, pix);
          output_data(b, y, x, 0) = pix[0];
          output_data(b, y, x, 1) = pix[1];
          output_data(b, y, x, 2) = pix[2];
        }
      }
    }
  }

  Tensor* output;

 private:
  float saturate_magnitude_;
};

}  // namespace tensorflow

#endif