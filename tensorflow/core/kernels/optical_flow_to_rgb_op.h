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

// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS
#include <math.h>
#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// just define the flow field related function here
class ColorCode {
private:
  int ncols;
  int colorwheel[60][3];

  void setcols(int r, int g, int b, int k)
  {
      colorwheel[k][0] = r;
      colorwheel[k][1] = g;
      colorwheel[k][2] = b;
  }

  void makecolorwheel()
  {
      int RY = 15;
      int YG = 6;
      int GC = 4;
      int CB = 11;
      int BM = 13;
      int MR = 6;
      ncols = RY + YG + GC + CB + BM + MR;
      int i;
      int k = 0;
      for (i = 0; i < RY; i++) setcols(255,    255*i/RY,   0,        k++);
      for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,    0,        k++);
      for (i = 0; i < GC; i++) setcols(0,      255,    255*i/GC,     k++);
      for (i = 0; i < CB; i++) setcols(0,      255-255*i/CB, 255,        k++);
      for (i = 0; i < BM; i++) setcols(255*i/BM,     0,    255,        k++);
      for (i = 0; i < MR; i++) setcols(255,    0,    255-255*i/MR, k++);
  }

public:
  ColorCode() {
    ncols = 0;
    makecolorwheel();
  }

  void ComputeColor(const float& fx, const float& fy, float *pix)
  {
    float rad = std::sqrt(fx * fx + fy * fy);
    float a = std::atan2(-fy, -fx) / M_PI;
    float fk = (a + 1.0) / 2.0 * (ncols-1);
    int k0 = static_cast<int>(fk);
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;
    for (int b = 0; b < 3; b++) {
      float col0 = static_cast<float>(colorwheel[k0][b]) / 255.0;
      float col1 = static_cast<float>(colorwheel[k1][b]) / 255.0;
      float col = (1 - f) * col0 + f * col1;
      if (rad <= 1)
        col = 1.0 - rad * (1.0 - col); // increase saturation with radius
      else
        col *= .75; // out of range
      pix[2 - b] = col;
    }
  }  
};

template <typename Device, typename T>
class OpticalFlowToRGBOp : public OpKernel {
 public:
  explicit OpticalFlowToRGBOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("saturate_magnitude", &saturate_magnitude_));
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
    OP_REQUIRES(context, input.dim_size(3) == 2,
                errors::InvalidArgument("the flow field must have a channel size of 2",
                                        input.shape().DebugString()));
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, TensorShape({batch_size, height,
                                                width, 3}),
                                &output));
    if (!context->status().ok()) return;

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();

    ColorCode color_code = ColorCode();
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
            if (isnan(fx) || isnan(fy)) {
              continue;
            }
            T rad = fx * fx + fy * fy;
            max_rad = std::max(rad, max_rad);
          }
        }
        if (max_rad == 0.0)
          max_rad = 1.0;
        max_mag = static_cast<float>(std::sqrt(max_rad));
      }
      // then do the actual computation
      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          const float fx = input_data(b, y, x, 0);
          const float fy = input_data(b, y, x, 1);
          if (isnan(fx) || isnan(fy)) {
            output_data(b, y, x, 0) = 0.0;
            output_data(b, y, x, 1) = 0.0;
            output_data(b, y, x, 2) = 0.0;
            continue;
          }
          float pix[3];
          color_code.ComputeColor(fx/max_mag, fy/max_mag, pix);
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
