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

#define EIGEN_USE_THREADS

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/resample_op_gpu.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

static inline float bicubicCoeffCPU(float x_) {
  float x = fabsf(x_);
  if (x <= 1.0f)     return x * x * (1.5f * x - 2.5f) + 1.0f;
  else if (x < 2.0f) return x * (x * (-0.5f * x + 2.5f) - 4.0f) + 2.0f;
  else               return 0.0f;
}

static inline float triangleCoeffCPU(float x) {
  if (-1.0f <= x && x < 0.0f) return x+1.0f;
  if (0.0f <= x && x <= 1.0f) return 1.0f-x;
  return 0.0f;
}

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class ResampleOp : public OpKernel {
 public:
  explicit ResampleOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("bicubic", &bicubic));
    OP_REQUIRES_OK(context, context->GetAttr("antialias", &antialias));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    // just need to get the size of the input
    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);

    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().DebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().DebugString()));

    auto sizes = shape_t.vec<int32>();
    OP_REQUIRES(context, sizes(0) > 0 && sizes(1) > 0,
                errors::InvalidArgument("shape_t's elements must be positive"));

    // Initialize shape to the batch size of the input, then add
    // the rest of the dimensions
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({batch_size, sizes(0),
                                                          sizes(1), channels}),
                                          &output));

    const int64 out_height = output->dim_size(1);
    const int64 out_width = output->dim_size(2);
    if (!context->status().ok()) return;

    // determine fx, fy and kernel size
    const float fx = static_cast<float>(in_width) / static_cast<float>(out_width);
    const float fy = static_cast<float>(in_height) / static_cast<float>(out_height);

    is_downsample = (fx > 1.0f) || (fy > 1.0f);
    bool do_antialias = is_downsample && antialias;

    if (bicubic) {
      kernel_width = 4;
    } else {
      kernel_width = 2;
    }

    const float ax = 1.0f / (do_antialias ? fx : 1.0f);
    const float ay = 1.0f / (do_antialias ? fy : 1.0f);

    const int64 rx = (fx < 1.0f) ? 2 : ceil(static_cast<float>(kernel_width)/ax);
    const int64 ry = (fy < 1.0f) ? 2 : ceil(static_cast<float>(kernel_width)/ay);

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<T, 4>::Tensor output_data = output->tensor<T, 4>();
    output_data.setZero();
    
    for (int64 y_out = 0; y_out < out_height; ++y_out) {
      const float y_in = y_out * fy + fx / 2.0f - 0.5f;
      const int64 y_in_round = static_cast<int64>(round(y_in));
      for (int64 x_out = 0; x_out < out_width; ++x_out) {
        const float x_in = x_out * fx + fy / 2.0f - 0.5f;
        const int64 x_in_round = static_cast<int64>(round(x_in));
        T wsum = 0;
        for(int64 y = y_in_round-ry; y <= y_in_round+ry; y++) {
          if (y<0 || y>=in_height) continue;
          for(int64 x = x_in_round-rx; x <= x_in_round+rx; x++) {
            if (x<0 || x>=in_width) continue;

            float dx = x_in - x;
            float dy = y_in - y;
            float w;
            if (bicubic)
              w = ax*bicubicCoeffCPU(ax*dx)
                * ay*bicubicCoeffCPU(ay*dy);
            else
              w = ax*triangleCoeffCPU(ax*dx)
                * ay*triangleCoeffCPU(ay*dy);
            wsum += w;
            for (int64 b = 0; b < batch_size; ++b) {
              for (int64 c = 0; c < channels; ++c) {
                output_data(b, y_out, x_out, c) += input_data(b, y, x, c) * w;
              }
            }
          }
        }
        if (wsum) {
          for (int64 b = 0; b < batch_size; ++b) {
            for (int64 c = 0; c < channels; ++c) {
              output_data(b, y_out, x_out, c) /= wsum;
            }
          }
        }
      }
    }
  }

 private:
  bool bicubic;
  bool antialias;
  bool is_downsample;
  int kernel_width;
};

template <typename Device, typename T>
class ResampleOpGrad : public OpKernel {
 public:
  explicit ResampleOpGrad(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("bicubic", &bicubic));
    OP_REQUIRES_OK(context, context->GetAttr("antialias", &antialias));
  }

  void Compute(OpKernelContext* context) override {
    // Grab and validate the input:
    const Tensor& input = context->input(0);
    const Tensor& original = context->input(1);

    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));

    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);

    const int64 out_height = original.dim_size(1);
    const int64 out_width = original.dim_size(2);

    // Initialize shape to the batch size of the input, then add
    // the rest of the dimensions
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({batch_size, out_height,
                                                          out_width, channels}),
                                          &output));
    // determine fx, fy and kernel size
    const float fx = static_cast<float>(out_width) / static_cast<float>(in_width);
    const float fy = static_cast<float>(out_height) / static_cast<float>(in_height);

    is_downsample = (fx > 1.0f) || (fy > 1.0f);
    bool do_antialias = is_downsample && antialias;

    if (bicubic) {
      kernel_width = 4;
    } else {
      kernel_width = 2;
    }

    const float ax = 1.0f / (do_antialias ? fx : 1.0f);
    const float ay = 1.0f / (do_antialias ? fy : 1.0f);

    const int64 rx = (fx < 1.0f) ? 2 : ceil(static_cast<float>(kernel_width)/ax);
    const int64 ry = (fy < 1.0f) ? 2 : ceil(static_cast<float>(kernel_width)/ay);

    // input is the gradient of the top
    typename TTypes<T, 4>::ConstTensor input_diff = input.tensor<T, 4>();
    // output is the gradient of the bottom
    typename TTypes<T, 4>::Tensor output_diff = output->tensor<T, 4>();
    output_diff.setZero();

    for (int64 y_out = 0; y_out < in_height; ++y_out) {
      const float y_in = y_out * fy + fx / 2.0f - 0.5f;
      const int64 y_in_round = static_cast<int>(round(y_in));
      for (int64 x_out = 0; x_out < in_width; ++x_out) {
        const float x_in = x_out * fx + fy / 2.0f - 0.5f;
        const int64 x_in_round = static_cast<int>(round(x_in));
        T wsum = 0;
        for(int64 y = y_in_round-ry; y <= y_in_round+ry; y++) {
          if (y<0 || y>=out_height) continue;
          for(int64 x = x_in_round-rx; x <= x_in_round+rx; x++) {
            if (x<0 || x>=out_width) continue;

            float dx = x_in - x;
            float dy = y_in - y;
            float w;
            // bicubic
            if (bicubic)
              w = ax*bicubicCoeffCPU(ax*dx)
                * ay*bicubicCoeffCPU(ay*dy);
            else
              w = ax*triangleCoeffCPU(ax*dx)
                * ay*triangleCoeffCPU(ay*dy);
            wsum += w;
          }
        }
        if (wsum) {
          for(int64 y = y_in_round-ry; y <= y_in_round+ry; y++) {
            if (y<0 || y>=out_height) continue;
            for(int64 x = x_in_round-rx; x <= x_in_round+rx; x++) {
              if (x<0 || x>=out_width) continue;

              float dx = x_in - x;
              float dy = y_in - y;
              float w;
              // bicubic
              if (bicubic)
                w = ax*bicubicCoeffCPU(ax*dx)
                  * ay*bicubicCoeffCPU(ay*dy) / wsum;
              else
                w = ax*triangleCoeffCPU(ax*dx)
                  * ay*triangleCoeffCPU(ay*dy) / wsum;
              for (int64 b = 0; b < batch_size; ++b) {
                for (int64 c = 0; c < channels; ++c) {
                  output_diff(b, y, x, c) += input_diff(b, y_out, x_out, c) * w;
                }
              }
            }
          }
        }
      }
    }
  }

 private:
  bool bicubic;
  bool antialias;
  bool is_downsample;
  int kernel_width;
};

#define REGISTER_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(Name("Resample")           \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T")             \
                              .HostMemory("size"),                \
                          ResampleOp<CPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("ResampleGrad")       \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T"),             \
                          ResampleOpGrad<CPUDevice, T>);

TF_CALL_float(REGISTER_KERNEL);
TF_CALL_double(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#if GOOGLE_CUDA

template <typename T>
class ResampleGPUOp : public OpKernel {
 public:
  explicit ResampleGPUOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("bicubic", &bicubic));
    OP_REQUIRES_OK(context, context->GetAttr("antialias", &antialias));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    // just need to get the size of the input
    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);

    const Tensor& shape_t = context->input(1);
    OP_REQUIRES(context, shape_t.dims() == 1,
                errors::InvalidArgument("shape_t must be 1-dimensional",
                                        shape_t.shape().DebugString()));
    OP_REQUIRES(context, shape_t.NumElements() == 2,
                errors::InvalidArgument("shape_t must have two elements",
                                        shape_t.shape().DebugString()));

    auto sizes = shape_t.vec<int32>();
    OP_REQUIRES(context, sizes(0) > 0 && sizes(1) > 0,
                errors::InvalidArgument("shape_t's elements must be positive"));

    // Initialize shape to the batch size of the input, then add
    // the rest of the dimensions
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({batch_size, sizes(0),
                                                          sizes(1), channels}),
                                          &output));

    const int64 out_height = output->dim_size(1);
    const int64 out_width = output->dim_size(2);
    if (!context->status().ok()) return;

    // determine fx, fy and kernel size
    fx = static_cast<float>(in_width) / static_cast<float>(out_width);
    fy = static_cast<float>(in_height) / static_cast<float>(out_height);
    is_downsample = (fx > 1.0f) || (fy > 1.0f);
    bool do_antialias = is_downsample && antialias;

    if (bicubic) {
      kernel_width = 4;
    } else {
      kernel_width = 2;
    }

    bool status = Resample<T>(
        input.flat<T>().data(), batch_size, in_height, in_width,
        channels, out_height, out_width, output->flat<T>().data(),
        fx, fy, kernel_width, do_antialias,
        context->eigen_gpu_device());

    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching Resample"));
    }
  }
 private:
  bool bicubic;
  bool antialias;
  bool is_downsample;
  float fx, fy;
  int kernel_width;
};

#define REGISTER_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(Name("Resample")           \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T")             \
                              .HostMemory("size"),                \
                          ResampleGPUOp<T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_KERNEL);

#undef REGISTER_KERNEL

template <typename T>
class ResampleGPUOpGrad : public OpKernel {
 public:
  explicit ResampleGPUOpGrad(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("bicubic", &bicubic));
    OP_REQUIRES_OK(context, context->GetAttr("antialias", &antialias));
  }

  void Compute(OpKernelContext* context) override {
    // Grab and validate the input:
    const Tensor& input = context->input(0);
    const Tensor& original = context->input(1);

    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));

    const int64 batch_size = input.dim_size(0);
    const int64 in_height = input.dim_size(1);
    const int64 in_width = input.dim_size(2);
    const int64 channels = input.dim_size(3);

    const int64 out_height = original.dim_size(1);
    const int64 out_width = original.dim_size(2);

    // Initialize shape to the batch size of the input, then add
    // the rest of the dimensions
    Tensor* output = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, TensorShape({batch_size, out_height,
                                                          out_width, channels}),
                                          &output));
    // determine fx, fy and kernel size
    fx = static_cast<float>(out_width) / static_cast<float>(in_width);
    fy = static_cast<float>(out_height) / static_cast<float>(in_height);
    is_downsample = (fx > 1.0f) || (fy > 1.0f);
    bool do_antialias = is_downsample && antialias;

    if (bicubic) {
      kernel_width = 4;
    } else {
      kernel_width = 2;
    }

    bool status = ResampleBackward(
        input.flat<T>().data(), batch_size, in_height,
        in_width, channels, out_height, out_width, output->flat<T>().data(),
        fx, fy, kernel_width, do_antialias,
        context->eigen_gpu_device());

    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching ResampleGrad"));
    }
  }
  bool bicubic;
  bool antialias;
  bool is_downsample;
  float fx, fy;
  int kernel_width;
};

#define REGISTER_KERNEL(T)                                           \
  REGISTER_KERNEL_BUILDER(Name("ResampleGrad")          \
                            .Device(DEVICE_GPU)                      \
                            .TypeConstraint<T>("T"),                  \
                          ResampleGPUOpGrad<T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#endif  // GOOGLE_CUDA

} // namespace tensorflow