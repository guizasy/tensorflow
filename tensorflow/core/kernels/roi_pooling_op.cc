/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <stdio.h>

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/roi_pooling_op_gpu.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

using std::min;
using std::max;

template <typename Device, typename T>
class RoiPoolOp : public OpKernel {
 public:
  explicit RoiPoolOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    auto bottom_data_flat = bottom_data.flat<T>();
    auto bottom_rois_flat = bottom_rois.flat<T>();

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // Number of ROIs
    const int num_rois = bottom_rois.dim_size(0);
    // batch size
    const int batch_size = bottom_data.dim_size(0);
    // data height
    const int data_height = bottom_data.dim_size(1);
    // data width
    const int data_width = bottom_data.dim_size(2);
    // Number of channels
    const int num_channels = bottom_data.dim_size(3);

    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[1] = pooled_height_;
    dims[2] = pooled_width_;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    // Create output tensors
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    Tensor* argmax_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &argmax_tensor));
    auto argmax = argmax_tensor->flat<int>();

    // Set all element of the output tensor to -inf.
    // Here the assumption is that this is always d1 with 
    const int N = output.size();
    for (int i = 0; i < N; i++) {
      output(i) = -std::numeric_limits<T>::infinity();
      argmax(i) = -1;
    }

    // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    int index_roi = 0;
    int index_output = 0;
    for (int n = 0; n < num_rois; ++n) {
      int roi_batch_ind = static_cast<int>(bottom_rois_flat(index_roi));
      int roi_start_w = static_cast<int>(round(bottom_rois_flat(index_roi + 1) * spatial_scale_));
      int roi_start_h = static_cast<int>(round(bottom_rois_flat(index_roi + 2) * spatial_scale_));
      int roi_end_w = static_cast<int>(round(bottom_rois_flat(index_roi + 3) * spatial_scale_));
      int roi_end_h = static_cast<int>(round(bottom_rois_flat(index_roi + 4) * spatial_scale_));
      CHECK_GE(roi_batch_ind, 0);
      CHECK_LT(roi_batch_ind, batch_size);

      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      const T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height_);
      const T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width_);

      const int index_data = roi_batch_ind * data_height * data_width * num_channels;

      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), data_height);
          hend = min(max(hend + roi_start_h, 0), data_height);
          wstart = min(max(wstart + roi_start_w, 0), data_width);
          wend = min(max(wend + roi_start_w, 0), data_width);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = index_output + (ph * pooled_width_ + pw) * num_channels;
          if (is_empty) {
            continue;
          }

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              for (int c = 0; c < num_channels; ++c) {
                const int index = (h * data_width + w) * num_channels + c;
                if (bottom_data_flat(index_data + index) > output(pool_index + c)) {
                  output(pool_index + c) = bottom_data_flat(index_data + index);
                  argmax(pool_index + c) = static_cast<int>(index);
                }
              }
            }
          }
        }
      }
      // Increment ROI index
      index_roi += bottom_rois.dim_size(1);
      index_output += pooled_height_ * pooled_width_ * num_channels;
    }
  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

template <typename Device, class T>
class RoiPoolOpGrad : public OpKernel {
 public:
  explicit RoiPoolOpGrad(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    const Tensor& argmax_data = context->input(2);
    const Tensor& out_backprop = context->input(3);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    OP_REQUIRES(context, argmax_data.dims() == 4,
                errors::InvalidArgument("argmax_data must be 4-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    auto bottom_rois_flat = bottom_rois.flat<T>();
    auto argmax_flat = argmax_data.flat<int>();
    auto out_backprop_flat = out_backprop.flat<T>();

    // Number of ROIs
    const int num_rois = bottom_rois.dim_size(0);
    // batch size
    const int batch_size = bottom_data.dim_size(0);
    // data height
    const int data_height = bottom_data.dim_size(1);
    // data width
    const int data_width = bottom_data.dim_size(2);
    // Number of channels
    const int num_channels = bottom_data.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto output_diff = output->flat<T>();

    // set everthing to zero
    const int N = output_diff.size();
    for (int i = 0; i < N; i++) {
      output_diff(i) = 0;
    }

    // For each ROI R = [batch_index x1 y1 x2 y2]: max pool over R
    int index_roi = 0;
    int index_output = 0;
    for (int n = 0; n < num_rois; ++n) {
      int roi_batch_ind = static_cast<int>(bottom_rois_flat(index_roi + 0));
      int roi_start_w = static_cast<int>(round(bottom_rois_flat(index_roi + 1) * spatial_scale_));
      int roi_start_h = static_cast<int>(round(bottom_rois_flat(index_roi + 2) * spatial_scale_));
      int roi_end_w = static_cast<int>(round(bottom_rois_flat(index_roi + 3) * spatial_scale_));
      int roi_end_h = static_cast<int>(round(bottom_rois_flat(index_roi + 4) * spatial_scale_));
      CHECK_GE(roi_batch_ind, 0);
      CHECK_LT(roi_batch_ind, batch_size);

      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      const T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height_);
      const T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width_);

      const int index_data = roi_batch_ind * data_height * data_width * num_channels;

      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height_)
          int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), data_height);
          hend = min(max(hend + roi_start_h, 0), data_height);
          wstart = min(max(wstart + roi_start_w, 0), data_width);
          wend = min(max(wend + roi_start_w, 0), data_width);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          if (is_empty) {
            continue;
          }

          const int pool_index = index_output + (ph * pooled_width_ + pw) * num_channels;
          for (int c = 0; c < num_channels; ++c) {
            const int index = static_cast<int>(argmax_flat(pool_index + c));
            output_diff(index_data + index) += out_backprop_flat(pool_index + c);
          }
        }
      }
      // Increment ROI index
      index_roi += bottom_rois.dim_size(1);
      index_output += pooled_height_ * pooled_width_ * num_channels;
    }
  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

#define REGISTER_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(Name("RoiPooling")                      \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T"),            \
                          RoiPoolOp<CPUDevice, T>);               \
  REGISTER_KERNEL_BUILDER(Name("RoiPoolingGrad")                  \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T"),            \
                          RoiPoolOpGrad<CPUDevice, T>);

TF_CALL_float(REGISTER_KERNEL);
TF_CALL_double(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#if GOOGLE_CUDA

template <class T>
class RoiPoolGPUOp : public OpKernel {
 public:
  explicit RoiPoolGPUOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // Number of ROIs
    const int num_rois = bottom_rois.dim_size(0);
    // data height
    const int data_height = bottom_data.dim_size(1);
    // data width
    const int data_width = bottom_data.dim_size(2);
    // Number of channels
    const int num_channels = bottom_data.dim_size(3);

    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[1] = pooled_height_;
    dims[2] = pooled_width_;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    // Create output tensors
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
    auto output = output_tensor->flat<T>();

    Tensor* argmax_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, output_shape, &argmax_tensor));
    auto argmax = argmax_tensor->flat<int>();

    bool status = RoiPoolForwardGPU(
        bottom_data.flat<T>().data(), spatial_scale_, num_rois, data_height,
        data_width, num_channels, pooled_height_, pooled_width_, bottom_rois.flat<T>().data(),
        output.data(), argmax.data(), context->eigen_gpu_device());

    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching Roi pooling layer"));
    }
  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

#define REGISTER_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(Name("RoiPooling")                      \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T"),            \
                          RoiPoolGPUOp<T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_KERNEL);

#undef REGISTER_KERNEL

template <class T>
class RoiPoolGPUOpGrad : public OpKernel {
 public:
  explicit RoiPoolGPUOpGrad(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_height", &pooled_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, pooled_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        pooled_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("pooled_width", &pooled_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, pooled_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        pooled_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& bottom_data = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    const Tensor& argmax_data = context->input(2);
    const Tensor& out_backprop = context->input(3);

    // data should have 4 dimensions.
    OP_REQUIRES(context, bottom_data.dims() == 4,
                errors::InvalidArgument("data must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    OP_REQUIRES(context, argmax_data.dims() == 4,
                errors::InvalidArgument("argmax_data must be 4-dimensional"));

    OP_REQUIRES(context, out_backprop.dims() == 4,
                errors::InvalidArgument("out_backprop must be 4-dimensional"));

    // Number of ROIs
    const int num_rois = bottom_rois.dim_size(0);
    // batch size
    const int batch_size = bottom_data.dim_size(0);
    // data height
    const int data_height = bottom_data.dim_size(1);
    // data width
    const int data_width = bottom_data.dim_size(2);
    // Number of channels
    const int channels = bottom_data.dim_size(3);

    // construct the output shape
    TensorShape output_shape = bottom_data.shape();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    bool status = RoiPoolBackwardGPU(
      out_backprop.flat<T>().data(), spatial_scale_, batch_size, num_rois, data_height,
      data_width, channels, pooled_height_, pooled_width_, bottom_rois.flat<T>().data(),
      output->flat<T>().data(), argmax_data.flat<int>().data(), context->eigen_gpu_device());

    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching RoiPoolingGrad"));
    }
  }
 private:
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

#define REGISTER_KERNEL(T)                                           \
  REGISTER_KERNEL_BUILDER(Name("RoiPoolingGrad")                     \
                            .Device(DEVICE_GPU)                      \
                            .TypeConstraint<T>("T"),                 \
                          RoiPoolGPUOpGrad<T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#endif  // GOOGLE_CUDA

} // namespace tensorflow