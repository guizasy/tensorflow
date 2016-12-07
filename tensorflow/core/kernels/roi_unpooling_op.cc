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
#include "tensorflow/core/kernels/roi_unpooling_op_gpu.h"
#endif  // GOOGLE_CUDA

// roi unpooling layer, here for one bit, just do bilinear interpolation
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

using std::min;
using std::max;

template <typename Device, class T>
class RoiUnpoolOp : public OpKernel {
 public:
  explicit RoiUnpoolOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("data_height", &data_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, data_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        data_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("data_width", &data_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, data_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        data_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));

    // Get the batch size
    OP_REQUIRES_OK(context,
                   context->GetAttr("batch_size", &batch_size_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    // bottom feat is the new feature to put it
    const Tensor& bottom_feat = context->input(0);
    const Tensor& bottom_rois = context->input(1);

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // bottom feature should have 4 dimensions
    OP_REQUIRES(context, bottom_feat.dims() == 4,
                errors::InvalidArgument("bottom feat must be 4-dimensional"));

    // Number of ROIs
    const int num_rois = bottom_rois.dim_size(0);
    // pool height
    const int pooled_height = bottom_feat.dim_size(1);
    // pool width
    const int pooled_width = bottom_feat.dim_size(2);
    // Number of channels
    const int num_channels = bottom_feat.dim_size(3);

    // construct the output shape
    int dims[4];
    dims[0] = batch_size_;
    dims[1] = data_height_;
    dims[2] = data_width_;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    Tensor* top_data = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data));

    auto bottom_rois_flat = bottom_rois.flat<T>();
    auto bottom_feat_flat = bottom_feat.flat<T>();
    auto top_data_flat = top_data->flat<T>();

    // copy the values from bottom data first
    // const int N = top_data_flat.size();
    // for (int i = 0; i < N; i++) {
    //   top_data_flat(i) = 0;
    // }
    top_data->tensor<T, 4>().setZero();

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
      CHECK_LT(roi_batch_ind, batch_size_);

      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      const T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      const T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

      const int index_data = roi_batch_ind * data_height_ * data_width_ * num_channels;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height)
          int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), data_height_);
          hend = min(max(hend + roi_start_h, 0), data_height_);
          wstart = min(max(wstart + roi_start_w, 0), data_width_);
          wend = min(max(wend + roi_start_w, 0), data_width_);

          T this_size = static_cast<T>((hend - hstart) * (wend - wstart));
          bool is_empty = (hend <= hstart) || (wend <= wstart);

          if (is_empty) {
            continue;
          }

          const int pool_index = index_output + (ph * pooled_width + pw) * num_channels;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              for (int c = 0; c < num_channels; ++c) {
                const int index = (h * data_width_ + w) * num_channels + c;
                top_data_flat(index_data + index) += bottom_feat_flat(pool_index + c) / this_size;
              }
            }
          }
        }
      }
      // Increment ROI index
      index_roi += bottom_rois.dim_size(1);
      index_output += pooled_height * pooled_width * num_channels;
    }
  }
 private:
  int data_height_;
  int data_width_;
  float spatial_scale_;
  int batch_size_;
};

template <typename Device, typename T>
class RoiUnpoolOpGrad : public OpKernel {
 public:
  explicit RoiUnpoolOpGrad(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("data_height", &data_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, data_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        data_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("data_width", &data_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, data_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        data_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));

    // Get the batch size
    OP_REQUIRES_OK(context,
                   context->GetAttr("batch_size", &batch_size_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& bottom_feat = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    // be of the original size
    const Tensor& top_diff = context->input(2);

    // data should have 4 dimensions.
    OP_REQUIRES(context, top_diff.dims() == 4,
                errors::InvalidArgument("top diff must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // Number of ROIs
    const int num_rois = bottom_rois.dim_size(0);
    // pool height
    const int pooled_height = bottom_feat.dim_size(1);
    // pool width
    const int pooled_width = bottom_feat.dim_size(2);
    // Number of channels
    const int num_channels = bottom_feat.dim_size(3);

    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[1] = pooled_height;
    dims[2] = pooled_width;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    // Create output tensors
    Tensor* bottom_diff = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &bottom_diff));

    auto bottom_rois_flat = bottom_rois.flat<T>();
    auto top_diff_flat = top_diff.flat<T>();
    auto bottom_diff_flat = bottom_diff->flat<T>();

    // Set the gradients to be zero
    // const int N = bottom_diff_flat.size();
    // for (int i = 0; i < N; i++) {
    //   bottom_diff_flat(i) = 0;
    // }
    bottom_diff->tensor<T, 4>().setZero();

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
      CHECK_LT(roi_batch_ind, batch_size_);

      int roi_height = max(roi_end_h - roi_start_h + 1, 1);
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      const T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
      const T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

      const int index_data = roi_batch_ind * data_height_ * data_width_ * num_channels;

      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * roi_height / pooled_height)
          //  end (excluded) = ceil((ph + 1) * roi_height / pooled_height)
          int hstart = static_cast<int>(floor(static_cast<T>(ph)
                                              * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                              * bin_size_w));
          int hend = static_cast<int>(ceil(static_cast<T>(ph + 1)
                                           * bin_size_h));
          int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                           * bin_size_w));

          hstart = min(max(hstart + roi_start_h, 0), data_height_);
          hend = min(max(hend + roi_start_h, 0), data_height_);
          wstart = min(max(wstart + roi_start_w, 0), data_width_);
          wend = min(max(wend + roi_start_w, 0), data_width_);

          T this_size = static_cast<T>((hend - hstart) * (wend - wstart));
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          if (is_empty) {
            continue;
          }

          const int pool_index = index_output + (ph * pooled_width + pw) * num_channels;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              for (int c = 0; c < num_channels; ++c) {
                const int index = (h * data_width_ + w) * num_channels + c;
                bottom_diff_flat(pool_index + c) += top_diff_flat(index_data + index) / this_size;
              }
            }
          }
        }
      }
      // Increment ROI index
      index_roi += bottom_rois.dim_size(1);
      index_output += pooled_height * pooled_width * num_channels;
    }
  }
 private:
  int data_height_;
  int data_width_;
  float spatial_scale_;
  int batch_size_;
};

#define REGISTER_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(Name("RoiUnpooling")                    \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T"),            \
                          RoiUnpoolOp<CPUDevice, T>);             \
  REGISTER_KERNEL_BUILDER(Name("RoiUnpoolingGrad")                \
                              .Device(DEVICE_CPU)                 \
                              .TypeConstraint<T>("T"),            \
                          RoiUnpoolOpGrad<CPUDevice, T>);

TF_CALL_float(REGISTER_KERNEL);
TF_CALL_double(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#if GOOGLE_CUDA

template <class T>
class RoiUnpoolGPUOp : public OpKernel {
 public:
  explicit RoiUnpoolGPUOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("data_height", &data_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, data_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        data_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("data_width", &data_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, data_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        data_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));

    // Get the batch size
    OP_REQUIRES_OK(context,
                   context->GetAttr("batch_size", &batch_size_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    // bottom feat is the new feature to put it
    const Tensor& bottom_feat = context->input(0);
    const Tensor& bottom_rois = context->input(1);

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // bottom feature should have 4 dimensions
    OP_REQUIRES(context, bottom_feat.dims() == 4,
                errors::InvalidArgument("bottom feat must be 4-dimensional"));

    // Number of ROIs
    const int num_rois = bottom_rois.dim_size(0);
    // pool height
    const int pooled_height = bottom_feat.dim_size(1);
    // pool width
    const int pooled_width = bottom_feat.dim_size(2);
    // Number of channels
    const int num_channels = bottom_feat.dim_size(3);

    // construct the output shape
    int dims[4];
    dims[0] = batch_size_;
    dims[1] = data_height_;
    dims[2] = data_width_;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    Tensor* top_data = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &top_data));

    bool status = RoiUnpoolForwardGPU(
      bottom_feat.flat<T>().data(), spatial_scale_, batch_size_, num_rois, data_height_,
      data_width_, num_channels, pooled_height, pooled_width, bottom_rois.flat<T>().data(),
      top_data->flat<T>().data(), context->eigen_gpu_device());

    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching Roi pooling layer"));
    }
  }
 private:
  int data_height_;
  int data_width_;
  float spatial_scale_;
  int batch_size_;
};

#define REGISTER_KERNEL(T)                                        \
  REGISTER_KERNEL_BUILDER(Name("RoiUnpooling")                    \
                              .Device(DEVICE_GPU)                 \
                              .TypeConstraint<T>("T"),            \
                          RoiUnpoolGPUOp<T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_KERNEL);

#undef REGISTER_KERNEL

template <class T>
class RoiUnpoolGPUOpGrad : public OpKernel {
 public:
  explicit RoiUnpoolGPUOpGrad(OpKernelConstruction* context) : OpKernel(context) {
    // Get the pool height
    OP_REQUIRES_OK(context,
                   context->GetAttr("data_height", &data_height_));
    // Check that pooled_height is positive
    OP_REQUIRES(context, data_height_ >= 0,
                errors::InvalidArgument("Need pooled_height >= 0, got ",
                                        data_height_));
    // Get the pool width
    OP_REQUIRES_OK(context,
                   context->GetAttr("data_width", &data_width_));
    // Check that pooled_width is positive
    OP_REQUIRES(context, data_width_ >= 0,
                errors::InvalidArgument("Need pooled_width >= 0, got ",
                                        data_width_));
    // Get the spatial scale
    OP_REQUIRES_OK(context,
                   context->GetAttr("spatial_scale", &spatial_scale_));

    // Get the batch size
    OP_REQUIRES_OK(context,
                   context->GetAttr("batch_size", &batch_size_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& bottom_feat = context->input(0);
    const Tensor& bottom_rois = context->input(1);
    // be of the original size
    const Tensor& top_diff = context->input(2);

    // data should have 4 dimensions.
    OP_REQUIRES(context, top_diff.dims() == 4,
                errors::InvalidArgument("top diff must be 4-dimensional"));

    // rois should have 2 dimensions.
    OP_REQUIRES(context, bottom_rois.dims() == 2,
                errors::InvalidArgument("rois must be 2-dimensional"));

    // Number of ROIs
    const int num_rois = bottom_rois.dim_size(0);
    // pool height
    const int pooled_height = bottom_feat.dim_size(1);
    // pool width
    const int pooled_width = bottom_feat.dim_size(2);
    // Number of channels
    const int num_channels = bottom_feat.dim_size(3);

    // construct the output shape
    int dims[4];
    dims[0] = num_rois;
    dims[1] = pooled_height;
    dims[2] = pooled_width;
    dims[3] = num_channels;
    TensorShape output_shape;
    TensorShapeUtils::MakeShape(dims, 4, &output_shape);

    // Create output tensors
    Tensor* bottom_diff = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &bottom_diff));

    bool status = RoiUnpoolBackwardGPU(
        top_diff.flat<T>().data(), spatial_scale_, num_rois, data_height_,
        data_width_, num_channels, pooled_height, pooled_width, bottom_rois.flat<T>().data(),
        bottom_diff->flat<T>().data(), context->eigen_gpu_device());

    if (!status) {
      context->SetStatus(
          errors::Internal("Failed launching RoiUnpoolingGrad"));
    }
  }
 private:
  int data_height_;
  int data_width_;
  float spatial_scale_;
  int batch_size_;
};

#define REGISTER_KERNEL(T)                                           \
  REGISTER_KERNEL_BUILDER(Name("RoiUnpoolingGrad")                   \
                            .Device(DEVICE_GPU)                      \
                            .TypeConstraint<T>("T"),                 \
                          RoiUnpoolGPUOpGrad<T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#endif  // GOOGLE_CUDA

} // namespace tensorflow